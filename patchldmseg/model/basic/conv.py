from typing import Any, Mapping, Optional, Tuple, Literal, Union
import torch.nn as nn
import torch.nn.functional as F
import torch
from inspect import signature

from patchldmseg.utils import constants, misc

class PaddingLayer(nn.Module):
    r"""Padding layer to handle the padding for uneven kernel sizes and strides.
    This replaces the inherent padding of a nn.ConvXd layer."""

    def __init__(self,
                 padding: Tuple[int, ...],
                 mode,):
        
        super().__init__()
        self._padding = padding
        self._mode = mode

    def forward(self, x):
        return F.pad(x, self._padding, mode=self._mode)


class ConvolutionalBlock(nn.Module):
    r"""Base class for convolutions uniting normalisation, activation, convolution and dropout
    in a single interface.

    Parameters
    ---------
    activation : str, optional
        The name of the activation function to be used. MUST match the exact spelling of torch
    act_args : tuple
        Additional arguments of the activation function
    conv_zero_init : bool
        If certain output layers should be initialised with zero weight and bias similar to Dhariwal.
    dimensions : int
        Dimensionality of the input i.e. whether 1D, 2D or 3D operations should be used
    dropout : int
        Percentage of dropout to be used [0, 100]
    kernel_size : int
        The size of the kernel for convolution
    in_channels : int
        Number of input channels to the first convolution of the entire block
    norm : str, optional
        The string of the normalisation method to be used. Supported are [batch, instance, layer, group]
    out_channels: int
        The desired output channels of the convolution
    padding : bool
        Whether padding should be used or not. The underlying function determines the optimal padding
    padding_mode : str
        The padding mode, i.e. which values to be used for padding
    preactivation : bool
        Whether the activation should precede the convolution. Preactivation would result in a convolution
        of Norm -> Activation -> Convolution. Otherwise, Convolution -> Norm -> Activation
    stride: int
        If a strided convolution should be used with the respective stride. Stride 1 does not downsample.
    dilation: int
        If a dilated convolution should be used. Currently, not really supported.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            dimensions: int,
            kernel_size: int = 3,
            norm: Optional[Literal['batch', 'layer', 'instance', 'group', 'ada_gn']] = 'batch',
            activation: Optional[str] = 'ReLU',
            act_args: Optional[tuple] = (False,),
            preactivation: Optional[bool] = False,
            padding: bool = False,
            padding_mode: Literal['constant', 'reflect', 'replicate', 'zeros'] = 'zeros',
            stride: int = 1,
            dilation: Optional[int] = None,
            dropout: Optional[int] = 0,
            conv_zero_init: bool = False
    ):
        assert padding_mode in ['constant', 'reflect', 'replicate', 'zeros'], f"{padding_mode} not supported"
        assert dilation is None or dilation == 1, "Dilation not supported at the moment"
        dilation = dilation or 1

        super().__init__()
        self._out_channels = out_channels
        self._preactivation = preactivation
        self._norm = None
        self._rest = None

        conv_layer = self.conv_layer(
            in_channels=in_channels,
            out_channels=out_channels,
            dimensions=dimensions,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            conv_zero_init=conv_zero_init
            )
        
        # Automatic Padding
        pad_input = self.determine_padding(
            dimensions=dimensions,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation)
        # constant padding mode is automatic zero padding as value in F.pad is 0  
        padding_mode = padding_mode if padding_mode != 'zeros' else 'constant'

        pad_layer = PaddingLayer(padding=pad_input, mode=padding_mode)

        # Normalisation
        if norm is not None:
            num_channels = in_channels if preactivation else out_channels
            if norm == 'batch':
                class_name = f'{norm.capitalize()}Norm{dimensions}d'
                norm_class = getattr(nn, class_name)
                norm_layer = norm_class(num_channels)

            else:
                class_name = f'GroupNorm'
                norm_class = getattr(nn, class_name)

                # taken from https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html
                if norm == 'instance':
                    groups = num_channels
                elif norm == 'layer':
                    groups = 1
                elif norm in (constants.ADAPTIVE_GROUP_NORM, 'group'):
                    groups = 32
                else:
                    raise RuntimeError("Wrong norm")

                try:
                    norm_layer = norm_class(groups, num_channels)
                except ValueError as e:
                    raise ValueError(e)
        else:
            norm_layer = nn.Identity()

        activation_layer = get_activation_layer(activation=activation,
                                                act_args=act_args)

        dropout_layer = nn.Dropout(p=dropout/100) if dropout else nn.Identity()

        # Sequencing
        if preactivation:
            module_list = nn.ModuleList(
                [norm_layer, activation_layer, dropout_layer, pad_layer, conv_layer])

        else:
            module_list = nn.ModuleList(
                [pad_layer, conv_layer, norm_layer, activation_layer, dropout_layer])
            
        self.block = nn.Sequential(*module_list)

    def __getitem__(self, item):
        r"""Index the respective item of the sequence"""
        return self.block[item]

    def forward(self, x) -> torch.Tensor:
        return self.block(x)
    
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        return super().load_state_dict(state_dict, strict)

    @property
    def out_channels(self):
        return self._out_channels
    
    @staticmethod
    def determine_padding(
        dimensions: int, 
        kernel_size: int, 
        padding: bool, 
        stride: int,
        dilation: int) -> Tuple[int, ...]:
                
        if padding:
            # Preserve spatial dimensions
            total_pad = dilation*(kernel_size - 1) + 1 - stride

            pad_input = (total_pad // 2 + total_pad % 2, total_pad // 2) * dimensions
        else:
            pad_input = (0, 0) * dimensions
        
        return pad_input

    @staticmethod
    def conv_layer(
            in_channels: int,
            out_channels: int,
            dimensions: int,
            kernel_size: int = 3,
            padding: bool = False,
            stride: int = 1,
            dilation: int = 1,
            conv_zero_init: bool = False) -> Union[torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d]:

        class_name = f'Conv{dimensions}d'
        conv_class = getattr(nn, class_name)
        conv_layer = conv_class(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
        )

        if conv_zero_init:
            conv_layer = misc.zero_init(conv_layer)
        return conv_layer


class ConvolutionalTransposeBlock(ConvolutionalBlock):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            dimensions: int,
            kernel_size: int = 3,
            norm: Optional[Literal['batch', 'layer', 'instance', 'group', 'ada_gn']] = 'batch',
            activation: Optional[str] = 'ReLU',
            act_args: Optional[tuple] = (False,),
            preactivation: Optional[bool] = False,
            padding: bool = False,
            padding_mode: Literal['constant', 'reflect', 'replicate', 'zeros'] = 'zeros',
            stride: int = 1,
            dilation: Optional[int] = None,
            dropout: int = 0,
            conv_zero_init: bool = False
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            dimensions=dimensions,
            kernel_size=kernel_size,
            norm=norm,
            activation=activation,
            act_args=act_args,
            preactivation=preactivation,
            padding=padding,
            padding_mode=padding_mode,
            stride=stride,
            dilation=dilation,
            dropout=dropout,
            conv_zero_init=conv_zero_init)

    @staticmethod
    def conv_layer(in_channels: int,
                       out_channels: int,
                       dimensions: int,
                       kernel_size: int = 3,
                       padding: bool = False,
                       padding_mode:Literal['zeros', 'reflect', 'replicate', 'circular'] = 'zeros',
                       stride: int = 1,
                       dilation: int = 1,
                       conv_zero_init: bool = False) -> nn.Sequential:

        if padding:
            # This is the counteract the automatic padding in ConvTranspose as I am already padding
            padding_val = (kernel_size - 1) * dilation
            output_padding = kernel_size % 2 if (stride > 1 or dilation > 1) else 0
        else:
            padding_val = 0
            output_padding = 0

        class_name = f'ConvTranspose{dimensions}d'
        conv_class = getattr(nn, class_name)
        conv_layer = conv_class(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding_val,
            dilation=dilation,
            output_padding=output_padding
        )
        if conv_zero_init:
            conv_layer = misc.zero_init(conv_layer)
        return conv_layer
    
    @staticmethod
    def check_spatial(x: torch.Tensor, h: torch.Tensor, stride: int):
        x_spat, h_spat = x.shape[2:], h.shape[2:]

        for x_s, h_s in zip(x_spat, h_spat):
            assert x_s * stride == h_s, f"Spatial dimension mismatch: {x_s} vs {h_s}"


class ResidualBlock(nn.Module):
    r"""Classic residual block with skip connection and time embedding"""

    def __init__(self,
                 dimensions: int,
                 in_channels: int,
                 out_channels: int,
                 activation: Optional[str] = 'ReLU',
                 act_args: tuple = (False,),
                 conv_zero_init: bool = False,
                 dropout: int = 0,
                 kernel_size: int = 3,
                 norm: Optional[Literal['batch', 'layer', 'instance', 'group', 'ada_gn']] = 'group',
                 emb_ch: Optional[int] = None,
                 padding: bool = True,
                 padding_mode: str = 'zeros',
                 preactivation: bool = False,
                 time_emb: bool = False,
                 ):
        super().__init__()

        self._out_ch = out_channels

        self.conv1 = ConvolutionalBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            dimensions=dimensions,
            kernel_size=kernel_size,
            norm=norm,
            activation=activation,
            act_args=act_args,
            preactivation=preactivation,
            padding=padding,
            padding_mode=padding_mode,
            stride=1,
            dropout=0)  # No dropout as I am using dropout when I combine the time emb in the out conv

        self.conv2 = ConvolutionalBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            dimensions=dimensions,
            kernel_size=kernel_size,
            norm=norm,
            activation=activation,
            act_args=act_args,
            preactivation=preactivation,
            padding=padding,
            padding_mode=padding_mode,
            stride=1,
            dropout=dropout,
            conv_zero_init=conv_zero_init
        )

        # Time Embedding
        if time_emb:
            _error_msg = f"'emb_ch' is {emb_ch}.\n" \
                         f"Make sure to specify it in the constructor"
            assert emb_ch is not None, _error_msg
            #out_features = 2 * out_channels if self.use_ada_gn else out_channels

            # Dense Layer
            self.emb_layer = nn.Sequential(
                ConvolutionalBlock(
                    in_channels=emb_ch,
                    out_channels=out_channels,
                    dimensions=dimensions,
                    kernel_size=1,
                    norm=None,
                    activation=activation,
                    preactivation=preactivation,
                    dropout=0
                )
            )

        self.res = nn.Identity() if in_channels == out_channels else ConvolutionalBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            dimensions=dimensions,
            kernel_size=1,
            norm=None,
            activation=None,
            preactivation=preactivation
        )

    @property
    def out_channels(self) -> int:
        return self._out_ch

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None, y: Optional[torch.Tensor] = None):
        res_con = self.res(x)
        h = self.conv1(x)

        if t is not None:
            try:
                emb = self.emb_layer(t)
            except AttributeError as e:
                raise AttributeError(e)

            h = h + emb

        h = self.conv2(h)
        return h + res_con


class BigGANResBlock(nn.Module):
    r"""The entire BigGAN Residual Block. It contains two convolutions and a time_embedding
    between them. It partially resembles the implementation of BigGAN (https://arxiv.org/abs/1809.11096)
    but follows more closely the implementations by Song et al. (https://arxiv.org/abs/2011.13456)
    and Dhariwal et al. (https://arxiv.org/abs/2105.05233).
    Also incorporates the option for up/downsampling and adaptive group norm as described by Nichol et al.
    (https://arxiv.org/abs/2102.09672)
    """

    def __init__(
            self,
            dimensions: int,
            in_channels: int,
            out_channels: int,
            activation: Optional[str] = 'ReLU',
            act_args: tuple = (False,),
            conv_zero_init: bool = False,
            dropout: int = 0,
            kernel_size: int = 3,
            norm: Optional[Literal['batch', 'layer', 'instance', 'group', 'ada_gn']] = 'batch',
            emb_ch: Optional[int] = None,
            padding: bool = True,
            padding_mode: str = 'zeros',
            preactivation: bool = False,
            spatial_factor: Optional[int] = None,
            time_emb: bool = False,
            upd_conv: Optional[bool] = None,
            up_or_down: Optional[Literal['up', 'down']] = None,
    ):
        r"""Initialises the BigGAN Residual Block

        Parameters
        ----------
        in_channels : int
            The input channels of the tensor to be fed through the residual block
        conv_zero_init : bool
            If certain output layers should be initialised with zero weight and bias similar to Dhariwal.
        """
        super().__init__()

        self._out_ch = out_channels
        self.use_ada_gn = norm == constants.ADAPTIVE_GROUP_NORM
        self.up_or_down = up_or_down

        if self.use_ada_gn:
            assert preactivation, "Adaptive group norm with label embedding requires preactivtion = True"

        # The first 3x3 Convolution with activation
        self.conv1 = ConvolutionalBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            dimensions=dimensions,
            kernel_size=kernel_size,
            norm=norm,
            activation=activation,
            act_args=act_args,
            preactivation=preactivation,
            padding=padding,
            padding_mode=padding_mode,
            stride=1,
            dropout=0)

        # Convolution to combine the time embeddings with the hidden state, i.e. the second convolution
        self.conv2 = ConvolutionalBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            dimensions=dimensions,
            kernel_size=kernel_size,
            norm=norm,
            activation=activation,
            act_args=act_args,
            preactivation=preactivation,
            padding=padding,
            padding_mode=padding_mode,
            stride=1,
            dropout=dropout,
            conv_zero_init=conv_zero_init
        )

        # Time Embedding
        if time_emb:
            _error_msg = f"'emb_ch' is {emb_ch}.\n" \
                         f"Make sure to specify it in the constructor"
            assert emb_ch is not None, _error_msg
            out_features = 2 * out_channels if self.use_ada_gn else out_channels

            # Dense Layer
            self.emb_layer = nn.Sequential(
                ConvolutionalBlock(
                    in_channels=emb_ch,
                    out_channels=out_features,
                    dimensions=dimensions,
                    kernel_size=1,
                    norm=None,
                    activation=activation,
                    preactivation=preactivation,
                    dropout=0
                )
            )
        self.res_conv = ConvolutionalBlock(in_channels=in_channels,
                                           out_channels=out_channels,
                                           dimensions=dimensions,
                                           kernel_size=1,
                                           norm=None,
                                           activation=None,
                                           preactivation=preactivation
                                           )

        if up_or_down:
            assert spatial_factor is not None
            assert upd_conv is not None

            if up_or_down == 'up':
                self.res_upd = Upsample(in_channels=out_channels,
                                        out_channels=out_channels,
                                        dimensions=dimensions,
                                        kernel_size=kernel_size,
                                        spatial_factor=spatial_factor,
                                        upd_conv=upd_conv,
                                        norm=None,
                                        activation=None,
                                        act_args=None)
                h_upd = Upsample(in_channels=out_channels,
                                 out_channels=out_channels,
                                 dimensions=dimensions,
                                 kernel_size=kernel_size,
                                 spatial_factor=spatial_factor,
                                 upd_conv=upd_conv,
                                 norm=None,
                                 activation=None,
                                 act_args=None)
            else:
                self.res_upd = Downsample(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dimensions=dimensions,
                    spatial_factor=spatial_factor,
                    upd_conv=upd_conv,
                    norm=None,
                    activation=None,
                    act_args=None)
                h_upd = Downsample(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dimensions=dimensions,
                    spatial_factor=spatial_factor,
                    upd_conv=upd_conv,
                    norm=None,
                    activation=None,
                    act_args=None)
        else:
            self.res_upd = nn.Identity()
            h_upd = nn.Identity()

        self.conv1.block.insert(-2, h_upd)
        self.conv1.stride = getattr(h_upd, 'spatial_factor', 1) 

    def forward(self,
                x: torch.Tensor,
                t_emb: Optional[torch.Tensor] = None):
        residual = self.res_conv(self.res_upd(x))
        h = self.conv1(x)

        if t_emb is not None:
            try:
                emb = self.emb_layer(t_emb)
                self._check_emb(h=h, emb=emb)

                if self.use_ada_gn:
                    w, b = torch.chunk(emb, 2, dim=1)
                    h = self.conv2[0](h) * (w + 1) + b
                    h = self.conv2[1:](h)
                else:
                    h = h + emb
                    h = self.conv2(h)
            except AttributeError:
                raise AttributeError(f"Embedding is passed but the linear layer 'emb_layer' "
                                     f"does not exist as 'use_time_emb' is set to False.\n"
                                     f"Make sure to initialise the class correctly.")
        assert h.shape == residual.shape
        return h + residual

    @property
    def out_channels(self):
        return self._out_ch

    def _check_emb(self, h, emb):
        r"""Checks if the embedding and the hidden state match up"""

        def _error_msg(dim: str,
                       h_dim: int,
                       emb_dim: int):
            return f"{dim.capitalize()} dimension mismatch.\n" \
                   f"Hidden (`{h_dim}`) and Emb (`{emb_dim}`)"

        # Check Batch Dimension
        h_b, emb_b = h.shape[0], emb.shape[0]
        assert h_b == emb_b, _error_msg('Batch', h_b, emb_b)

        # Check channel dim
        h_c, emb_c = h.shape[1], emb.shape[1]
        if self.use_ada_gn:
            assert 2 * h_c == emb_c, _error_msg('Channel', 2 * h_c, emb_c)
        else:
            assert h_c == emb_c, _error_msg('Channel', h_c, emb_c)

        # Check spatial dimension
        h_spat, emb_spat = h.shape[2:], emb.shape[2:]
        assert len(h_spat) == len(emb_spat), _error_msg('Spatial', len(h_spat), len(emb_spat))
        assert all(_emb_spat == 1 for _emb_spat in emb_spat), f"Spatial dimension must be 1 for all not {h_spat}"


class Downsample(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dimensions: int,
                 upd_conv: bool = True,
                 kernel_size: int = 3,
                 norm: Optional[Literal['batch', 'layer', 'instance', 'group', 'ada_gn']] = 'batch',
                 activation: Optional[str] = 'ReLU',
                 act_args: Optional[tuple] = (False,),
                 preactivation: Optional[bool] = True,
                 padding: bool = True,
                 spatial_factor: int = 2,
                 dropout: int = 0):
        super().__init__()

        self._out_channels = out_channels
        self.spatial_factor = spatial_factor

        if upd_conv:
            self.down = ConvolutionalBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                dimensions=dimensions,
                kernel_size=kernel_size,
                norm=norm,
                activation=activation,
                act_args=act_args,
                preactivation=preactivation,
                padding=padding,
                dropout=dropout,
                stride=spatial_factor)
        else:
            pooling_fn = getattr(nn, f"AvgPool{dimensions}d")
            self.down = pooling_fn(kernel_size=spatial_factor)

    def forward(self,
                x: torch.Tensor):
        h = self.down(x)
        self.check_spatial(x=x, h=h, spatial_factor=self.spatial_factor)
        return h

    @property
    def out_channels(self):
        return self._out_channels
    
    @staticmethod
    def check_spatial(x: torch.Tensor, h: torch.Tensor, spatial_factor: int):
        x_spat, h_spat = x.shape[2:], h.shape[2:]

        for x_s, h_s in zip(x_spat, h_spat):
            assert x_s // spatial_factor == h_s, f"Spatial dimension mismatch: {x_s} vs {h_s}"


class Upsample(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dimensions: int,
                 upd_conv: bool = True,
                 kernel_size: int = 3,
                 norm: Optional[Literal['batch', 'layer', 'instance', 'group', 'ada_gn']] = 'batch',
                 activation: Optional[str] = 'ReLU',
                 act_args: Optional[tuple] = (False,),
                 preactivation: Optional[bool] = True,
                 padding: bool = True,
                 spatial_factor: int = 2,
                 dropout: int = 0):
        r"""Init Upsample Class

        Parameters
        ----------
        """

        super().__init__()

        self._out_channels = out_channels if upd_conv else in_channels
        self.spatial_factor = spatial_factor

        self.up = torch.nn.Sequential(Interpolate(spatial_factor=spatial_factor))
        if upd_conv:
            self.up.append(ConvolutionalBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                dimensions=dimensions,
                kernel_size=kernel_size,
                norm=norm,
                activation=activation,
                act_args=act_args,
                preactivation=preactivation,
                padding=padding)
            )

    def forward(self, x):
        h = self.up(x)
        self.check_spatial(x=x, h=h, spatial_factor=self.spatial_factor)
        return self.up(x)

    @property
    def out_channels(self):
        return self._out_channels
    
    @staticmethod
    def check_spatial(x: torch.Tensor, h: torch.Tensor, spatial_factor: int):
        x_spat, h_spat = x.shape[2:], h.shape[2:]

        for x_s, h_s in zip(x_spat, h_spat):
            assert int(x_s * spatial_factor) == h_s, f"Spatial dimension mismatch: {x_s} vs {h_s}"


class Interpolate(nn.Module):
    def __init__(self,
                 spatial_factor: int,
                 mode: str = "nearest"):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.spatial_factor = spatial_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.spatial_factor, mode=self.mode)
        return x


def get_activation_layer(activation: Optional[str], act_args: Optional[tuple]) -> nn.Module:
    r"""Obtains the activation layer from having a string 'activation' and the respective
    activation arguments.

    Returns
    -------
    torch.nn.Module
        The activation layer or None depending on the 'activation' string.
    """
    if activation is not None:
        activation_class = getattr(nn, activation)
        if act_args is not None:
            len_args = len(act_args)
            len_act = len(signature(activation_class).parameters)
            if not len_args == len_act:
                raise AttributeError(f"Param --act_args contains {len_args} parameters but {len_act} are required. "
                                    f"Make sure you include all necessary parameters with --act_args")
        else:
            act_args = tuple()
        return activation_class(*act_args)
    else:
        return nn.Identity()


def add_if_not_none(module_list, module: Optional[nn.Module]):
    r"""Adds a module if it is not None.
    Notes
    -----
    This is an inplace operation and passes the module list by reference. As a result,
    the list will be altered in the process of this function.
    """
    if module is not None and isinstance(module, nn.Module):
        module_list.append(module)
