from typing import Optional, Tuple, Literal, List
import torch
from torch import nn as nn
from torch.nn.functional import pad as torchpad

from patchldmseg.model.basic.conv import BigGANResBlock, ResidualBlock, ConvolutionalBlock
from patchldmseg.model.basic.attention import AttentionBlock


class EmbeddingSequential(nn.Sequential):
    r"""Sequential block that allows the utilisation
    with more than just x as a parameter"""

    def forward(self, x: torch.Tensor, emb: Optional[torch.Tensor] = None):
        for layer in self:
            x = layer(x, emb)
        return x

    @property
    def out_channels(self) -> int:
        return [layer for layer in self][-1].out_channels


class DiffusionBlock(nn.Module):
    r"""Diffusion Block of  UNet Encoder or Decoder composed of one residual block
    with two convolutions, optional attention and time_embedding.
    It roughly follows the BigGAN architecture utilised
    by Song et al. (https://arxiv.org/abs/2011.13456) and
    Dhariwal et al. (https://arxiv.org/abs/2105.05233).

    Parameters
    ----------
    activation : str, optional
        The name of the activation function to be used. MUST match the exact spelling of torch
    act_args : tuple
        Additional arguments of the activation function
    attention_ch_per_head : int
        Number of channels per attention head. If this is set to None, attention_heads determines
        the number of attention heads. If it is set to an integer, number of heads
        are determined automatically.
    attention_heads : int
        Number of attention heads used in the AttentionBlock
    attention_res : tuple, optional
        Attention resolutions i.e. where attention should take place.
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
        The string of the normalization method to be used. Supported are [batch, instance, layer, group]
    out_channels : int
        The output channels after both convolutions and the pooling operation. The first block currently
        creates them, whereas the others maintain the feature dimension.
    emb_ch: int
        The output channels of the initial embedding that is created in the respective DiffusionUNet,
        i.e. a child class of UNetBase
    padding : bool
        Whether padding should be used or not. The underlying function determines the optimal padding
    padding_mode : str
        The padding mode, i.e. which values to be used for padding
    patch_size : int
        The patch size of the input patch of this block. This is required as attention
        is utilised as attention is performed only on specific attention resolutions,
        specified by 'att_res'
    preactivation : bool
        Whether the activation should precede the convolution
    use_biggan: bool
        Whether the BigGanResidualBlock should be used or the conventiaonal residual block
    time_emb: bool  
        Whether the time embedding should be used or not
    """

    def __init__(self,
                 dimensions: int,
                 in_channels: int,
                 out_channels: int,
                 activation: Optional[str] = 'ReLU',
                 act_args: tuple = (False,),
                 attention_heads: int = 1,
                 attention_ch_per_head: Optional[int] = None,
                 attention_res: Optional[Tuple[int, ...]] = (32, 16, 8),
                 conv_zero_init: bool = False,
                 dropout: int = 0,
                 kernel_size: int = 3,
                 norm: Optional[Literal['batch', 'instance', 'group', 'layer', 'ada_gn']] = 'group',
                 emb_ch: Optional[int] = None,
                 padding: bool = True,
                 padding_mode: str = 'zeros',
                 patch_size: Optional[int] = None,
                 preactivation: bool = False,
                 use_biggan: bool = True,
                 time_emb: bool = True,
                 ):
        super().__init__()

        res_block = BigGANResBlock if use_biggan else ResidualBlock

        diffusion_block = nn.ModuleList([res_block(
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
            dropout=dropout,
            time_emb=time_emb,
            emb_ch=emb_ch,
            conv_zero_init=conv_zero_init
        )])

        in_channels = out_channels

        if attention_res is not None and patch_size in attention_res:
            _error_msg = f"'patch_size' is {patch_size}, " \
                         f"'att_res' is {attention_res} and" \
                         f"'attention_heads' is {attention_heads}.\nAll of them need to specified"
            assert patch_size is not None and attention_res and attention_heads, _error_msg
            diffusion_block.append(AttentionBlock(
                in_channels=in_channels,
                attention_ch_per_head=attention_ch_per_head,
                dimensions=dimensions,
                attention_heads=attention_heads,
                conv_zero_init=conv_zero_init
            ))

        self.diffusion_block = EmbeddingSequential(*diffusion_block)
        self._out_channels = out_channels

    def forward(self,
                x: torch.Tensor,
                t_emb: Optional[torch.Tensor] = None):
        return self.diffusion_block(x, t_emb)

    @property
    def out_channels(self):
        return self._out_channels


class DiffusionUNetEncoder(nn.Module):
    r"""Encoder class of the Diffusion UNet with time embedding and attention.
    Convolutions are carried out as ResidualBlocks.
    The Encoder also includes the initial convolution to increase the feature channel
    size

    Parameters
    ---------
    activation : str, optional
        The name of the activation function to be used. MUST match the exact spelling of torch
    act_args : tuple
        Additional arguments of the activation function
    attention_ch_per_head : int
        Number of channels per attention head. If this is set to None, attention_heads determines
        the number of attention heads. If it is set to an integer, number of heads
        are determined automatically.
    attention_heads : int
        Number of attention heads used in the AttentionBlock
    attention_res : tuple
        Attention resolutions i.e. where attention should take place.
    channel_factor : int
        The multiplicative factor of the channel dimension. It is a multiplicative factor for each
        depth layer of the Encoder. The input channels are multiplied by this factor in each EncodingBlock.
    conv_zero_init : bool
        If certain output layers should be initialised with zero weight and bias similar to Dhariwal.
    dimensions : int
        Dimensionality of the input i.e. whether 1D, 2D or 3D operations should be used
    dropout : int
        Percentage of dropout to be used [0, 100]
    emb_ch: int
        The output channels of the initial embedding that is created in the respective DiffusionUNet,
        i.e. a child class of UNetBase
    hidden_channels: int
        The hidden channels of the initial convolution to increase the feature channels.
    kernel_size : int
        The size of the kernel for convolution
    in_channels : int
        Number of input channels to the first convolution of the entire block
    norm : str, optional
        The string of the normalization method to be used. Supported are [batch, instance, layer, group]
    num_res_blocks : int
        Number of residual blocks (with optional attention) per layer.
    padding : bool
        Whether padding should be used or not. The underlying function determines the optimal padding
    padding_mode : str
        The padding mode, i.e. which values to be used for padding
    patch_size : int
        The patch size of the input patch i.e. the initial patch size prior to the UNet.
        This is required as attention is utilised as attention is performed only on specific
        attention resolutions, specified by 'att_res'
    preactivation : bool
        Whether the activation should precede the convolution. Preactivation would result in a convolution
        of Norm -> Activation -> Convolution. Otherwise, Convolution -> Norm -> Activation
    spatial_factor : int
        The reduction of the spatial dimension by the pooling layer.
    upd_conv : str
        If the model should use convolution for upsampling/downsampling or traditional pooling/interpolation
    """
    # NOTE: Currently only used for diffusion! Does not work otherwise
    def __init__(self,
                 dimensions: int,
                 hidden_channels: int,
                 in_channels: int,
                 patch_size: int,
                 activation: Optional[str] = 'ReLU',
                 act_args: tuple = (False,),
                 attention_ch_per_head: Optional[int] = None,
                 attention_heads: int = 1,
                 attention_res: Optional[Tuple[int, ...]] = (32, 16, 8),
                 channel_factor: Tuple[int, ...] = (1, 2, 4, 8),
                 conv_zero_init: bool = False,
                 dropout: int = 0,
                 emb_ch: Optional[int] = None,
                 kernel_size: int = 3,
                 norm: Optional[Literal['batch', 'instance', 'group', 'layer', 'ada_gn']] = 'batch',
                 num_res_blocks: int = 1,
                 padding: bool = True,
                 padding_mode: str = 'zeros',
                 preactivation: bool = False,
                 spatial_factor: int = 2,
                 upd_conv: Optional[bool] = True
                 ):
        super().__init__()

        self._sc_ch = []

        # Initial convolution
        out_channels = hidden_channels
        self.encoder_blocks = nn.ModuleList([
            ConvolutionalBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                dimensions=dimensions,
                kernel_size=kernel_size,
                norm=None,
                preactivation=False,
                padding=True,
                activation=None,
                dropout=0,
                dilation=None,
                stride=1)])
        self._sc_ch.append(hidden_channels)

        self.ch_in_out = {}

        for encoder_depth, enc_ch_factor in enumerate(channel_factor):
            in_channels = self.encoder_blocks[-1].out_channels
            out_channels = hidden_channels * enc_ch_factor
            for _ in range(num_res_blocks):
                self.encoder_blocks.append(
                    DiffusionBlock(
                        dimensions=dimensions,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        activation=activation,
                        act_args=act_args,
                        attention_ch_per_head=attention_ch_per_head,
                        attention_heads=attention_heads,
                        attention_res=attention_res,
                        conv_zero_init=conv_zero_init,
                        dropout=dropout,
                        kernel_size=kernel_size,
                        norm=norm,
                        emb_ch=emb_ch,
                        padding=padding,
                        padding_mode=padding_mode,
                        patch_size=patch_size,
                        preactivation=preactivation,
                    ))
                self._sc_ch.append(out_channels)

                self.ch_in_out[f'res_{encoder_depth}_{_}'] = (
                    in_channels, out_channels)

                in_channels = out_channels

            # Pooling
            if encoder_depth != len(channel_factor) - 1:
                self.encoder_blocks.append(
                    BigGANResBlock(
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
                        spatial_factor=spatial_factor,
                        time_emb=True,
                        emb_ch=emb_ch,
                        upd_conv=upd_conv,
                        up_or_down='down',
                        conv_zero_init=conv_zero_init))
                self._sc_ch.append(out_channels)
                patch_size //= spatial_factor

                self.ch_in_out[f'res_{encoder_depth}'] = (
                    in_channels, out_channels)

        self._patch_size = patch_size

    def forward(self,
                x: torch.Tensor,
                time_emb: Optional[torch.Tensor] = None):
        skip_connections = []
        h = x
        for encoder_block in self.encoder_blocks:
            if isinstance(encoder_block, ConvolutionalBlock):
                # NOTE: in_conv is in encoder_blocks and does not have any t_emb
                h = encoder_block(h)
            else:
                h = encoder_block(h, time_emb)
            skip_connections.append(h)

        return h, skip_connections

    @property
    def out_channels(self) -> int:
        return self.encoder_blocks[-1].out_channels

    @property
    def out_patch_size(self) -> int:
        r"""Returns the output patch size after the entire encoder took place"""
        return self._patch_size

    @property
    def skip_connection_out_ch(self) -> List[int]:
        r"""Returns the out_channels for the encoder for each layer in it."""
        return self._sc_ch


class DiffusionUNetDecoder(nn.Module):
    r"""Decoder class of the UNet with time embedding and attention.
    Convolutions are carried out as ResidualBlocks

    Parameters
    ---------
    activation : str, optional
        The name of the activation function to be used. MUST match the exact spelling of torch
    act_args : tuple
        Additional arguments of the activation function
    attention_ch_per_head : int
        Number of channels per attention head. If this is set to None, attention_heads determines
        the number of attention heads. If it is set to an integer, number of heads
        are determined automatically.
    attention_heads : int
        Number of attention heads used in the AttentionBlock
    attention_res : tuple
        Attention resolutions i.e. where attention should take place.
    channel_factor : int
        The multiplicative factor of the channel dimension. It is a multiplicative factor for each
        depth layer of the Encoder. The input channels are multiplied by this factor in each EncodingBlock.
    concat_sc : bool
        If True, the skip connections of the encoder will be concatenated along the channel dimension
        in the decoder. If False, the skip connections will be summed with the features of the upsampled layer
    conv_zero_init : bool
        If certain output layers should be initialised with zero weight and bias similar to Dhariwal.
    dimensions : int
        Dimensionality of the input i.e. whether 1D, 2D or 3D operations should be used
    dropout : int
        Percentage of dropout to be used [0, 100]
    emb_ch: int
        The output channels of the initial embedding that is created in the respective DiffusionUNet,
        i.e. a child class of UNetBase
    hidden_channels: int
        The hidden channels of the initial convolution used in the encoder to increase the feature channels.
    in_channels : int
        Number of input channels to the first convolution of decoder. Is typically the output channel size
        of the last layer of the encoder. Skip connections are solved internally
    kernel_size : int
        The size of the kernel for convolution
    norm : str, optional
        The string of the normalization method to be used. Supported are [batch, instance, layer, group]
    num_res_blocks : int
        Number of residual blocks (with optional attention) per layer.
    padding : bool
        Whether padding should be used or not. The underlying function determines the optimal padding
    padding_mode : str
        The padding mode, i.e. which values to be used for padding
    patch_size : int
        The patch size of the input patch i.e. the initial patch size prior to the UNet.
        This is required as attention is utilised as attention is performed only on specific
        attention resolutions, specified by 'att_res'
    preactivation : bool
        Whether the activation should precede the convolution. Preactivation would result in a convolution
        of Norm -> Activation -> Convolution. Otherwise, Convolution -> Norm -> Activation
    sc_ch_enc: list
        Skip-connection channels from the encoder. Used to determine the input channel size
        after concatenating the skip_connections
    spatial_factor : int
        The reduction of the spatial dimension by the pooling layer.
    upd_conv : str
        If the model should use convolution for upsampling/downsampling or traditional pooling/interpolation
    """
    def __init__(self,
                 dimensions: int,
                 in_channels: int,
                 hidden_channels: int,
                 sc_ch_enc: List[int],
                 activation: Optional[str] = 'ReLU',
                 act_args: tuple = (False,),
                 attention_ch_per_head: Optional[int] = None,
                 attention_heads: int = 1,
                 attention_res: Optional[Tuple[int, ...]] = (32, 16, 8),
                 channel_factor: Tuple[int, ...] = (1, 2, 4, 8),
                 conv_zero_init: bool = False,
                 dropout: int = 0,
                 emb_ch: Optional[int] = None,
                 kernel_size: int = 3,
                 norm: Optional[Literal['batch', 'instance', 'group', 'layer', 'ada_gn']] = 'group',
                 num_res_blocks: int = 1,
                 padding: bool = True,
                 padding_mode: str = 'zeros',
                 patch_size: Optional[int] = None,
                 preactivation: bool = False,
                 spatial_factor: int = 2,
                 upd_conv: Optional[bool] = True
                 ):
        super().__init__()
        self.decoder_blocks = nn.ModuleList()

        self.ch_in_out = {}

        for decoder_depth, dec_ch_factor in enumerate(reversed(channel_factor)):
            for _ in range(num_res_blocks + 1):
                in_channels = in_channels + sc_ch_enc.pop()
                out_channels = hidden_channels * dec_ch_factor
                self.decoder_blocks.append(DiffusionBlock(
                    dimensions=dimensions,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    activation=activation,
                    act_args=act_args,
                    attention_ch_per_head=attention_ch_per_head,
                    attention_heads=attention_heads,
                    attention_res=attention_res,
                    conv_zero_init=conv_zero_init,
                    dropout=dropout,
                    kernel_size=kernel_size,
                    norm=norm,
                    emb_ch=emb_ch,
                    padding=padding,
                    padding_mode=padding_mode,
                    patch_size=patch_size,
                    preactivation=preactivation
                ))
                self.ch_in_out[f'res_{len(channel_factor) - 1 - decoder_depth}_{_}'] = (
                    in_channels, out_channels)
                in_channels = out_channels

            # Upsampling
            if decoder_depth != len(channel_factor) - 1:
                self.decoder_blocks.append(
                    BigGANResBlock(
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
                        spatial_factor=spatial_factor,
                        time_emb=True,
                        emb_ch=emb_ch,
                        upd_conv=upd_conv,
                        up_or_down='up',
                        conv_zero_init=conv_zero_init
                    ))
                patch_size *= 2

                self.ch_in_out[f'Up_{len(channel_factor) - 1 - decoder_depth}'] = (
                    in_channels, out_channels)

    def forward(self,
                x: torch.Tensor,
                skip_connections: torch.Tensor,
                t_emb: Optional[torch.Tensor] = None):

        h = x
        for decoder_block in self.decoder_blocks:
            if isinstance(decoder_block, DiffusionBlock):
                # Only use the skip connections in the DiffusionBlocks but not in the upsampling
                assert h.size()[2:] == skip_connections[-1].size()[2:], \
                    "skip connection concat requires same spatial size"
                h = torch.cat([h, skip_connections.pop()], dim=1)
            h = decoder_block(h, t_emb)

        return h

    @property
    def out_channels(self) -> int:
        return self.decoder_blocks[-1].out_channels

    @staticmethod
    def center_crop(skip_connection, x):
        skip_shape = torch.Tensor(skip_connection.shape)
        x_shape = torch.Tensor(x.shape)
        crop = skip_shape[2:] - x_shape[2:]
        half_crop_1 = torch.floor(crop / 2.0).type(torch.int)
        half_crop_2 = torch.ceil(crop / 2.0).type(torch.int)
        # If skip_connection is 10, 20, 30 and x is (6, 14, 12)
        # Then pad will be (-2, -2, -3, -3, -9, -9)
        # Pytorch needs reversed order for pad function
        pad = -torch.stack((half_crop_1, half_crop_2)).t().flatten().flip(0)
        skip_connection = torchpad(skip_connection, pad.tolist())
        # NOTE: if x is bigger than skip_con, this can lead to an enlarged feature map ...
        # This can mess up the alignment with the target label mask.
        return skip_connection


