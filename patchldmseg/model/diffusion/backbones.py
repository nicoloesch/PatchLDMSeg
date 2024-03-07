import torch.nn as nn
from typing import Optional, Tuple, Literal
from patchldmseg.model.diffusion.embedding import (
    TimeEmbedding, 
    PosEmbedding, 
    CoordinateEmbedding,
    LabelEmbedding
)
from patchldmseg.model.diffusion.diffusion_helper import VarianceType
from patchldmseg.model.diffusion.diffusion_model import (
    EmbeddingSequential, 
    DiffusionUNetDecoder,
    DiffusionUNetEncoder
)
from patchldmseg.model.basic.attention import AttentionBlock, AttentionPool
from patchldmseg.model.basic.conv import ConvolutionalBlock, BigGANResBlock, get_activation_layer
from patchldmseg.utils.misc import expand_dims
from patchldmseg.input.datasets import Dataset, BraTS2023
import torch

class UNetDiffusion(nn.Module):
    """UNet used for the Diffusion process. Inspired by Dhariwal et al. following a similar
    structure. Utilises timestep embedding.

    Parameters
    ----------
    activation : str, optional
        The name of the activation function to be used. MUST match the exact spelling of torch
    act_args : tuple
        Additional arguments of the activation function
    attention_heads : int
        Number of attention heads used in the AttentionBlock of the U-Net
    attention_res : tuple
        Attention resolutions i.e. where attention should take place based on the patch size.
        I.e. an attention_res of 16 would result to activate attention (if attention is True)
        as soon as the patch size reached 16^dim. If this is set to None, attention is not used.
    attention_ch_per_head : int
        Number of channels per attention head. If this is set to None, attention_heads determines
        the number of attention heads. If it is set to an integer, number of heads
        are determined automatically.
    channel_factor : tuple
        The multiplicative factor of the channel dimension. It is a multiplicative factor for each
        depth layer of the UNet. The input channels are multiplied by this factor.
    concat_sc : bool
        If True, the skip connections of the encoder will be concatenated along the channel dimension
        in the decoder. If False, the skip connections will be summed with the features of the upsampled layer
    conditional_sampling : bool
        If conditional sampling based on the label should be used
    conv_zero_init : bool
        If certain output layers should be initialised with zero weight and bias similar to Dhariwal.
    diffusion_steps: int
        How many forward diffusion steps should be carried out
    diffusion_var_type: patchldmseg.model.diffusion.diffusion_helper.VarianceType
        The type of the predicted variance (LEARNED, FIXED, LEARNED_RANGE)
    dimensions : int
        Dimensionality of the input i.e. whether 2D or 3D operations should be used
    dropout : int
        Percentage of dropout to be used [0, 100]
    hidden_channels : int
        Number of channels the first convolution creates as output channels. The intention behind this is to
        increase the features before the contraction path of the UNet in a dedicated convolutional layer.
    in_channels : int
        Number of input channels to the first convolution of the entire block
        i.e. the number of channels of the input data
    kernel_size : int
        The size of the kernel for convolution
    out_channels : int
        Number of output channels of the network.
    norm : str, optional
        The string of the normalization method to be used. Supported are [batch, instance, layer, group, ada_gn]
    num_res_blocks : int
        Number of residual blocks (with optional attention) per layer.
    padding : bool
        Whether padding should be used or not. The underlying function determines the optimal padding
    padding_mode : str
        The padding mode, i.e. which values to be used for padding
    patch_size : Tuple of int
        The patch size of the input patch i.e. the initial patch size prior to the UNet.
        This is required as attention is utilised as attention is performed only on specific
        attention resolutions, specified by `attention_res`
    preactivation : bool
        Whether the activation should precede the convolution. Preactivation would result in a convolution
        of Norm -> Activation -> Convolution. Otherwise, Convolution -> Norm -> Activation
    spatial_factor : int
        The reduction of the spatial dimension by the pooling layer.
    upd_conv : str
        If the model should use convolution for upsampling/downsampling or traditional pooling/interpolation
    pos_emb : Literal['lin', 'sin']
        The mode of the position embedding. 'Lin' utilises the linear position embedding of paper
        by Bieder et. al, 'sin' utilises the sinusoidal position embedding of the paper by Vaswani et. al.
    p_unconditional: float
        The probability of training an unconditional model. This is part of the
        classifier-free guidance implementation. Typical values are around 0.1 -
        0.2. Setting it to 0.0 (default), will train the (un)conditional model 
        only based on conditional_sampling.
    """

    def __init__(self,
                 dimensions: int,
                 hidden_channels: int,
                 in_channels: int,
                 out_channels: int,
                 dataset: Dataset,
                 patch_size: Tuple[int, ...],
                 activation: Optional[str] = 'ReLU',
                 act_args: tuple = (False,),
                 attention_heads: int = 1,
                 attention_ch_per_head: Optional[int] = None,
                 attention_res: Optional[Tuple[int, ...]] = (32, 16, 8),
                 channel_factor: Tuple[int, ...] = (1, 2, 4, 8),
                 conditional_sampling: bool = False,
                 conv_zero_init: bool = False,
                 diffusion_steps: int = 1000,
                 diffusion_var_type: VarianceType = VarianceType.FIXED_SMALL,
                 dropout: int = 0,
                 kernel_size: int = 3,
                 norm: Optional[Literal['batch', 'instance', 'group', 'layer', 'ada_gn']] = 'group',
                 num_res_blocks: int = 1,
                 padding: bool = True,
                 padding_mode: str = 'zeros',
                 preactivation: bool = False,
                 spatial_factor: int = 2,
                 upd_conv: bool = True,
                 pos_emb: Optional[Literal['sin', 'lin']] = None,
                 is_ldm: bool = False,
                 p_unconditional: float = 0.,
                 **kwargs):

        assert in_channels == out_channels, f"Wrong configuration for diffusion with in_channels={in_channels}" \
                                            f"and out_channels={out_channels}."
        if diffusion_var_type in [VarianceType.LEARNED, VarianceType.LEARNED_RANGE]:
            out_channels *= 2
            self._is_learned_variance = True
        else:
            self._is_learned_variance = False

        super(UNetDiffusion, self).__init__()

        # Attributes
        self.diffusion_steps = diffusion_steps

        # Timestep embedding
        t_emb_ch = 4 * hidden_channels
        self.t_emb = TimeEmbedding(
            in_channels=hidden_channels,
            out_channels=t_emb_ch,
            dimensions=dimensions,
            activation=activation,
            act_args=act_args
        )

        self._is_ldm = is_ldm
        self.pos_emb_mode = pos_emb
        if pos_emb is not None:
            if pos_emb == 'sin':
                assert isinstance(dataset, BraTS2023), "Positional embedding only supported for BraTS"
                pos_emb_ch = 4 * hidden_channels
                self.pos_emb = PosEmbedding(
                    pos_features=hidden_channels,
                    out_features=pos_emb_ch,
                    dimensions=dimensions,
                    activation=activation,
                    act_args=act_args
                )
            elif pos_emb == 'lin':
                self.pos_emb = CoordinateEmbedding(max_spatial_dim=256)  #BraTSTransform resizes to [256,256,128]
                pos_emb_ch = 0
                if not is_ldm:
                    in_channels += 3  # Add the positional embedding here if there is not LDM.
            else:
                raise AttributeError(f"Positional embedding mode {pos_emb} not supported")
        else:
            pos_emb_ch = 0
            self.pos_emb = None

        emb_ch = t_emb_ch + pos_emb_ch       

        # Conditional Sampling
        self.p_unconditional = p_unconditional
        self.conditional_sampling = conditional_sampling
        if conditional_sampling:
            self.label_emb = LabelEmbedding(
                num_embeddings=2, 
                embedding_dim=emb_ch)


        self.encoder = DiffusionUNetEncoder(
            dimensions=dimensions,
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            activation=activation,
            act_args=act_args,
            attention_heads=attention_heads,
            attention_ch_per_head=attention_ch_per_head,
            attention_res=attention_res,
            channel_factor=channel_factor,
            conv_zero_init=conv_zero_init,
            dropout=dropout,
            kernel_size=kernel_size,
            norm=norm,
            num_res_blocks=num_res_blocks,
            emb_ch=emb_ch,
            padding=padding,
            padding_mode=padding_mode,
            patch_size=max(patch_size),
            preactivation=preactivation,
            spatial_factor=spatial_factor,
            upd_conv=upd_conv
        )

        # Bottleneck
        in_channels_bn = self.encoder.out_channels
        out_channels_bn = in_channels_bn
        self.bottleneck = EmbeddingSequential(
            BigGANResBlock(
                dimensions=dimensions,
                in_channels=in_channels_bn,
                out_channels=out_channels_bn,
                activation=activation,
                act_args=act_args,
                conv_zero_init=conv_zero_init,
                dropout=dropout,
                kernel_size=kernel_size,
                norm=norm,
                emb_ch=emb_ch,
                padding=padding,
                padding_mode=padding_mode,
                preactivation=preactivation,
                spatial_factor=spatial_factor,
                time_emb=True,
                upd_conv=None,
                up_or_down=None),
            AttentionBlock(in_channels=in_channels_bn,
                           attention_heads=attention_heads,
                           attention_ch_per_head=attention_ch_per_head,
                           dimensions=dimensions,
                           conv_zero_init=conv_zero_init),
            BigGANResBlock(
                dimensions=dimensions,
                in_channels=in_channels_bn,
                out_channels=out_channels_bn,
                activation=activation,
                act_args=act_args,
                conv_zero_init=conv_zero_init,
                dropout=dropout,
                kernel_size=kernel_size,
                norm=norm,
                emb_ch=emb_ch,
                padding=padding,
                padding_mode=padding_mode,
                preactivation=preactivation,
                spatial_factor=spatial_factor,
                time_emb=True,
                upd_conv=None,
                up_or_down=None),
        )

        self.decoder = DiffusionUNetDecoder(
            dimensions=dimensions,
            hidden_channels=hidden_channels,
            in_channels=self.encoder.out_channels,
            sc_ch_enc=self.encoder.skip_connection_out_ch,
            activation=activation,
            act_args=act_args,
            attention_heads=attention_heads,
            attention_ch_per_head=attention_ch_per_head,
            attention_res=attention_res,
            channel_factor=channel_factor,
            conv_zero_init=conv_zero_init,
            dropout=dropout,
            kernel_size=kernel_size,
            norm=norm,
            num_res_blocks=num_res_blocks,
            emb_ch=emb_ch,
            padding=padding,
            padding_mode=padding_mode,
            patch_size=self.encoder.out_patch_size,
            preactivation=preactivation,
            spatial_factor=spatial_factor,
            upd_conv=upd_conv)

        # Out convolution
        self.out_conv = ConvolutionalBlock(
            in_channels=self.decoder.out_channels,
            out_channels=out_channels,
            dimensions=dimensions,
            kernel_size=kernel_size,
            norm=norm,
            activation=activation if preactivation else None,
            act_args=act_args,
            preactivation=preactivation,
            padding=padding,
            padding_mode=padding_mode,
            dropout=0,
            conv_zero_init=conv_zero_init
        )

    def forward(self,
                x_t: torch.Tensor,
                t: torch.Tensor,
                y: Optional[torch.Tensor] = None,
                location: Optional[torch.Tensor] = None,
                p_unconditional: float = 0.):
        r"""Wrapper for UNetBase forward function. Create all embeddings prior to passing it through 
        the UNet."""
        emb = self.t_emb(t)

        if self.pos_emb is not None:
            assert location is not None, "Provide location for positional embedding"
            if self.pos_emb_mode == 'sin':
                assert isinstance(self.pos_emb, PosEmbedding), "Positional embedding must be of type PosEmbedding"
                emb = torch.cat((emb, self.pos_emb(location=location)), dim=1)
            elif self.pos_emb_mode == 'lin':
                assert isinstance(self.pos_emb, CoordinateEmbedding), "Positional embedding must be of type CoordinateEmbedding"
                x_t = self.pos_emb(x_t, location)
            else:
                raise AttributeError(f"Positional embedding mode {self.pos_emb_mode} not supported")
            
        if self.conditional_sampling:
            label_emb = self.label_emb(y, p_unconditional=p_unconditional)
            emb = emb + expand_dims(label_emb, emb.dim())

        h, skip_connections = self.encoder(x_t, emb)
        h = self.bottleneck(h, emb)
        h = self.decoder(h, skip_connections, emb)
        h = self.out_conv(h)
        
        self._validate_out_shape(in_tensor=x_t, out_tensor=h)
        return h
    
    def _validate_out_shape(self, in_tensor: torch.Tensor, out_tensor: torch.Tensor):
        
        in_shape = list(in_tensor.shape)
        out_shape = list(out_tensor.shape)

        assert in_shape[2:] == out_shape[2:], "Spatial dimensions need to be the same in a UNet"
        assert in_shape[0] == out_shape[0], "Batch size needs to be the same in a UNet"

        if self.pos_emb_mode != 'lin' or (self._is_ldm and self.pos_emb_mode == 'lin'):
            if self._is_learned_variance:
                assert 2 * in_shape[1] == out_shape[1], "Learned variance requires double the channels"
            else:
                assert in_shape[1] == out_shape[1], "Channel size needs to be the same in a UNet"
                
            
        else:
            if self._is_learned_variance:
                assert 2 * (in_shape[1] - 3) == out_shape[1], "Learned variance requires double the channels"
            else:
                assert in_shape[1] - 3 == out_shape[1], "Input requires 3 embedding channels"
