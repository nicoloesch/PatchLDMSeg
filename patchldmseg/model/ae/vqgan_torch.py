from torch import nn
import torch
import torch.nn.functional as F

from patchldmseg.model.basic.conv import ConvolutionalBlock, ResidualBlock, ConvolutionalTransposeBlock, BigGANResBlock, ResidualBlock
from patchldmseg.model.basic.attention import AttentionBlock
from patchldmseg.model.diffusion.diffusion_model import DiffusionBlock
from patchldmseg.model.diffusion.diffusion_model import EmbeddingSequential


from typing import Tuple, Literal, Optional, Any, Union, List, Optional, Type


class VQGANEncoder(nn.Module):
    r"""VQGAN Encoder inspired by Rombach et. al
    https://github.com/CompVis/latent-diffusion

    Very similar to the DiffusionUNetEncoder, however re-implemented to reduce confusion with arguments
    and added new arguments

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    hidden_channels : int
        The hidden channels of the initial convolution to increase the feature channels.
    norm : str
        The string of the normalization method to be used. Supported are [batch, instance, layer, group]
    padding : bool
        Whether padding should be used or not. The underlying function determines the optimal padding
    padding_mode : str
        The padding mode to be used. Supported are [zeros, reflect, replicate, circular]
    preactivation : bool
        Whether the activation should precede the convolution. Preactivation would result in a convolution
        of Norm -> Activation -> Convolution. Otherwise, Convolution -> Norm -> Activation
    channel_factor : tuple of int
        The multiplicative factor of the channel dimension. It is a multiplicative factor for each
        depth layer of the Encoder. The input channels are multiplied by this factor in each EncodingBlock.
    spatial_factor : int
        The reduction of the spatial dimension by the pooling layer. Default: 2
    """
    def __init__(
            self,
            channel_factor: Tuple[int, ...],
            dimensions: int,
            hidden_channels: int,
            in_channels: int,
            z_channels: int,
            num_res_blocks: int,
            patch_size: int,
            activation: Optional[Literal['LeakyReLU', 'ReLU', 'SiLU']] = 'SiLU',
            act_args: tuple = (False, ),
            attention_heads: int = 1,
            attention_res: Optional[Tuple[int, ...]] = (8, 16, 32),
            dropout: int = 0,
            kernel_size: int = 3,
            norm: Optional[Literal['batch', 'instance', 'group', 'layer', 'ada_gn']] = 'group',
            padding: bool = True,
            padding_mode: Literal['zeros', 'reflect', 'replicate', 'circular'] = 'replicate',
            preactivation: bool = True,
            spatial_factor: int = 2,
            upd_conv: bool = True,
            emb_ch: Optional[int] = None,
            ) -> None:
        super().__init__()


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
                        attention_heads=attention_heads,
                        attention_res=attention_res,
                        conv_zero_init=False,
                        dropout=dropout,
                        kernel_size=kernel_size,
                        norm=norm,
                        emb_ch=emb_ch,
                        time_emb=emb_ch is not None,  #Make use of the embedding
                        padding=padding,
                        padding_mode=padding_mode,
                        patch_size=patch_size,
                        preactivation=preactivation,
                    ))

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
                        emb_ch=emb_ch,
                        time_emb=emb_ch is not None,  #Make use of the embedding
                        upd_conv=upd_conv,
                        up_or_down='down',
                        conv_zero_init=False))
                patch_size //= spatial_factor

        self._patch_size = patch_size

        # Middle Blocks
        in_channels_bn = self.encoder_blocks[-1].out_channels
        out_channels_bn = in_channels_bn

        self.bottleneck = EmbeddingSequential(
            BigGANResBlock(
                dimensions=dimensions,
                in_channels=in_channels_bn,
                out_channels=out_channels_bn,
                activation=activation,
                act_args=act_args,
                conv_zero_init=False,
                dropout=dropout,
                kernel_size=kernel_size,
                norm=norm,
                emb_ch=emb_ch,
                time_emb=emb_ch is not None,  #Make use of the embedding
                padding=padding,
                padding_mode=padding_mode,
                preactivation=preactivation,
                spatial_factor=spatial_factor,
                upd_conv=None,
                up_or_down=None),
            AttentionBlock(in_channels=in_channels_bn,
                           attention_heads=attention_heads,
                           dimensions=dimensions,
                           conv_zero_init=False),
            BigGANResBlock(
                dimensions=dimensions,
                in_channels=in_channels_bn,
                out_channels=out_channels_bn,
                activation=activation,
                act_args=act_args,
                conv_zero_init=False,
                dropout=dropout,
                kernel_size=kernel_size,
                norm=norm,
                emb_ch=emb_ch,
                time_emb=emb_ch is not None,  #Make use of the embedding
                padding=padding,
                padding_mode=padding_mode,
                preactivation=preactivation,
                spatial_factor=spatial_factor,
                upd_conv=None,
                up_or_down=None),
        )

        self.out = ConvolutionalBlock(
            in_channels=out_channels_bn,
            out_channels=z_channels,
            dimensions=dimensions,
            kernel_size=kernel_size,
            norm=norm,
            activation=activation,
            act_args=act_args,
            preactivation=preactivation,
            padding=padding,
            padding_mode=padding_mode,
            dropout=0,
            stride=1,
            conv_zero_init=False,
        )

        self._out_channels = z_channels

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, x: torch.Tensor, emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Do not need skip connections in autoencoder
        h = x

        for encoder_block in self.encoder_blocks:
            if isinstance(encoder_block, ConvolutionalBlock):
                h = encoder_block(h)
            else:
                h = encoder_block(h, emb)

        h = self.bottleneck(h, emb)
        return self.out(h)
    
    @property
    def out_ps(self) -> int:
        return self._patch_size
    

class VQGANDecoder(nn.Module):
    r"""VQGAN Decoder inspired by Rombach et. al
    https://github.com/CompVis/latent-diffusion

    Very similar to the DiffusionUNetDecoder, however re-implemented to reduce confusion with arguments
    and added new arguments

    """
    def __init__(
            self,
            z_channels: int,
            channel_factor: Tuple[int, ...],
            dimensions: int,
            hidden_channels: int,
            num_res_blocks: int,
            out_channels: int,
            patch_size: int,
            activation: Optional[Literal['LeakyReLU', 'ReLU', 'SiLU']] = 'SiLU',
            act_args: tuple = (False, ),
            attention_heads: int = 1,
            attention_res: Optional[Tuple[int, ...]] = (8, 16, 32),
            dropout: int = 0,
            kernel_size: int = 3,
            norm: Optional[Literal['batch', 'layer', 'instance', 'group', 'ada_gn']] = 'group',
            padding: bool = True,
            padding_mode: Literal['zeros', 'reflect', 'replicate', 'circular'] = 'replicate',
            preactivation: bool = True,
            spatial_factor: int = 2,
            upd_conv: bool = True,
            emb_ch: Optional[int] = None,
            ) -> None:
        super().__init__()

        out_ch = hidden_channels*channel_factor[-1]
        self.conv_in = ConvolutionalBlock(
            in_channels=z_channels,
            out_channels=out_ch,
            dimensions=dimensions,
            kernel_size=kernel_size,
            norm=None,
            activation=None,
            stride=1,
            padding=True
        )

        self.bottleneck = EmbeddingSequential(
            BigGANResBlock(
                dimensions=dimensions,
                in_channels=out_ch,
                out_channels=out_ch,
                activation=activation,
                act_args=act_args,
                conv_zero_init=False,
                dropout=dropout,
                kernel_size=kernel_size,
                norm=norm,
                emb_ch=emb_ch,
                time_emb=emb_ch is not None,  #Make use of the embedding
                padding=padding,
                padding_mode=padding_mode,
                preactivation=preactivation,
                spatial_factor=spatial_factor,
                upd_conv=None,
                up_or_down=None),
            AttentionBlock(in_channels=out_ch,
                           attention_heads=attention_heads,
                           dimensions=dimensions,
                           conv_zero_init=False),
            BigGANResBlock(
                dimensions=dimensions,
                in_channels=out_ch,
                out_channels=out_ch,
                activation=activation,
                act_args=act_args,
                conv_zero_init=False,
                dropout=dropout,
                kernel_size=kernel_size,
                norm=norm,
                emb_ch=emb_ch,
                time_emb=emb_ch is not None,  #Make use of the embedding
                padding=padding,
                padding_mode=padding_mode,
                preactivation=preactivation,
                spatial_factor=spatial_factor,
                upd_conv=None,
                up_or_down=None),
        )

        self.decoder_blocks = nn.ModuleList()

        in_ch = out_ch
        for decoder_depth, dec_ch_factor in enumerate(reversed(channel_factor)):
            for _ in range(num_res_blocks + 1):
                out_ch = hidden_channels * dec_ch_factor
                self.decoder_blocks.append(DiffusionBlock(
                    dimensions=dimensions,
                    in_channels=in_ch,
                    out_channels=out_ch,
                    activation=activation,
                    act_args=act_args,
                    attention_heads=attention_heads,
                    attention_res=attention_res,
                    conv_zero_init=False,
                    dropout=dropout,
                    kernel_size=kernel_size,
                    norm=norm,
                    emb_ch=emb_ch,
                    time_emb=emb_ch is not None,  #Make use of the embedding
                    padding=padding,
                    padding_mode=padding_mode,
                    patch_size=patch_size,
                    preactivation=preactivation
                ))
                in_ch = out_ch

            # Upsampling
            if decoder_depth != len(channel_factor) - 1:
                self.decoder_blocks.append(
                    BigGANResBlock(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        dimensions=dimensions,
                        kernel_size=kernel_size,
                        norm=norm,
                        activation=activation,
                        act_args=act_args,
                        preactivation=preactivation,
                        padding=padding,
                        dropout=dropout,
                        spatial_factor=spatial_factor,
                        emb_ch=emb_ch,
                        time_emb=emb_ch is not None,  #Make use of the embedding
                        upd_conv=upd_conv,
                        up_or_down='up',
                        conv_zero_init=False
                    ))
                patch_size = int(patch_size * spatial_factor)

        self.decoder_blocks = nn.Sequential(*self.decoder_blocks)

        self.out = ConvolutionalBlock(
            in_channels=self.decoder_blocks[-1].out_channels,
            out_channels=out_channels,
            dimensions=dimensions,
            kernel_size=kernel_size,
            norm=norm,
            activation=activation,
            act_args=act_args,
            preactivation=preactivation,
            padding=padding,
            padding_mode=padding_mode,
            dropout=0,
            stride=1,
            conv_zero_init=False,
        )

    def forward(self, z: torch.Tensor, emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.conv_in(z)
        h = self.bottleneck(h, emb)

        for decoder_block in self.decoder_blocks:
            if isinstance(decoder_block, ConvolutionalBlock):
                h = decoder_block(h)
            else:
                h = decoder_block(h, emb) 
        return self.out(h)


class VQGANEncoderKhader(nn.Module):
    r"""Encoder for VQGAN model. 
    Inspired by work from Khader et al. https://github.com/FirasGit/medicaldiffusion
    
    Parameters
    ----------
    in_channels : int
        Number of input channels.
    hidden_channels : int
        The hidden channels of the initial convolution to increase the feature channels.
    norm : str
        The string of the normalization method to be used. Supported are [batch, instance, layer, group]
    padding : bool
        Whether padding should be used or not. The underlying function determines the optimal padding
    padding_mode : str
        The padding mode to be used. Supported are [zeros, reflect, replicate, circular]
    preactivation : bool
        Whether the activation should precede the convolution. Preactivation would result in a convolution
        of Norm -> Activation -> Convolution. Otherwise, Convolution -> Norm -> Activation
    channel_factor : tuple of int
        The multiplicative factor of the channel dimension. It is a multiplicative factor for each
        depth layer of the Encoder. The input channels are multiplied by this factor in each EncodingBlock.
    spatial_factor : int
        The reduction of the spatial dimension by the pooling layer. Default: 2
    """
    def __init__(
            self,
            channel_factor: Tuple[int, ...],
            dimensions: int,
            hidden_channels: int,
            in_channels: int,
            patch_size: int,
            activation: Optional[Literal['LeakyReLU', 'ReLU', 'SiLU']] = 'SiLU',
            act_args: tuple = (False, ),
            dropout: int = 0,
            kernel_size: int = 3,
            norm: Optional[Literal['batch', 'instance', 'group', 'layer', 'ada_gn']] = 'group',
            padding: bool = True,
            padding_mode: Literal['zeros', 'reflect', 'replicate', 'circular'] = 'replicate',
            preactivation: bool = True,
            spatial_factor: int = 2,
            emb_ch: Optional[int] = None,
            *args,
            **kwargs
            ) -> None:
        super().__init__()

        self.in_conv = ConvolutionalBlock(
            in_channels=in_channels,
            out_channels=hidden_channels,
            dimensions=dimensions,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            padding_mode=padding_mode,
            norm=None,
            activation=None,
            act_args=act_args,)
        
        self.encoder_blocks = nn.ModuleList()

        in_channels = hidden_channels

        # Do not use pooling in last layer
        spatial_factors = [spatial_factor for _ in range(len(channel_factor))]

        # Define out_channels here to get rid of error msg as I am overwriting them anyways
        out_channels = hidden_channels

        for ch_factor, stride in zip(channel_factor, spatial_factors):
            out_channels = ch_factor * hidden_channels
            patch_size //= stride

            self.encoder_blocks.append(
                # Pooling Layer (except for the last layer)
                ConvolutionalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    dimensions=dimensions,
                    kernel_size=4,
                    stride=stride,
                    padding=padding,
                    padding_mode=padding_mode,
                    norm=None,
                    activation=None,
                    act_args=act_args))
            
            self.encoder_blocks.append(
                ResidualBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    dimensions=dimensions,
                    activation=activation,
                    act_args=act_args,
                    dropout=dropout,
                    kernel_size=kernel_size,
                    padding=padding,
                    padding_mode=padding_mode,
                    norm=norm,
                    preactivation=preactivation,
                    time_emb=emb_ch is not None,  #Make use of the embedding
                    emb_ch=emb_ch,
                ))
            in_channels = out_channels

        self.final_block = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=out_channels),
            nn.SiLU()
        )

        self._out_channels = out_channels
        self._out_ps = patch_size

    @property
    def out_channels(self) -> int:
        return self._out_channels
    
    @property
    def out_ps(self) -> int:
        return self._out_ps
    
    def forward(self, x: torch.Tensor, emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.in_conv(x)

        for encoder_block in self.encoder_blocks:
            if isinstance(encoder_block, ConvolutionalBlock):
                # NOTE: in_conv is in encoder_blocks and does not have any t_emb
                h = encoder_block(h)
            else:
                h = encoder_block(h, emb)
        return self.final_block(h)


class VQGANDecoderKhader(nn.Module):
    """Decoder for VQGAN model.
    Inspired by work from Khader et al. https://github.com/FirasGit/medicaldiffusion

    Notes
    -----
    The transpose convolution is currently not working

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    hidden_channels : int
        The hidden channels of the initial convolution to increase the feature channels.
    norm : str
        The string of the normalization method to be used. Supported are [batch, instance, layer, group]
    padding : bool
        Whether padding should be used or not. The underlying function determines the optimal padding
    padding_mode : str
        The padding mode to be used. Supported are [zeros, reflect, replicate, circular]
    preactivation : bool
        Whether the activation should precede the convolution. Preactivation would result in a convolution
        of Norm -> Activation -> Convolution. Otherwise, Convolution -> Norm -> Activation
    channel_factor : tuple of int
        The multiplicative factor of the channel dimension. It is a multiplicative factor for each
        depth layer of the Encoder. The input channels are multiplied by this factor in each EncodingBlock.
    spatial_factor : int
        The reduction of the spatial dimension by the pooling layer. Default: 2
    """
    def __init__(
            self,
            channel_factor: Tuple[int, ...],
            dimensions: int,
            hidden_channels: int,
            out_channels: int,
            activation: Optional[Literal['LeakyReLU', 'ReLU', 'SiLU']] = 'SiLU',
            act_args: tuple = (False, ),
            dropout: int = 0,
            kernel_size: int = 3,
            norm: Optional[Literal['batch', 'layer', 'instance', 'group', 'ada_gn']] = 'group',
            padding: bool = True,
            padding_mode: Literal['zeros', 'reflect', 'replicate', 'circular'] = 'replicate',
            preactivation: bool = True,
            spatial_factor: int = 2,
            emb_ch: Optional[int] = None,
            *args,
            **kwargs,
            ) -> None:
        super().__init__()

        self.final_block = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=channel_factor[-1] * hidden_channels),
            nn.SiLU()
        )

        stride_factors = [spatial_factor for _ in range(len(channel_factor))]

        self.decoder_blocks = nn.ModuleList()

        for in_ch_factor, out_ch_factor, stride in zip(
            reversed(channel_factor),
            reversed((1, ) + channel_factor[:-1]),
            stride_factors):

            in_ch = in_ch_factor * hidden_channels
            out_ch = out_ch_factor * hidden_channels

            layer_blocks = [
                # Up convolution
                ConvolutionalTransposeBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    dimensions=dimensions,
                    kernel_size=4,
                    stride=stride,
                    padding=padding,
                    padding_mode=padding_mode,
                    norm=None,
                    activation=None,
                ),
                ResidualBlock(
                    in_channels=out_ch,
                    out_channels=out_ch,
                    dimensions=dimensions,
                    activation=activation,
                    act_args=act_args,
                    dropout=dropout,
                    kernel_size=kernel_size,
                    padding=padding,
                    padding_mode=padding_mode,
                    norm=norm,
                    preactivation=preactivation,
                    time_emb=emb_ch is not None,  #Make use of the embedding
                    emb_ch=emb_ch),

                ResidualBlock(
                    in_channels=out_ch,
                    out_channels=out_ch,
                    dimensions=dimensions,
                    activation=activation,
                    act_args=act_args,
                    dropout=dropout,
                    kernel_size=kernel_size,
                    padding=padding,
                    padding_mode=padding_mode,
                    norm=norm,
                    preactivation=preactivation,
                    time_emb=emb_ch is not None,  #Make use of the embedding
                    emb_ch=emb_ch)
            ]
            self.decoder_blocks.extend(layer_blocks)

            self.out_conv = ConvolutionalBlock(
                in_channels=out_ch,
                out_channels=out_channels,
                dimensions=dimensions,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                padding_mode=padding_mode,
                norm=None,
                activation=None,
            )

        self._out_channels = out_channels
    
    @property
    def out_channels(self) -> int:
        return self._out_channels
    
    def forward(self, x: torch.Tensor, emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.final_block(x)

        for decoder_block in self.decoder_blocks:
            if isinstance(decoder_block, ConvolutionalTransposeBlock):
                h = decoder_block(h)
            else:
                h = decoder_block(h, emb)
        return self.out_conv(h)


class EmbeddingEMA(nn.Module):
    def __init__(self, num_tokens, codebook_dim, decay=0.99, eps=1e-5):
        super().__init__()
        self.decay = decay
        self.eps = eps        
        weight = torch.randn(num_tokens, codebook_dim)
        self.weight = nn.Parameter(weight, requires_grad = False)
        self.cluster_size = nn.Parameter(torch.zeros(num_tokens), requires_grad = False)
        self.embed_avg = nn.Parameter(weight.clone(), requires_grad = False)
        self.update = True

    def forward(self, embed_id):
        return F.embedding(embed_id, self.weight)

    def cluster_size_ema_update(self, new_cluster_size):
        self.cluster_size.data.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)

    def embed_avg_ema_update(self, new_embed_avg): 
        self.embed_avg.data.mul_(self.decay).add_(new_embed_avg, alpha=1 - self.decay)

    def weight_update(self, num_tokens):
        n = self.cluster_size.sum()
        smoothed_cluster_size = (
                (self.cluster_size + self.eps) / (n + num_tokens * self.eps) * n
            )
        #normalize embedding average with smoothed cluster size
        embed_normalized = self.embed_avg / smoothed_cluster_size.unsqueeze(1)
        self.weight.data.copy_(embed_normalized)
        

class EMAVectorQuantizer(nn.Module):

    r"""Codebook for vector quantisation following the implementation
    of Rombach et al.
    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/vqvae/quantize.py"""
    def __init__(
            self,
            num_codes: int,
            embedding_dim: int,
            beta: float = 0.25,
            ema_decay: float = 0.99,

            ) -> None:
        super().__init__()
        # Attributes
        self.num_codes = num_codes
        self.embedding_dim = embedding_dim
        self.beta = beta

        if ema_decay == 0.0:
            self.embedding = nn.Embedding(num_codes, embedding_dim)
        elif 1.0 >= ema_decay > 0.0:
            self.embedding = EmbeddingEMA(num_tokens=num_codes, codebook_dim=embedding_dim, decay=ema_decay)
        else:
            raise ValueError('ema_decay must be in [0.0, 1.0]')


    def forward(self, z) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # Flatten the input and reshape to [-1, embedding_dim]
        z = z.movedim(1, -1).contiguous()
        z_flat = z.view(-1, self.embedding_dim)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        distance = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True) + 
            torch.sum(self.embedding.weight ** 2, dim=1) -
            2 * torch.einsum('bd,nd->bn', z_flat, self.embedding.weight))
        

        # find closest encoding
        encoding_indices = torch.argmin(distance, dim=1)


        z_q = self.embedding(encoding_indices).view(z.shape)
        encodings = F.one_hot(encoding_indices, self.num_codes).type(z.dtype)     
        avg_probs = torch.mean(encodings, dim=0)

        # Prevent division by zero
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(
            avg_probs + max(1e-10, torch.finfo(avg_probs.dtype).tiny))))

        # EMA Update
        if self.training and isinstance(self.embedding, EmbeddingEMA):
            #EMA cluster size
            encodings_sum = encodings.sum(0)            
            self.embedding.cluster_size_ema_update(encodings_sum)
            #EMA embedding average
            embed_sum = torch.matmul(encodings.transpose(0,1), z_flat)            
            self.embedding.embed_avg_ema_update(embed_sum)
            #normalize embed_avg and update weight
            self.embedding.weight_update(self.num_codes)

        # Compute loss for embedding
        loss = self.beta * F.mse_loss(z_q.detach(), z)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        z_q = torch.movedim(z_q, -1, 1).contiguous()
        return z_q, loss, perplexity, encodings, encoding_indices

class NLayerDiscriminator(nn.Module):
    def __init__(
            self,
            input_channels: int,
            dimensions: Literal[2, 3],
            hidden_channels: int = 64,
            num_layers: int = 3,
            norm_layer: Union[Type[nn.SyncBatchNorm],
                              Type[nn.BatchNorm2d],
                              Type[nn.BatchNorm3d]] = nn.SyncBatchNorm,
            use_sigmoid: bool = False,
            get_intermediate_features: bool = True,
            ) -> None:
        super().__init__()
        
        self.get_intermediate_features = get_intermediate_features
        self.num_layers = num_layers

        self.model = nn.ModuleList()

        kernel_size = 4

        # NOTE: Padding differently between Khader and Rombach
        # I have Rombach which is (kernel - 1) // 2 (floor)
        # Khader is ceil((kernel - 1) / 2)
        self.model.append(ConvolutionalBlock(
                in_channels=input_channels, 
                out_channels=hidden_channels,
                dimensions=dimensions, 
                kernel_size=kernel_size, 
                stride=2, 
                padding=True,
                activation='LeakyReLU',
                act_args=(0.2, True)))

        num_features = hidden_channels
        feature_factor = [2 ** i for i in range(1, num_layers)]

        for ff in feature_factor:
            num_features_prev = num_features
            num_features = min(num_features * ff, 512)
            self.model.append(
                ConvolutionalBlock(
                    in_channels=num_features_prev,
                    out_channels=num_features,
                    dimensions=dimensions,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=True,
                    norm='group',
                    preactivation=False,  # conv-norm-act
                    activation='LeakyReLU',
                    act_args=(0.2, True)))

        
        num_features_prev = num_features
        num_features = min(num_features, 512)

        self.model.append(
            ConvolutionalBlock(
                in_channels=num_features_prev,
                out_channels=num_features,
                dimensions=dimensions,
                kernel_size=kernel_size,
                stride=1,
                padding=True,
                norm='group',
                preactivation=False,  # conv-norm-act
                activation='LeakyReLU',
                act_args=(0.2, True)))

        self.model.append(
            ConvolutionalBlock(
                in_channels=num_features,
                out_channels=1,
                dimensions=dimensions,
                kernel_size=kernel_size,
                stride=1,
                padding=True,
                activation=None,
            ))

        if use_sigmoid:
            self.model.append(nn.Sequential(nn.Sigmoid()))
            
        if not get_intermediate_features:
            # Combine all layers together for easier forward
            self.model = nn.Sequential(*self.model)


    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        if self.get_intermediate_features:
            features = [x]
            for layer in self.model:
                features.append(layer(features[-1]))
            return features[-1], features[1:]
        else:
            return self.model(x), _
