import torch
import torch.nn as nn

from typing import Optional

from patchldmseg.utils.misc import zero_init


class QKVAttention(nn.Module):
    r"""Core module for attention with virtual attention heads along the channel
    dimension"""
    def __init__(self,
                 attention_heads: int,
                 dimensions: int):
        super().__init__()
        self._num_ah = attention_heads
        self._dimensions = dimensions

        # Softmax activation
        self._softmax = nn.Softmax(dim=1)


    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        if qkv.dim() > 3:
            requires_reshape = True
            spat = tuple(qkv.shape[2:])
            qkv = torch.flatten(qkv, start_dim=2)
        elif qkv.dim() == 3:
            requires_reshape = False
            spat = None
        else:
            raise RuntimeError(f"Unsupported shape for qkv {qkv.shape}")
            
        b, c_3_ah, length = qkv.shape

        assert c_3_ah % (3 * self._num_ah) == 0
        c_3 = c_3_ah // self._num_ah

        scale = torch.sqrt(torch.as_tensor(1. / (c_3 // 3)))

        q, k, v = torch.chunk(qkv.reshape(b * self._num_ah, c_3, length).contiguous(), chunks=3, dim=1)


        attention_map = torch.einsum(self.create_einsum_string(qk=True), q, k).contiguous()
        attention_map = self._softmax(torch.mul(attention_map, scale)).contiguous()
        self_attention_map_ah = torch.einsum(self.create_einsum_string(qk=False), attention_map, v).contiguous()

        self_attention_map = self_attention_map_ah.reshape(b, c_3_ah // 3, length).contiguous()

        if requires_reshape:
            assert isinstance(spat, tuple)
            self_attention_map = self_attention_map.reshape(b, c_3_ah // 3, *spat).contiguous()

        return self_attention_map

    def create_einsum_string(self, qk: bool):
        r"""Returns the einsum string for the attention.
        Parameters
        ----------
        qk: bool
            Whether the q/k attention (first) is done or the attention_map/v (second)
        """

        if qk:
            return "bcs,bct->bst"
        else:
            return "bst,bct->bcs"



class AttentionBlock(nn.Module):
    r"""Implementation of the visual attention module.
    The implementation follows the Paper by Vaswani et. al 'Attention is all you need'
    and the visual explanation (https://www.youtube.com/watch?v=mMa2PmYJlCo).
    Also incorporates pooling conditioned on the input following
    https://arxiv.org/pdf/2103.00020.pdf (https://github.com/openai/CLIP/blob/main/clip/model.py)

    Parameters
    ----------
    in_channels : int
        Input channel dimension
    out_channels : int, optional
        Number of output_channels. If not specified, out_channels = in_channels
    attention_heads : int
        Number of attention heads for multi-head attention
    attention_ch_per_head : int
        Number of channels per attention head. If this is set to None, attention_heads determines
        the number of attention heads. If it is set to an integer, number of heads
        are determined automatically.
    dimensions : int
        The dimensionality of the input patch. Required to get the right convolution.
    conv_zero_init : bool
        If the output convolution should have zero weight and bias as in Dhariwal
    """

    def __init__(self,
                 in_channels: int,
                 dimensions: int,
                 out_channels: Optional[int] = None,
                 attention_heads: int = 1,
                 attention_ch_per_head: Optional[int] = None,
                 conv_zero_init: bool = False
                 ):

        super().__init__()

        self._dimensions = dimensions
        self._in_channels = in_channels
        out_channels = out_channels or in_channels
        emb_channels = in_channels  # // 8  # Reduction in computational load

        if attention_ch_per_head is None:
            num_attention_heads = attention_heads
        else:
            _error_msg = (
                f"Attention embedding channels must be divisible by the number of attention channels per head"
                f" and must be larger than the number of attention channels per head"
            )
            assert emb_channels % attention_ch_per_head == 0 and emb_channels >= attention_ch_per_head, _error_msg
            num_attention_heads = emb_channels // attention_ch_per_head
        self._num_ah = num_attention_heads

        self._norm = nn.GroupNorm(num_groups=32, num_channels=in_channels)

        conv_class = getattr(torch.nn, f'Conv{dimensions}d')
        self.qkv_conv = conv_class(in_channels=in_channels,
                             out_channels=emb_channels * 3,
                             kernel_size=1)
        
        self.qkv_attention = QKVAttention(attention_heads=num_attention_heads,
                                          dimensions=dimensions)

        conv_layer = conv_class(in_channels=emb_channels,
                               out_channels=out_channels,
                               kernel_size=1)
        if conv_zero_init:
            conv_layer = zero_init(conv_layer)

        self.ah_conv = conv_layer
        self.skip = nn.Identity() if in_channels == out_channels else conv_class(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, t_emb: Optional[torch.Tensor] = None):

        skip = self.skip(x)

        qkv = self.qkv_conv(self._norm(x))
        
        # Self attention map with virtual attention heads dim
        self_attention_map_ah = self.qkv_attention(qkv)

        # Reshape the self attention map into [B,C*num_ah,Spat]
        self_attention_map = self.ah_conv(self_attention_map_ah)

        assert skip.shape == self_attention_map.shape
        return (skip + self_attention_map).contiguous()

    @property
    def out_channels(self):
        r"""Returns the input channels as this block keeps the spatial dimensions and the channel dimension."""
        return self._in_channels


class AttentionPool(nn.Module):
    r"""Attention pooling node similar to the Attention Block. This method relies on flattened input patches due to the usage of mean 
    and concatenation.

    Parameters
    ----------
    emb_channels : int
        Number of channels for the embedding
    out_channels : int, optional
        Number of output_channels. If not specified, out_channels = emb_channels
    attention_heads : int
        Number of attention heads for multi-head attention
    attention_ch_per_head : int
        Number of channels for each attention head. Is used instead of attention_heads if specified. Default -1,
        meaning that attention_heads is used.
    dimensions : int
        The dimensionality of the input patch. Only required if pooling is False
    conv_zero_init : bool
        If the output convolution should have zero weight and bias as in Dhariwal
    spatial_dim : int
        The spatial dimension of the patch. Required for the positional embedding
    """

    def __init__(self,
                 emb_channels: int,
                 spatial_dim: int,
                 dimensions: int,
                 attention_heads: int = 1,
                 attention_ch_per_head: int = -1,
                 out_channels: Optional[int] = None,
                 conv_zero_init: bool = False
                 ):
        super().__init__()
        out_channels = out_channels or emb_channels
        self._pos_emb = nn.Parameter(
            torch.randn(emb_channels, spatial_dim ** dimensions + 1) / torch.sqrt(
                torch.as_tensor(emb_channels)))
        
        if attention_ch_per_head == -1:
            num_attention_heads = attention_heads
        else:
            assert emb_channels % attention_ch_per_head == 0, "Attention channels must be divisible by the number of attention channels per head"
            num_attention_heads = emb_channels // attention_ch_per_head

        self.qkv_conv = nn.Conv1d(in_channels=emb_channels,
                             out_channels=emb_channels * 3,
                             kernel_size=1)
        
        self.qkv_attention = QKVAttention(attention_heads=num_attention_heads,
                                          dimensions=dimensions)

        conv_layer = nn.Conv1d(
            in_channels=emb_channels * attention_heads,
            out_channels=out_channels,
            kernel_size=1)

        if conv_zero_init:
            conv_layer = zero_init(conv_layer)

        self.ah_conv = conv_layer

    def forward(self, x):
        x = torch.flatten(x, start_dim=2)

        x = torch.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)
        x = x + self._pos_emb[None, :, :]

        qkv = self.qkv_conv(x)
        self_attention_map = self.qkv_attention(qkv)

        h = self.ah_conv(self_attention_map)
        return h[:, :, 0]
