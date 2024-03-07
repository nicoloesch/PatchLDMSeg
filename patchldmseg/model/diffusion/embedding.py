from torch import nn
import torch
from typing import Optional

from patchldmseg.model.basic.conv import get_activation_layer

class LabelEmbedding(nn.Module):
    r"""This class is for label embedding for class-conditional diffusion models
    including classifier-free guidance with masking."""

    def __init__(
            self, 
            embedding_dim: int,
            num_embeddings: int = 2,
            ) -> None:
        super().__init__()

        self.embedding_layer = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim)

    def forward(self, 
                y: Optional[torch.Tensor], 
                p_unconditional: float = 0.) -> torch.Tensor:
        r"""Forward pass of the label embedding layer.

        Parameters
        ----------
        y : torch.Tensor, optional
            The label to be embedded
        p_unconditional : float
            The probability of having a null label embedding (i.e. no label embedding). 
            Setting a value of 0. utilises the entire label embedding, whereas a value of 1
            will disable the entire label embedding.
        """

        if y is not None:
            if y.dim() == 2:
                y = y.squeeze(dim=1)
            assert y.dim() == 1, "Only mutually exclusive class embeddings supported"

            assert y.max() + 1 <= self.embedding_layer.num_embeddings, (
                f"Label is out of range of the embedding layer. Please check the label range"
                f" ({int(y.min())}, {int(y.max())}) and the number of embeddings ({self.embedding_layer.num_embeddings})"
            )

            assert 0.0 <= p_unconditional <= 1.0, (
                f"Probability of unconditional label embedding must be between 0 and 1 but is {p_unconditional}"
            )

            label_emb = self.embedding_layer(y)

            # Full/No label embedding if set to bool

            # floating point comparisons are difficult with precision
            # as we are usually interested in 0.1, 0.2 etc, we can use this threshold to 
            # make sure that p_unconditional = 1.0 always gets no label embedding
            # and p_unconditional = 0.0 always gets the full label embedding
            eps = 1e-4  

            # Mask out the class embedding with probability 1 - p
            mask = torch.zeros_like(y).unsqueeze(-1).float().uniform_(0, 1) < p_unconditional - eps 

            # Set to 0 where classifier free guidance is active
            label_emb = torch.where(mask, 0., label_emb)

        else:
            # Single batch domain as it is added to the embedding
            label_emb = torch.zeros(
                size=(1, self.embedding_layer.embedding_dim), 
                device=self.embedding_layer.weight.device)

        return label_emb

class PosEmbedding(nn.Module):
    r"""This class is for positional embedding for multidimensional
    diffusion models, mostly designed for patch based approaches.
    The module embeds the bounding box coordinates of the patch into 
    a learnable representation.
    
    Parameters
    ----------
    pos_features: int
        How many features should be extracted for each axis through the positional
        embedding.
    out_features: int
        The output features of the embedding layer.
    dimensions: int
        The dimensions of the input patch. Required to expand the embedding to the 
        correct shape.
    activation: str, optional
        Activation layer (if specified)
    act_args: tuple
        The specific arguments required for the activation function
    """

    def __init__(self,
                 pos_features: int,
                 dimensions: int,
                 out_features: int,
                 activation: Optional[str] = 'ReLU',
                 act_args: tuple = (False, )) -> None:
        super().__init__()

        self.pos_features = pos_features // 2  # half for start and end
        self.dimensions = dimensions

        act = get_activation_layer(activation=activation, act_args=act_args)
        # The 3 originates from the 3D volumes being processed regardless if 2D or 3D patch extraction
        # The location has start_end for all 3 dimensions of the volume
        self.mlp = nn.Sequential(
            nn.Linear(
                in_features=pos_features * 3,  #feature for each dimension
                out_features=out_features),
            act,
            nn.Linear(in_features=out_features,
                      out_features=out_features),
        )

    def forward(self, location: torch.Tensor) -> torch.Tensor: 

        embedding = torch.cat([
            sinusoidal_position_embedding(
                features=self.pos_features,
                pos=location[..., loc],
                device=location.device) for loc in range(location.shape[-1])
        ], dim=-1)

        embedding = self.mlp(embedding)

        # Unsqueeze the embeddings to match the input shape
        embedding = embedding[(...,) + (None,) * self.dimensions]

        return embedding

class CoordinateEmbedding:
    r"""Coordinate embedding by Bieder et al."""
    def __init__(self,
                 max_spatial_dim: int):
        self._ce_cache = None
        self._max_spatial_dim = max_spatial_dim

    def __call__(
            self, 
            x: torch.Tensor, 
            location: torch.Tensor):
        if self._ce_cache is None:
            self._ce_cache = self._coordinate_embedding(
                x, 
                max_spatial_dim=self._max_spatial_dim)
        
        sliced_ce = self._extract_patch_with_location(location=location, embedding=self._ce_cache)
        try:
            return torch.cat([x, sliced_ce], dim=1)
        except RuntimeError as e:
            # NOTE: the embedding cannot be added in the case of LDM as the location 
            # is for the initial dim and not for the reduced dim. We just return x then
            # as the position is already encoded with the VQGAN-encoder
            if "Sizes of tensors must match except in dimension 1" in e.args[0]:
                return x 
            else:
                raise RuntimeError(e)

    
    @staticmethod
    def _extract_patch_with_location(location: torch.Tensor, embedding: torch.Tensor):
        r"""
        Parameters
        ----------
        location : torch.Tensor
            The location of the patch with the torchio format of
            [start_x, start_y, start_z, end_x, end_y, end_z]
        """
        cropped_embeddings = []
        embedding = embedding.to(location.device)

        for batch_location in location:
            assert len(batch_location) == 6, "Unsupported location format"

            # Get start and end coordinates
            s_x, s_y, s_z, e_x, e_y, e_z = batch_location
            cropped_embeddings.append(embedding[..., s_x:e_x, s_y:e_y, s_z:e_z])
        
        return torch.stack(cropped_embeddings, dim=0).squeeze(-1)  # squeeze the last dimension as it is 1 in 2D
    
    @staticmethod
    def _coordinate_embedding(
            x: torch.Tensor, 
            max_spatial_dim: int):
        r"""Coordinate embedding by Bieder et al.
        
        Parameters
        ----------
        x : torch.Tensor
            The input patch tensor for which embedding is required
        location : torch.Tensor
            The locational information of the patch. Required for slicing out the correct position of the 
            patch
        max_spatial_dim : int
            The maximum spatial dimension. For Brats (240x240x155) this would be 240 (if it is not padded).
        """
        dim = 3 # always three-dimensional as we are processing 3D volumes even in 2D
        return torch.stack(torch.meshgrid(dim * [torch.linspace(-1, 1, max_spatial_dim)], indexing='ij'), dim=0)


class TimeEmbedding(nn.Module):
    r"""This class is a time step embedder, which is required for the diffusion process
    as parameters are shared between the denoising steps and positional information is
    crucial to distinguish where in the denoising sequence one currently is to learn the
    respective noise contribution.

    A Great video about the transformer and how the positional embedding works there can be found
    https://www.youtube.com/watch?v=dichIcUZfOw here.

    Notes
    -----
        Linear embedding of the time would not work, as there is not always a fixed size diffusion
        process (less prominent in images than in text but argument still holds)

    """

    # NOTE: This embedding is quite crucial and has a lot of potential to modify the learning

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dimensions: int,
                 activation: Optional[str] = 'ReLU',
                 act_args: tuple = (False,),
                 ):
        r"""Initialisation of the TimeEmbedding class

        Parameters
        ----------
        in_channels : int
            Number of input channels which result in the extraction of as many embeddings from
            the timestep embedding
        out_channels: int
            Number of output channels of the layer.
        dimensions : int
            Number of spatial dimensions of the output [B, C, num_spatial_dim]. The same as the input tensor,
            onto which the time embedding is added
        activation : str, optional
            Activation layer (if specified)
        act_args : tuple
            The specific arguments required for the activation function
        """
        super().__init__()
        self.features = in_channels
        self.dimensions = dimensions

        # Include Learner for the positional embedding
        act = get_activation_layer(activation=activation, act_args=act_args)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=in_channels,
                      out_features=out_channels),
            act,
            nn.Linear(in_features=out_channels,
                      out_features=out_channels),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        embedding = sinusoidal_position_embedding(
            features=self.features,
            pos=timesteps, 
            device=timesteps.device)
        embedding = self.mlp(embedding)

        # Unsqueeze the embeddings to match the input shape
        embedding = embedding[(...,) + (None,) * self.dimensions]

        return embedding

def sinusoidal_position_embedding(features: int, pos: torch.Tensor, device: torch.device):
    """Position embedding similar to Vaswani et al. 'Attention is all you need'.
    Follows the guide 'A Gentle Introduction to Positional Encoding in Transformer Models, Part 1'
    of Machine Learning Mastery and partially the video listed above in the class description
    (which is FALSE to an extent regarding 2i and 2i+1) (see
    https://datascience.stackexchange.com/questions/51065/what-is-the-positional-encoding-in-the-transformer-model
    first comment 'Correction')

    Notes
    -----
    d: representation dimension - size of the position embedding
    (same size as the features/embeddings of the image)

    i: 0 <= i < d/2 indices for the split between sine and cosine depending on odd or even i

    .. math::
        PE_{pos,2i} = \\sin{\\frac{pos}{10000 ^ {\\frac{2i}{d}}}} \n
        PE_{pos,2{i+1}} = \\cos{\\frac{pos}{10000 ^ {\\frac{2i}{d}}}}
    """

    # Assure, that the timesteps are just of Size batch_size
    # IDEA: This can be extended in the future and might need to be adapted
    #   depending on how I am going to include the feature dimensions
    assert pos.dim() == 1

    half_dim = features // 2

    # NOTE: we are not building the entire [t,d] dimension but already utilise the t's within the batch
    #   resulting in a [bxd] matrix -> is faster
    pe = torch.zeros((pos.shape[0], features), device=device)

    k = torch.arange(half_dim, device=device)
    freq = torch.exp(torch.log(torch.as_tensor([10000.], device=device)) * -k / half_dim)

    # As time is only of size BS, we need to unsqueeze the last dimension to allow multiplication
    emb = pos.unsqueeze(-1) * freq

    # Sinus embedding if i = 2k, cosine for i = 2k + 1
    pe[:, 2 * k] = emb.sin()
    pe[:, 2 * k + 1] = emb.cos()

    return pe