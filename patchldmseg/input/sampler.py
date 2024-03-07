from typing import Optional, Tuple, Union, Dict, Generator
import torch
import numpy as np
import warnings
from enum import Enum

from patchldmseg.utils.misc import calc_distance_map, Stage

from torchio.data.subject import Subject
from torchio.typing import TypeSpatialShape, TypeTripletInt
from torchio.constants import CHANNELS_DIMENSION, LOCATION, NUM_SAMPLES
from torchio.data.sampler import GridSampler, LabelSampler
from torchio.data import Subject


class BraTSDiffusionSampler(LabelSampler):
    r"""The sampler is designed to sample both healthy (no label map present) and diseased samples from
    a subject with label map. The difference to the classical LabelSampler of torchio is the distance-based
    probability for the healthy/background samples. The further away a center pixel is from the label map,
    the more likely it will be sampled.

    Parameters
    ----------
    patch_size : TypeSpatialShape
        The required patch size for sampling. See :class:`~torchio.data.PatchSampler`.
    prob_map_name : str, optional
        Name of the image in the input subject that will be used as a sampling probability map. If none is in
        the subject, the probability map will be calculated
    label_probabilities : dict
        Dictionary containing the probabilities for each class with healthy = 0 and diseased = 1. The class
        is represented by the key in the dictionary and the probability of this class by the value.
        E.g. {0: 0.5, 1: 1} will create patches with equal probability of being healthy and diseased.
    """

    def __init__(self,
                 patch_size: Tuple[int, ...],
                 samples_per_volume: int,
                 in_dim: int,
                 out_dim: int,
                 stage: Stage,
                 prob_map_name: Optional[str] = None,
                 label_probabilities: Optional[Dict[int, float]] = None,
                 **kwargs):

        assert out_dim in (2, 3), "Only 2D and 3D supported output dimensions supported"
        assert in_dim == 3, "Only 3D input images are supported for BraTS"

        self._out_dim = out_dim
        self._in_dim = in_dim
        self._is_test = stage == Stage.TEST
        self._samples_per_volume = samples_per_volume

        if out_dim == 2:
            # Slice_dim is the spatial dimension where to index a 3D volume to get a 2D image
            self._slice_dim = 2  # the z-dim
            assert self._slice_dim == 2

        label_probabilities = self._validate_label_dict(label_probabilities)
        patch_size = self._parse_patch_size(patch_size)

        super().__init__(patch_size=patch_size,
                         label_name=prob_map_name,
                         label_probabilities=label_probabilities)

        # Checks if the correct attributes are set
        if self.label_probabilities_dict is not None:

            assert list(self.label_probabilities_dict.keys())[0] == 0, "First index of dict (healthy) requires 0 as key"
            assert list(self.label_probabilities_dict.keys())[1] == 1, "Second index of dict (diseased) requires 1 as key"
            assert len(label_probabilities) == 2, "This sample only supports two classes: Healthy (0) and Diseased (1)"
        else: 
            raise RuntimeError("No label_probabilities_dict was set. This is required for the BraTS sampler.")

    def _parse_patch_size(self, ps: Tuple[int, ...]) -> TypeTripletInt:
        r"""This function validates and processes the patch size. It assures that the
        sampler has the correct patch_size shape for the given out_dim."""
        if isinstance(ps, tuple):  
            assert all([isinstance(dim, int) for dim in ps]), f'Patch size {ps} contains unsupported values'  
                   
            _error_msg = (f"Out dimensions {self._out_dim} need to match with the patch_size"
                          f"{len(ps)}.")
            assert self._out_dim == len(ps), _error_msg

            if self._in_dim == 3 and self._out_dim == 2:
                ps = tuple(ps + (1,))
        elif isinstance(ps, int):
            if self._out_dim == 2:
                ps = (ps, ps, 1)
            elif self._out_dim == 3:
                ps = (ps, ps, ps)
            else:
                raise RuntimeError(f"Unsupported out_dim format `{self._out_dim}`given")
        else:
            raise RuntimeError("Unsupported patch_size format given")
        return ps

    @staticmethod
    def _validate_label_dict(label_dict: Optional[Dict[int, float]] = None) -> Dict[int, float]:
        if label_dict is None:
            label_dict = {0: 0.5, 1: 0.5}
        else:
            assert all([isinstance(key, int) for key in label_dict.keys()]), f'label_dict contains unsupported values as keys {list(label_dict.keys())}'
        
        return label_dict

    def __call__(
            self, 
            subject: Subject, 
            num_patches: Optional[int] = None) -> Generator[Subject, None, None]:
        
        subject.check_consistent_space()
        if np.any(self.patch_size > subject.spatial_shape):
            message = (
                f'Patch size {tuple(self.patch_size)} cannot be'
                f' larger than image size {tuple(subject.spatial_shape)}'
            )
            raise RuntimeError(message)
        
        probability_map = self.get_probability_map(subject)
        num_max_patches = int(torch.count_nonzero(probability_map))
        requested_patches = min(num_max_patches, self._samples_per_volume)
        setattr(subject, NUM_SAMPLES, requested_patches)

        if num_patches is None:
            num_patches = requested_patches
        return self._generate_patches(subject, probability_map, num_patches)


    def _generate_patches(
        self,
        subject: Subject,
        probability_map: torch.Tensor,
        num_patches: Optional[int] = None,
    ) -> Generator[Subject, None, None]:
        
        probability_map_array = self.process_probability_map(
            probability_map,
            subject,
        )
        cdf = self.get_cumulative_distribution_function(probability_map_array)

        patches_left = num_patches if num_patches is not None else True
        while patches_left:
            yield self.extract_patch(subject, probability_map_array, cdf)
            if num_patches is not None:
                patches_left -= 1


    def get_probability_map(self, subject: Subject) -> torch.Tensor:
        r"""This is a overload of the initial get_probability_map function to return
        also the number of available samples. This is required for the Queue
        to know how many samples are available for the current subject."""
        if self._out_dim == 2:
            if self._is_test:
                return self._brats_2d_prob_map_test(subject)
            else:
                return self._brats_2d_prob_map_train(subject)
        else:
            return self._brats_3d_prob_map(subject)

    def _brats_2d_prob_map_preparation(
            self, subject
    ) -> Tuple[torch.Tensor, torch.Tensor, list, torch.Tensor, int]:
        r"""Prepares the BraTS Probability Map and extract slices depending on the z_lims to have
        real brain slices without background."""

        assert self._in_dim == 3 and self._out_dim == 2
        slice_dim = self._slice_dim + 1
        label_map = self.get_probability_map_image(subject).data.float()
        assert label_map.max() == 1., "Make sure to remap the labels to a binary case in Transforms"
        probability_map = torch.zeros(label_map.shape[slice_dim])
        z_length = label_map.shape[slice_dim]

        reduce_dims = list(range(label_map.dim()))
        reduce_dims.pop(slice_dim)

        foreground_values = torch.stack(
            [torch.sum(
            intensity_img.data > intensity_img.data.min(), 
            dim=tuple(reduce_dims)
            ) for intensity_img in subject.get_images(intensity_only=True)], 0)

        max_pixels_per_slice = sum(subject.spatial_shape[:self._slice_dim])
        max_value_per_slice = torch.max(foreground_values, dim=0).values
        slice_mask = torch.gt(max_value_per_slice, 0.1 * max_pixels_per_slice) 

        return probability_map, label_map, reduce_dims, slice_mask, z_length

    def _check_and_expand_2d_prob_map(
            self, probability_map: torch.Tensor, subject: Subject) -> torch.Tensor:
        r"""Checks if it is a valid probability map i.e. at least one element is non_zero"""

        assert self._out_dim == 2, "This function is only for 2D probability maps"
        assert probability_map.dim() == 1, "Only single slice dims are supported"

        if not torch.any(torch.gt(probability_map.sum(), 0.0)):
            raise RuntimeError("No slice was found that matches the condition")
        else:
            probability_map /= probability_map.sum()

            assert torch.le(torch.abs(torch.sub(torch.sum(probability_map), 1.)), 1e-3)

            returned = torch.zeros(subject.spatial_shape)
            ps_x, ps_y = tuple(self.patch_size[:2].astype(int) // 2)
            returned[ps_x, ps_y, :] =  probability_map
            return returned.unsqueeze(0)

    def _brats_2d_prob_map_test(self, subject: Subject) -> torch.Tensor:
        r"""Processes the probability map for all BraTS images from 3D to 2D using the last
        dimension to slice from for the test case. This means that all slices specified by the slice indices
        are selected."""
        probability_map, label_map, reduce_dims, slice_mask, _ = self._brats_2d_prob_map_preparation(subject)

        probability_map[slice_mask] = 1
        return self._check_and_expand_2d_prob_map(probability_map, subject)

    def _brats_2d_prob_map_train(self, subject: Subject) -> torch.Tensor:
        r"""Processes the probability map for all BraTS images from 3D to 2D using the last
        dimension to slice from. Slices out samples and obtains them based on whether they are healthy
        or not."""
        probability_map, label_map, reduce_dims, slice_mask, _ = self._brats_2d_prob_map_preparation(subject)

        assert self.label_probabilities_dict is not None
        label_probs = torch.Tensor(list(self.label_probabilities_dict.values()))
        normalized_probs = label_probs / label_probs.sum()

        label_elements = torch.sum(torch.greater(label_map, 0), dim=reduce_dims)
        label_elements = label_elements[slice_mask]
        num_non_label_elements = torch.sum(torch.eq(label_elements, 0))

        probability_map[slice_mask] = torch.where(
            label_elements == 0,
            normalized_probs[0] / num_non_label_elements,
            normalized_probs[1] / (label_elements.numel() - num_non_label_elements))

        return self._check_and_expand_2d_prob_map(probability_map, subject)

    def _brats_3d_prob_map(self, subject: Subject, margin: int = 3) -> torch.Tensor: 
        
        label_map = self.get_probability_map_image(subject).data.float()
        patch_size = self.patch_size
        label_probabilities_dict = self.label_probabilities_dict
        assert label_probabilities_dict is not None, "Label probabilities dict is None"

        patch_size = patch_size.astype(int)
        ini_i, ini_j, ini_k = patch_size // 2
        spatial_shape = np.array(label_map.shape[1:])
        if np.any(patch_size > spatial_shape):
            message = (
                f'Patch size {patch_size}'
                f'larger than label map {spatial_shape}'
            )
            raise RuntimeError(message)
        crop_fin_i, crop_fin_j, crop_fin_k = crop_fin = (patch_size - 1) // 2
        fin_i, fin_j, fin_k = spatial_shape - crop_fin
        # See https://github.com/fepegar/torchio/issues/458
        label_map = label_map[:, ini_i:fin_i, ini_j:fin_j, ini_k:fin_k]

        multichannel = label_map.shape[0] > 1

        if multichannel:
            raise NotImplementedError("Not yet implemented")
        else:
            label_map = label_map.squeeze(0)

        probability_map = torch.zeros_like(label_map)
        label_probs = torch.Tensor(list(label_probabilities_dict.values()))
        normalized_probs = label_probs / label_probs.sum()

        # HEALTHY
        healthy_mask = torch.eq(label_map, list(label_probabilities_dict.keys())[0])
        healthy_prob_map = calc_distance_map(torch.logical_not(healthy_mask).int())

        ps_half_norm = torch.norm(torch.as_tensor(patch_size // 2, dtype=torch.float))
        healthy_prob_map = torch.where(
            torch.ge(healthy_prob_map, ps_half_norm + margin**self._out_dim),
            healthy_prob_map, 0.)


        BraTSDiffusionSampler._check_max_ps(patch_size, healthy_prob_map)
        hpm_norm = (healthy_prob_map * normalized_probs[0]) / (healthy_prob_map.sum())
        probability_map.masked_scatter_(healthy_mask, hpm_norm[healthy_mask])

        # DISEASED
        diseased_mask = torch.eq(label_map, list(label_probabilities_dict.keys())[1])
        diseased_prob = normalized_probs[1] / diseased_mask.sum()
        probability_map[diseased_mask] = diseased_prob

        # See https://github.com/fepegar/torchio/issues/458
        padding = ini_k, crop_fin_k, ini_j, crop_fin_j, ini_i, crop_fin_i
        probability_map = torch.nn.functional.pad(
            probability_map,
            padding,
        )
        if not multichannel:
            probability_map = probability_map.unsqueeze(0)

        return probability_map

    @staticmethod
    def _check_max_ps(patch_size: np.ndarray,
                      healthy_prob_map: torch.Tensor) -> None:
        r"""Checks if the distance from the ground-truth mask depicted by the euclidean distance in healthy_prob_map
        is big enough to allow the selected patch size.

        Parameters
        ----------
        patch_size : np.ndarray
            The selected patch size for sampling.
        healthy_prob_map : torch.Tensor
            The euclidean distance map of the tumour label/ the distance from the tumour label. Is calculated
            by patchldmseg.utils.misc.calc_distance_map()
        """
        dimension = healthy_prob_map.dim()
        ini_i, ini_j, ini_k = patch_size // 2
        crop_fin_i, crop_fin_j, crop_fin_k = (patch_size - 1) // 2
        padding = ini_k, crop_fin_k, ini_j, crop_fin_j, ini_i, crop_fin_i

        healthy_prob_map = torch.nn.functional.pad(
            healthy_prob_map,
            padding)
        healthy_prob_map = torch.where(torch.ge(healthy_prob_map, 0.), healthy_prob_map, 0.)
        max_size = torch.as_tensor(healthy_prob_map.shape, dtype=torch.float).unsqueeze(-1)

        BraTSDiffusionSampler._check_max_ps_non_loop(healthy_prob_map, dimension, max_size, patch_size)

    @staticmethod
    def _check_max_ps_non_loop(healthy_prob_map: torch.Tensor,
                               dimension: int,
                               max_size: torch.Tensor,
                               patch_size: np.ndarray):
        r"""Is currently slower by a factor of 5 than the loop mainly due to the computational load of the norm"""
        indices = torch.nonzero(healthy_prob_map, as_tuple=True)

        min_indices = torch.stack([torch.min(torch.stack(indices).float(), dim=0).values for _ in range(dimension)])
        max_indices = torch.stack(
            [torch.min(max_size - torch.stack(indices), dim=0).values for _ in range(dimension)])

        minimum = torch.minimum(torch.norm(min_indices.float(), dim=0), healthy_prob_map[indices])

        maximum = torch.minimum(
            torch.norm(max_indices.float(), dim=0),
            minimum
        )

        # As I am calculating it from a cubic (eg. 16^3) patch size, I can also now calculate it back
        max_ps_half_norm = maximum.max()
        max_theo_ps = int(torch.sqrt(torch.div(torch.pow(max_ps_half_norm, 2) * 4., dimension)))
        assert np.all(patch_size <= max_theo_ps), f"Max theoretical patch size `{max_theo_ps}`" \
                                                  f"is larger than the selected one `{patch_size}`.\n" \
                                                  f"This would prevent having real healthy images"



class GridSamplerSD(GridSampler):
    r"""A grid sampler that works with an entire subjects dataset instead of having a single grid_sampler
    per subject. It also works with 3D volumes that are processed as 2D slices by extracting slice by slice.
    """

    def __init__(self,
                 dummy_subject: Subject,
                 spatial_size: TypeSpatialShape,
                 dimensions: int,
                 patch_size: TypeSpatialShape,
                 patch_overlap: TypeSpatialShape = (0, 0, 0),
                 padding_mode: Union[str, float, None] = None,
                 ):
        r"""Initialisation

        Parameters
        ----------
        dummy_subject : Subject
            Dummy first subject that the inheritance of GridSampler works instead of overwriting
            the entire class with pretty much the same methods.

        spatial_size: TypeSpatialShape
            Int or tuple of ints for the single spatial size that should be used
        """
        self._spatial_shape = spatial_size
        self._dimensions = dimensions

        super().__init__(subject=dummy_subject,
                         patch_size=patch_size,
                         patch_overlap=patch_overlap,
                         padding_mode=padding_mode)

        del self.subject

    def __getitem__(self, index):
        raise NotImplementedError("This class is intended to be used with a torch.utils.data.DataLoader "
                                  "and a torchio.Queue \n"
                                  "and has therefore no object to be indexed stored.")

    def __len__(self):
        return self.num_samples

    @property
    def num_samples(self):
        return len(self.locations)

    @property
    def spatial_shape(self) -> TypeTripletInt:
        return self._spatial_shape

    def _compute_locations(self, subject=None):
        sizes = self.spatial_shape, self.patch_size, self.patch_overlap
        self._parse_sizes(*sizes)
        return self._get_patches_locations(*sizes)


class GridAggregatorSD:
    r"""Aggregate patches for dense inference.

    This class overwrites torchio.data.inference.GridAggregator to be compatible with
    a subjects dataset

    Parameters
    ----------
    sampler: GridSamplerSD
        Instance of :class:`~torchio.data.GridSampler` used to
        extract the patches.
    overlap_mode: str
        If ``'crop'``, the overlapping predictions will be
            cropped. If ``'average'``, the predictions in the overlapping areas
            will be averaged with equal weights. If ``'hann'``, the predictions
            in the overlapping areas will be weighted with a Hann window
            function. See the `grid aggregator tests`_ for a raw visualization
            of the three modes.

    Notes
    -----
    Adapted from NiftyNet. See `this NiftyNet tutorial
    <https://niftynet.readthedocs.io/en/dev/window_sizes.html>`_ for more
    information about patch-based sampling.
    """

    class State(Enum):
        r"""Enumerate Item for the Aggregator state"""
        NOT_FULL = 0
        FULL = 1
        OVERFULL = -1  # The case when a patch fills the aggregator but the next one is already for the new batch

    def __init__(self, sampler: GridSamplerSD, overlap_mode: str = 'crop'):
        self.volume_padded = sampler.padding_mode is not None
        self.spatial_shape = sampler.spatial_shape
        self._output_tensor: Optional[torch.Tensor] = None
        self.patch_overlap = sampler.patch_overlap
        self.patch_size = sampler.patch_size
        self._parse_overlap_mode(overlap_mode)
        self.overlap_mode = overlap_mode
        self._avgmask_tensor: Optional[torch.Tensor] = None
        self._hann_window: Optional[torch.Tensor] = None

        self._state = GridAggregatorSD.State.NOT_FULL

    @property
    def state(self) -> "GridAggregatorSD.State":
        return self._state

    def is_full(self) -> bool:
        return self._state == GridAggregatorSD.State.FULL

    @staticmethod
    def _parse_overlap_mode(overlap_mode):
        if overlap_mode not in ('crop', 'average', 'hann'):
            message = (
                'Overlap mode must be "crop", "average" or "hann" but '
                f' "{overlap_mode}" was passed'
            )
            raise ValueError(message)

    def _crop_patch(
            self,
            patch: torch.Tensor,
            location: np.ndarray,
            overlap: np.ndarray,
    ) -> Tuple[torch.Tensor, np.ndarray]:
        half_overlap = overlap // 2
        index_ini, index_fin = location[:3], location[3:]

        crop_ini = half_overlap.copy()
        crop_fin = half_overlap.copy()

        if self.volume_padded:
            pass
        else:
            crop_ini *= index_ini > 0
            crop_fin *= index_fin != self.spatial_shape

        new_index_ini = index_ini + crop_ini
        new_index_fin = index_fin - crop_fin
        new_location = np.hstack((new_index_ini, new_index_fin))

        patch_size = patch.shape[-3:]
        i_ini, j_ini, k_ini = crop_ini
        i_fin, j_fin, k_fin = patch_size - crop_fin
        cropped_patch = patch[:, i_ini:i_fin, j_ini:j_fin, k_ini:k_fin]
        return cropped_patch, new_location

    def _initialize_output_tensor(self, batch: torch.Tensor) -> None:
        if self._output_tensor is not None:
            return
        num_channels = batch.shape[CHANNELS_DIMENSION]
        self._output_tensor = torch.zeros(
            num_channels,
            *self.spatial_shape,
            dtype=batch.dtype,
        )

    def _initialize_avgmask_tensor(self, batch: torch.Tensor) -> None:
        if self._avgmask_tensor is not None:
            return
        num_channels = batch.shape[CHANNELS_DIMENSION]
        self._avgmask_tensor = torch.zeros(
            num_channels,
            *self.spatial_shape,
            dtype=batch.dtype,
        )

    @staticmethod
    def _get_hann_window(patch_size):
        hann_window_3d = torch.as_tensor([1])
        # create a n-dim hann window
        for spatial_dim, size in enumerate(patch_size):
            window_shape = np.ones_like(patch_size)
            window_shape[spatial_dim] = size
            hann_window_1d = torch.hann_window(
                size + 2,
                periodic=False,
            )
            hann_window_1d = hann_window_1d[1:-1].view(*window_shape)
            hann_window_3d = hann_window_3d * hann_window_1d
        return hann_window_3d

    def _initialize_hann_window(self) -> None:
        if self._hann_window is not None:
            return
        self._hann_window = self._get_hann_window(self.patch_size)

    def add_batch(self,
                  tensor: torch.Tensor,
                  location: torch.Tensor) -> Union[None, torch.Tensor]:
        batch_idx = self.update_state(location)

        if self._state == GridAggregatorSD.State.NOT_FULL:
            self._add_batch(tensor.detach(), location.detach())
            return None

        else:
            if self._state == GridAggregatorSD.State.OVERFULL:
                assert batch_idx is not None
                self._add_batch(tensor[:batch_idx].detach(), location[:batch_idx].detach())
            elif self._state == GridAggregatorSD.State.FULL:
                self._add_batch(tensor.detach(), location.detach())
            else:
                raise RuntimeError(f"State of the aggregator `{self.state}` is not defined")

            total_tensor = self._get_output_tensor().unsqueeze(0).detach().clone()
            self._clear()

            if self._state == GridAggregatorSD.State.OVERFULL:
                assert batch_idx is not None
                self._add_batch(tensor[batch_idx:].detach(), location[batch_idx:].detach())

            return total_tensor.to(tensor.device)

    def _add_batch(
            self,
            batch_tensor: torch.Tensor,
            locations: torch.Tensor,
    ) -> None:
        """Add batch processed by a CNN to the output prediction volume.

        This method is the old add_batch method and functions exactly the same.

        Args:
            batch_tensor: 5D tensor, typically the output of a convolutional
                neural network, e.g. ``batch['image'][torchio.DATA]``.
            locations: 2D tensor with shape :math:`(B, 6)` representing the
                patch indices in the original image. They are typically
                extracted using ``batch[torchio.LOCATION]``.
        """
        batch = batch_tensor.cpu()
        locations = locations.cpu().numpy()
        patch_sizes = locations[:, 3:] - locations[:, :3]
        # There should be only one patch size
        assert len(np.unique(patch_sizes, axis=0)) == 1
        input_spatial_shape = tuple(batch.shape[-3:])
        target_spatial_shape = tuple(patch_sizes[0])
        if input_spatial_shape != target_spatial_shape:
            message = (
                f'The shape of the input batch, {input_spatial_shape},'
                ' does not match the shape of the target location,'
                f' which is {target_spatial_shape}'
            )
            raise RuntimeError(message)
        self._initialize_output_tensor(batch)
        assert isinstance(self._output_tensor, torch.Tensor)
        if self.overlap_mode == 'crop':
            for patch, location in zip(batch, locations):
                cropped_patch, new_location = self._crop_patch(
                    patch,
                    location,
                    self.patch_overlap,
                )
                i_ini, j_ini, k_ini, i_fin, j_fin, k_fin = new_location
                self._output_tensor[
                :,
                i_ini:i_fin,
                j_ini:j_fin,
                k_ini:k_fin,
                ] = cropped_patch
        elif self.overlap_mode == 'average':
            self._initialize_avgmask_tensor(batch)
            assert isinstance(self._avgmask_tensor, torch.Tensor)
            for patch, location in zip(batch, locations):
                i_ini, j_ini, k_ini, i_fin, j_fin, k_fin = location
                self._output_tensor[
                :,
                i_ini:i_fin,
                j_ini:j_fin,
                k_ini:k_fin,
                ] += patch
                self._avgmask_tensor[
                :,
                i_ini:i_fin,
                j_ini:j_fin,
                k_ini:k_fin,
                ] += 1
        elif self.overlap_mode == 'hann':
            self._initialize_avgmask_tensor(batch)
            self._initialize_hann_window()

            if self._output_tensor.dtype != torch.float32:
                self._output_tensor = self._output_tensor.float()

            assert isinstance(self._avgmask_tensor, torch.Tensor)  # for mypy
            if self._avgmask_tensor.dtype != torch.float32:
                self._avgmask_tensor = self._avgmask_tensor.float()

            for patch, location in zip(batch, locations):
                i_ini, j_ini, k_ini, i_fin, j_fin, k_fin = location

                patch = patch * self._hann_window
                self._output_tensor[
                :,
                i_ini:i_fin,
                j_ini:j_fin,
                k_ini:k_fin,
                ] += patch
                self._avgmask_tensor[
                :,
                i_ini:i_fin,
                j_ini:j_fin,
                k_ini:k_fin,
                ] += self._hann_window

    def _get_output_tensor(self) -> torch.Tensor:
        """Get the aggregated volume after dense inference."""
        assert isinstance(self._output_tensor, torch.Tensor)
        if self._output_tensor.dtype == torch.int64:
            message = (
                'Medical image frameworks such as ITK do not support int64.'
                ' Casting to int32...'
            )
            warnings.warn(message, RuntimeWarning)
            self._output_tensor = self._output_tensor.type(torch.int32)
        if self.overlap_mode in ['average', 'hann']:
            assert isinstance(self._avgmask_tensor, torch.Tensor)  # for mypy
            # true_divide is used instead of / in case the PyTorch version is
            # old and one the operands is int:
            # https://github.com/fepegar/torchio/issues/526
            output = torch.true_divide(
                self._output_tensor, self._avgmask_tensor,
            )
        else:
            output = self._output_tensor
        if self.volume_padded:
            from torchio.transforms import Crop
            border = self.patch_overlap // 2
            cropping = border.repeat(2)
            crop = Crop(cropping)  # type: ignore[arg-type]
            return crop(output)  # type: ignore[return-value]
        else:
            return output

    def update_state(self, location: torch.Tensor) -> Optional[int]:
        r"""This function determines if an aggregator is full.

        Returns
        -------

       batch_idx : optional, int
            If the current batch will overflow the current aggregator, we need to be able to truncate the
            tensor to be put in the aggregator by this value

        """

        batch_size = location.shape[0]
        split_idx = location.shape[1] // 2
        start = location[..., :split_idx]
        stop = location[..., split_idx:]

        for batch_idx in range(batch_size):
            is_new_batch = all(start[batch_idx] == 0)
            is_last_batch = torch.equal(stop[batch_idx],
                                        torch.as_tensor(self.spatial_shape,
                                                        device=location.device))

            if is_new_batch and not batch_idx == 0:
                self._state = GridAggregatorSD.State.OVERFULL
                return batch_idx

            elif is_last_batch and batch_idx == batch_size - 1:
                self._state = GridAggregatorSD.State.FULL
                return None

        self._state = GridAggregatorSD.State.NOT_FULL
        return None

    def _clear(self):
        r"""Resets the output tensor as they are re-initialised in _add_batch"""
        self._output_tensor = None
        self._hann_window = None
        self._avgmask_tensor = None