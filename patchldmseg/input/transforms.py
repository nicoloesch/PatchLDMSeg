import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torchio.data.subject import Subject
import torchio.transforms as tt
from torch.nn import functional as tf
from torchio.transforms import LabelTransform, Transform, SpatialTransform
from torchio.transforms.transform import TypeMaskingMethod
from torchio.transforms.preprocessing.intensity.normalization_transform import NormalizationTransform
from torchio.typing import TypeRangeFloat
from torchio import Image, Subject, DATA
import torchvision.transforms.functional as F

from torchvision.utils import _log_api_usage_once

from patchldmseg.utils.misc import Stage
from patchldmseg.utils import constants


class BraTSTransforms:
    r"""Wrapper class for all transformations required for a SubjectsDataset. This includes
    preprocessing and augmentation (spatial and intensity)

    Parameters
    ----------
    augment: bool
        Whether augmentation should be utilised to generate more samples
    dataset: str
        The utilised dataset
    diffusion: bool
        If the current utilised model is a diffusion model
    labels_in: dict
        Input label mapping between the key (string) depicting the type of label (e.g. 'bg'
        and the associated integer in the segmentation label (e.g. 1)
    labels_out: dict
        Mapping between the output classes (key, str) and the associated combination of input
        classes (value, tuple). E.g. the 'tumour' output class contains 'et', 'ncr', 'edema
    multiclass_pred: bool
        If multiclass or multilabel prediction should be utilised. This defines whether a
        one-hot or a multi-hot transformation should be applied to the label map.
    num_classes: int
        Number of classes for one-hot/multi-hot transformation. -1 infers the correct number of classes.
    resample: bool
        If resampling should be utilised. Refer to `torchio.transforms.Resample` for more information
    stage: Stage
        The stage of the experiment. One of Stage.TRAIN, Stage.VAL or Stage.TEST
    """
    def __init__(
        self,
        augment: bool,
        diffusion: bool,
        labels_in: Dict[str, int],
        labels_out: Dict[str, Tuple[str, ...]],
        multiclass_pred,
        num_classes: int,
        resample: bool,
        binary_min_max: Tuple[float, float],
        stage: Stage,
        output_size: Optional[Tuple[int, ...]] = None,
        ):

        self._stage = stage

        # Hyperparameters
        self._num_classes = num_classes
        self._resample = resample
        self._labels_in = labels_in
        self._labels_out = labels_out
        self._multiclass_pred = multiclass_pred
        self._diffusion = diffusion
        self._augment = augment
        self._binary_min_max = binary_min_max
        self._crop_size = output_size

        # Trafo specific
        self._preprocessing = None
        self._augmentation = None
        self._spatial = None
        self._intensity = None

    @property
    def stage(self) -> Stage:
        return self._stage

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def transformations(self) -> tt.Compose:
        trafos = [self.preprocessing, self.postprocessing]
        if self.stage == Stage.TRAIN:
            trafos.insert(1, self.augmentation)
        return tt.Compose(trafos)

    @property
    def preprocessing(self) -> tt.Compose:
        r"""Return the preprocessing transformations"""
        if self._preprocessing is None:
            self._preprocessing = self._compose_preprocessing()
        return self._preprocessing

    @property
    def augmentation(self) -> tt.Compose:
        r"""Return the augmentation transformations"""
        if self._augment:
            if self._augmentation is None:
                self._augmentation = tt.Compose([tt.RandomFlip(axes=0)])
        else:
            self._augmentation = tt.Compose([])

        return self._augmentation
    
    @property
    def postprocessing(self) -> "RotateSITK":
        return RotateSITK()

    def _compose_preprocessing(self) -> tt.Compose:
        transforms = []
        transforms.append(Rescale(
            out_min_max=(-1, 1)))

        remapping_dict = {}
        for lbl, lbl_idx in self._labels_in.items():
            if lbl in constants.BACKGROUND_STRINGS:
                value = lbl_idx
            else:
                value = 1
            remapping_dict[lbl_idx] = value

        transforms.append(tt.RemapLabels(remapping_dict))

        if self._crop_size is not None and len(self._crop_size) == 2:
            transforms.append(BraTSResize(output_size=self._crop_size))
        return tt.Compose(transforms)


class PILToTorchio:

    def __init__(self) -> None:
        from torchvision.utils import _log_api_usage_once
        _log_api_usage_once(self)

    def __call__(self, pic):
        """
        .. note::

            A deep copy of the underlying array is performed.

        Args:
            pic (PIL Image): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        tensor = F.pil_to_tensor(pic)

        # Now do the weird transpose() of torchio while loading the data
        tensor = torch.transpose(tensor, 1, 2).contiguous()

        if tensor.dim() < 4:
            diff_dim = 4 - tensor.dim()
            tensor = tensor[(...,) + (None,) * diff_dim]
        return tensor

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class RotateSITK(Transform):
    def __init__(self):
        super().__init__()

    def apply_transform(self, subject: Subject) -> Subject:
        for image in subject.get_images(intensity_only=False):
            image.set_data(self.rotate(image.data))
        return subject
    
    @staticmethod
    def is_invertible():
        return True
    
    def inverse(self):
        return self

    def rotate(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.transpose(tensor, 1, 2).contiguous()
    

class CenterCropResize(torch.nn.Module):
    def __init__(self, 
                 size: Tuple[int, int],
                 interpolation: F.InterpolationMode = F.InterpolationMode.BILINEAR):
        super().__init__()
        _log_api_usage_once(self)
        self.size = size
        self.interpolation = interpolation

    def forward(self, img):
        _, h, w = F.get_dimensions(img)
        min_size = min(h,w)

        center_cropped = F.center_crop(img, [min_size, min_size])

        return F.resize(center_cropped, list(self.size), interpolation=self.interpolation, antialias=True)
    

class BraTSResize(SpatialTransform):
    def __init__(self,
                 output_size: Tuple[int, int],
                 top_cut: int = 15,
                 bottom_cut: int = 20,):
        super().__init__()

        assert len(output_size) == 2, "This transformation is only for 2D sampling of 3D BraTS samples."
        self._input_size = (240, 240, 155)
        self._intermediate_size_divisible_by_2 = (256, 256, 128)
        scaling_factor = self._parse_output_size(self._intermediate_size_divisible_by_2, output_size)

        assert top_cut + bottom_cut == 35, "The sum of top and bottom cut must be 35"

        self._top_cut = top_cut
        self._bottom_cut = bottom_cut

        self.upscale_trafo = tt.Resize(target_shape=self._intermediate_size_divisible_by_2)
        self.output_trafo = tt.Resize(
            target_shape=(np.asarray(self._intermediate_size_divisible_by_2) / scaling_factor).tolist())

    @staticmethod
    def _parse_output_size(intermediate_size: Tuple[int, int, int],
                           output_size: Tuple[int, ...]) -> float:
        
        assert len(intermediate_size) >= len(output_size), "Output size has more dimensions that the supported size"
        assert all(os % 2 == 0 for os in output_size), "Output size must be divisible by 2"
        scaling_factor = set([intermediate_size[i]/output_size[i] for i in range(len(output_size))])

        assert len(scaling_factor) == 1, "Scaling factor must be the same in all dimensions"
        scaling_factor = scaling_factor.pop()

        return scaling_factor
    
    def apply_transform(self, subject: Subject) -> Subject:
        assert subject.spatial_shape == self._input_size, "Mismatch in input sizes"

        for image in self.get_images(subject):
            slice_indices = torch.arange(self._bottom_cut, image.shape[-1] - self._top_cut, 1)
            image.data = image.data[..., slice_indices]

        return self.output_trafo(self.upscale_trafo(subject))


class Rescale(NormalizationTransform):
    """Rescale intensity values to a certain range.

    Args:
        out_min_max: Range :math:`(n_{min}, n_{max})` of output intensities.
            If only one value :math:`d` is provided,
            :math:`(n_{min}, n_{max}) = (-d, d)`.
        percentiles: Percentile values of the input image that will be mapped
            to :math:`(n_{min}, n_{max})`. They can be used for contrast
            stretching, as in `this scikit-image example`_. For example,
            Isensee et al. use ``(0.5, 99.5)`` in their `nn-UNet paper`_.
            If only one value :math:`d` is provided,
            :math:`(n_{min}, n_{max}) = (0, d)`.
        masking_method: See
            :class:`~torchio.transforms.preprocessing.intensity.NormalizationTransform`.
        in_min_max: Range :math:`(m_{min}, m_{max})` of input intensities that
            will be mapped to :math:`(n_{min}, n_{max})`. If ``None``, the
            minimum and maximum input intensities will be used.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.

    Example:
        >>> import torchio as tio
        >>> ct = tio.ScalarImage('ct_scan.nii.gz')
        >>> ct_air, ct_bone = -1000, 1000
        >>> rescale = tio.RescaleIntensity(
        ...     out_min_max=(-1, 1), in_min_max=(ct_air, ct_bone))
        >>> ct_normalized = rescale(ct)

    .. _this scikit-image example: https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_equalize.html#sphx-glr-auto-examples-color-exposure-plot-equalize-py
    .. _nn-UNet paper: https://arxiv.org/abs/1809.10486
    """  # noqa: B950

    def __init__(
        self,
        out_min_max: TypeRangeFloat = (0, 1),
        percentiles: TypeRangeFloat = (0, 100),
        masking_method: TypeMaskingMethod = None,
        in_min_max: Optional[TypeRangeFloat] = None,
        **kwargs,
    ):
        super().__init__(masking_method=masking_method, **kwargs)
        self.out_min_max = out_min_max
        self.out_min, self.out_max = self._parse_range(
            out_min_max,
            'out_min_max',
        )
        self.percentiles = self._parse_range(
            percentiles,
            'percentiles',
            min_constraint=0,
            max_constraint=100,
        )

        if in_min_max is not None:
            self.in_min_max = self._parse_range(
                in_min_max,
                'in_min_max',
            )
            self.in_min, self.in_max = self.in_min_max
        else:
            self.in_min_max = None
            self.in_min = None
            self.in_max = None

        self.args_names = [
            'out_min_max',
            'percentiles',
            'masking_method',
            'in_min_max',
        ]
        self.invert_transform = False

    def is_invertible(self):
        return self.in_min_max is not None

    def apply_normalization(
        self,
        subject: Subject,
        image_name: str,
        mask: torch.Tensor,
    ) -> None:
        image = subject[image_name]
        image.set_data(self.rescale(image.data, mask, image_name))

    def rescale(
            self, tensor: torch.Tensor, 
            mask: torch.Tensor, 
            image_name: str) -> torch.Tensor:
        array = tensor.clone().float().numpy()
        mask = mask.numpy()
        if not mask.any():
            message = (
                f'Rescaling image "{image_name}" not possible'
                ' because the mask to compute the statistics is empty'
            )
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            return tensor
        values = array[mask] 
        cutoff = np.percentile(values, self.percentiles)
        np.clip(array, *cutoff, out=array)  # type: ignore[call-overload]
        if self.in_min_max is None:
            in_min, in_max = self._parse_range(
                (array.min(), array.max()),
                'in_min_max',
            )
        else:
            in_min, in_max = self.in_min_max
        
        in_range = in_max - in_min
        if in_range == 0:  # should this be compared using a tolerance?
            message = (
                f'Rescaling image "{image_name}" not possible'
                ' because all the intensity values are the same'
            )
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            return tensor
        out_range = self.out_max - self.out_min
        if self.invert_transform:
            array -= self.out_min
            array /= out_range
            array *= in_range
            array += in_min
        else:
            array -= in_min
            array /= in_range
            array *= out_range
            array += self.out_min
        return torch.as_tensor(array)
        

