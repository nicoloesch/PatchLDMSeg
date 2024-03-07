import abc
from typing import Any, Literal, Optional, Tuple, Dict, Iterable

from patchldmseg.utils.misc import TASK



class Dataset(abc.ABC):
    r"""Asbtract base class for ever image related attribute of an ImageDatset. Supports 
    super() initialisation through passing of **kwargs. 
    
    Parameters
    ----------
    channels : int
        The number of channels for each image. RGB has 3, BraTS usually has 4
    num_samples : int
        The number of samples that should be plotted.
    dimension : int
        Spatial dimensions of each image. 2D or 3D.
    binary_min_max : tuple of float
        The min and max values in a binary scale i.e. 2^n
    in_spatial_size : tuple of int, optional
        The spatial size of the input image. Can be optional for datasets where all images are not the same size.
    out_spatial_size : tuple of int
        The output spatial size of the dataset after cropping/sampling. Must match with dimension.
    shifted_min_max : tuple of float, optional
        The min and max values in a shifted scale. Currently not used.
    img_seq: iterable of str
        The sequence of the images that should be loaded. In natural images, this would be 'rgb', whereas 
        BraTS contains 't1', 't1ce', 't2', 'flair'.
    """
    def __init__(
            self,
            channels: int,
            num_samples: int,
            img_seq: Iterable[str],
            dimensions: Literal[2,3],
            binary_min_max: Tuple[float, float],
            in_spatial_size: Tuple[int, ...],
            out_spatial_size: Tuple[int, ...],
            shifted_min_max: Optional[Tuple[float, float]] = None,
            **kwargs,
            ):

        super().__init__(**kwargs)
        assert len(out_spatial_size) == dimensions, "Mismatch between number of output dimensions and output spatial size"
        self._binary_min_max = binary_min_max
        self._shifted_min_max = shifted_min_max
        self._dimensions = dimensions
        self._in_spatial_size = in_spatial_size
        self._out_spatial_size = out_spatial_size
        self._channels = channels
        self._num_samples = num_samples
        self._img_seq = img_seq

        self._num_train = 0
        self._num_val = 0
        self._num_test = 0
        self._num_pred = 0

    @property
    def binary_min_max(self) -> Tuple[float, float]:
        return self._binary_min_max

    @property
    def shifted_min_max(self) -> Optional[Tuple[float, float]]:
        return self._shifted_min_max

    @property
    def in_spatial_size(self) -> Tuple[int, ...]:
        return self._in_spatial_size

    @property
    def out_spatial_size(self) -> Tuple[int, ...]:
        return self._out_spatial_size
    
    @property
    def img_seq(self) -> Iterable[str]:
        return self._img_seq
    
    @property
    def num_train(self) -> int:
        return self._num_train
    
    @property
    def num_val(self) -> int:
        return self._num_val
    
    @property
    def num_test(self) -> int:
        return self._num_test
    
    @property
    def num_pred(self) -> int:
        return self._num_pred
    
    @abc.abstractmethod
    def train_dataset(self, is_distributed: bool):
        r"""Return the train_dataset, which is created by the train split
        
        Parameter
        ---------
        is_distributed : bool
            Whether one utilises DDP or not. Only relevant for BraTS as the Queue object requires the subject_sampler.
        """

    @abc.abstractmethod
    def val_dataset(self, is_distributed: bool):
        r"""Return the val_dataset, which is created by the val split.
        
        Parameter
        ---------
        is_distributed : bool
            Whether one utilises DDP or not. Only relevant for BraTS as the Queue object requires the subject_sampler.
        """

    @abc.abstractmethod
    def test_dataset(self, is_distributed: bool):
        r"""Return the test_dataset, which is created by the test split.
        
        Parameter
        ---------
        is_distributed : bool
            Whether one utilises DDP or not. Only relevant for BraTS as the Queue object requires the subject_sampler.
        """

    @abc.abstractmethod
    def pred_dataset(self, is_distributed: bool):
        r"""Return the predict_dataset, which is created by the predict split. This is the only dataset that does not 
        have any ground-truth labels
        
        Parameter
        ---------
        is_distributed : bool
            Whether one utilises DDP or not. Only relevant for BraTS as the Queue object requires the subject_sampler.
        """

    @abc.abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        r"""Method to save the state of the dataset"""

    @abc.abstractmethod
    def load_state_dict(self, state_dict: Dict[str, Any]):
        r"""Method to restore the state of the dataset based on the saved state_dict"""

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def channels(self) -> int:
        return self._channels
    
    @property
    def num_samples(self) -> int:
        return self._num_samples
    
    @staticmethod
    def _parse_num_samples(
            total_label: int, total_unlabel: int, 
            num_train: int, num_val: int, num_test: int, num_pred: int,
            task: TASK,
    ) -> Tuple[int, int, int, int]:

        num_train, num_val, num_test = Dataset._parse_num_label(total_label, num_train, num_val, num_test, task)
        num_pred = Dataset._parse_num_nonlabel(total_unlabel, num_pred)
        return num_train, num_val, num_test, num_pred

    @staticmethod
    def _parse_num_label(total_label: int, num_train: int, num_val: int, num_test: int, task: TASK) -> Tuple[int, int, int]:
        r"""This function automatically determines the size for each of the labelled subsets (train, val, test). 
        Providing -1 for each of the subsets results in a 80/10/10 split. Providing a positive integer will result
        in sampling exactly that amount of samples, provided that not more samples are requested than available."""
        num_val = Dataset._parse_auto(total_label, num_val, 0.1)
        num_test = Dataset._parse_auto(total_label, num_test, 0.1)
        num_train = total_label - num_val - num_test if num_train == -1 else num_train
        assert sum(
            [num_train, num_val, num_test]
            ) <= total_label, f'Selected too many samples (train {num_train} + val {num_val} + test {num_test} > total {total_label})'
        return num_train, num_val, num_test

    @staticmethod
    def _parse_num_nonlabel(total_nonlabel: int, num_pred: int) -> int:
        r"""
        This function automatically determines the size for the unlabelled subset (pred). Providing -1 results in
        a 100% split. Providing a positive integer will result in sampling exactly that amount of samples.
        """
        return Dataset._parse_auto(total_nonlabel, num_pred, 1.0)
    
    @staticmethod
    def _parse_auto(total: int, num: int, factor: float) -> int:
        if num == -1:
            num = int(factor * total)
        elif num >= 0:
            assert num <= total, 'Selected too many samples'
        else:
            raise NotImplementedError(f"num must be [-1, {total}]")
        return num