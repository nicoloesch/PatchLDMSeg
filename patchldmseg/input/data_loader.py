
from typing import Optional, Dict, Tuple, Union, Any, Literal
import pathlib
import abc

import torchio as tio
from torch.utils.data import DataLoader, DistributedSampler
import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy

import patchldmseg.input.datasets as pid
import patchldmseg.utils.misc as pum



class DataModule(pl.LightningDataModule, abc.ABC):

    """ Pytorch LightningDataModule for all Datasets.

    Parameters
    ----------
    augment: bool
        If augmentation of the data should be utilised.
    batch_size: int
        The batch size of each step
    conditional_sampling : bool
        If conditional sampling based on the label should be used
    dataset: str
        The utilised dataset. Supported: Brats2023
    datasets_root: str
        Absolute path to where all datasets are stored
    diffusion: bool
        If a diffusion model for the associated `task` is specified by --model. It is required
        for downstream functions to work as the model class can currently not be retrieved
        from the argparser
    dimensions: int
        Dimensionality of the input. Supported: 2,3
    fg_prob: int
        How much more likely a foreground class will be sampled from the entire set. Only applies to
        non-Diffusion models.
    num_workers: int
        Number of subprocesses to use for data loading.
    multiclass_pred: bool
        Determines if multiclass prediction of multilabel should be used
    patches_per_subj: int
        Number of sampled patches/slices per subject and the associated volumes.
    patch_size : int
        The patch size sampled for each subject. Only applies to `dimensions=3` and is otherwise
        omitted.
    patch_overlap: int
        Overlap between patches for the test step in the `torchio.GridAggregator`. Higher overlap
        smoothens out dissimilarities on the edge between two patches (somewhat like averaging).
    resample: bool
        Whether resampling to a specific resolution should be utilised.
    task: TASK
        The selected task of the model.
    to_ras: bool
        If the transformation to RAS should be used. See `torchio.transforms.preprocessing.ToCanonical`
    num_test: int
        Number of test subjects. -1 results in the maximum of available test samples of the dataset
    num_train: int
        Number of training samples. -1 results in 90% of the available training samples of the dataset if 
        num_val is also -1. If num_val is 0, it results in the entire train dataset and if num_val > 0, it 
        results in the residual number of samples left after the num_val samples have been subtracted.
    num_val: int
        Number of validation subjects. -1 results in 10% of the available training samples of the dataset if
        num_train is also -1. If num_train is 0, it results in the entire train dataset. Special case:
        num_train + num_val > num_total_train results in the copying of the training dataset for the validation.
        This is only advised in generative sampling at the moment and will be removed in future updates.
    """
    def __init__(self,
                 augment: bool,
                 batch_size: int,
                 conditional_sampling: bool,
                 dataset_str: Literal["BraTS_2023"],
                 datasets_root: str,
                 diffusion: bool,
                 dimensions: Literal[2, 3],
                 in_channels: int,
                 num_workers: int,
                 patches_per_subj: int,
                 patch_overlap: int,
                 patch_size: Union[int, Tuple[int, ...]],
                 task: pum.TASK,
                 fg_prob: float = 1.0,
                 num_test: int = -1,
                 num_train: int = -1,
                 num_val: int = -1,
                 num_pred: int = -1,
                 brats_2023_subtask: Optional[Literal['Glioma', 'Pediatric']] = None,
                 pin_memory: bool = True,
                 use_queue: bool = False,
                 multiclass_pred: bool = True,
                 resample: bool = False):
        assert pathlib.Path(datasets_root, dataset_str).is_dir(), f"Dataset {dataset_str} not found in {datasets_root}"

        # This needs to be set as argument linking does not work otherwise. 
        # And we need to overwrite the hparams
        self.patch_size = patch_size = self._parse_patch_size(patch_size, dimensions, dataset_str)  #type: ignore
        super().__init__()
        self.save_hyperparameters()

        self.dataset: pid.Dataset = {
            'BraTS_2023': pid.BraTS2023
            }[dataset_str](**self.hparams)  #type: ignore
        
        if 'brats' in dataset_str.lower():
            self._set_hparam('num_workers', 0)  


    def _get_ddp_sampler_and_shuffle(
            self, dataset, stage: pum.Stage) -> Tuple[Optional[DistributedSampler], bool]:
        is_brats = isinstance(self.dataset, pid.BaseBraTS)
        if self.is_distributed and not is_brats:
            sampler = DistributedSampler(dataset)
        else:
            sampler = None
        return sampler, sampler is None and not is_brats and stage==pum.Stage.TRAIN
        

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(pum.Stage.TRAIN)
    
    def val_dataloader(self) -> DataLoader:
        return self._dataloader(pum.Stage.VAL)
    
    def test_dataloader(self) -> DataLoader:
        return self._dataloader(pum.Stage.TEST)
    
    def predict_dataloader(self) -> DataLoader:
        return self._dataloader(pum.Stage.PRED)

    def _dataloader(self, stage: pum.Stage) -> DataLoader:
        r"""Interface for all dataloaders"""
        dataset = getattr(self.dataset, f"{stage.value}_dataset")(self.is_distributed)
        sampler, shuffle = self._get_ddp_sampler_and_shuffle(dataset, stage)
        return DataLoader(
            dataset, 
            batch_size=self._get_hparam('batch_size'), 
            num_workers=self._get_hparam('num_workers'),
            sampler=sampler,
            shuffle=shuffle,
            pin_memory=self._get_hparam("pin_memory"))

    @property
    def num_train(self) -> int:
        return self.dataset.num_train

    @property
    def num_val(self) -> int:
        return self.dataset.num_val

    @property
    def num_test(self) -> int:
        return self.dataset.num_test
    
    @property
    def num_pred(self) -> int:
        return self.dataset.num_pred
    
    
    @property
    def is_distributed(self) -> bool:
        if self.trainer is not None:
            return isinstance(self.trainer.strategy, DDPStrategy)
        else:
            return False
        
    @property
    def drop_last(self) -> bool:
        return True
    
    def state_dict(self) -> Dict[str, Any]:
        r"""The state of the datamodule that is stored upon saving the checkpoint. This includes
        all necessary attributes to restore the state and continue training with exactly the same datamodule"""

        state_dict = {'hparams': self.hparams,
                      'patch_size': self.patch_size,
                      'dataset': self.dataset.state_dict()}
        return state_dict        

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        r"""Loads the state of the datamodule through the state_dict and restores it."""
        self.patch_size = state_dict.get('patch_size')
        self.dataset = self.dataset.load_state_dict(state_dict.get('dataset'))
    
    def _get_hparam(self, key: str) -> Any:
        try:
            return getattr(self.hparams, key)
        except AttributeError:
            raise AttributeError(
                f"Attribute {key} not found in hparams. Please check the"
                f"spelling and make sure you called `self.save_hyperparameters()` in __init__ of the child class"
                )
        
    def _set_hparam(self, key: str, value: Any) -> None:
        try:
            setattr(self.hparams, key, value)
        except AttributeError:
            raise AttributeError("`self.hparams` not found. Make sure you called `self.save_hyperparameters()`")
    
    @staticmethod
    def _parse_patch_size(patch_size: Union[int, Tuple[int, ...]], dimension: int, dataset: str) -> Tuple[int, ...]:
        r"""This function alters the patch size depending on the dataset and the dimensionality."""
        if dimension == 2:
            if isinstance(patch_size, int):
                return (patch_size, ) * dimension
            elif isinstance(patch_size, tuple):
                assert len(patch_size) == dimension, "Patch size for 2D images must be a tuple of length 2"
                return patch_size
            else:
                raise RuntimeError("Unsupported type of patch_size")
        elif dimension == 3:
            assert 'brats' in dataset.lower(), "3D currently only supported for BraTS"
            if isinstance(patch_size, int):
                return (patch_size, patch_size, patch_size)
            elif isinstance(patch_size, tuple):
                _error_msg = "Patch size for 3D BraTS images must be a tuple of length 3 with all items being equal."
                assert len(patch_size) == 3 and len(set(patch_size)) == 1, _error_msg
                return patch_size
        else:
            raise NotImplementedError(f"Dimension {dimension} not supported")