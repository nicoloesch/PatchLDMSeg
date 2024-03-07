import SimpleITK as sitk
import abc
import copy
import pathlib
import random

from typing import (
    Any, 
    Generator, 
    Dict, 
    List, 
    Literal, 
    Optional, 
    Tuple, 
    Union
)

import patchldmseg.utils.misc as aum
from patchldmseg.input.transforms import BraTSTransforms
from patchldmseg.utils import constants
from patchldmseg.input.sampler import BraTSDiffusionSampler, GridAggregatorSD, GridSamplerSD
from patchldmseg.input import parse_data
from patchldmseg.input.datasets.base_dataset import Dataset


import numpy as np
import pandas as pd
from torch.utils.data import DistributedSampler
from torch.utils.data.dataset import Dataset as TorchDataset
import torchio as tio
from torchio.data.subject import Subject
from torchio.constants import NUM_SAMPLES
from torchio.typing import TypeSpatialShape
  

class Aggregators:
    def __init__(
            self,
            pred_aggregator: Optional[GridAggregatorSD],
            target_aggregator: Optional[GridAggregatorSD],
            original_aggregator: Optional[GridAggregatorSD],
            recon_aggregator: Optional[GridAggregatorSD],
            fg_mask_aggregator: Optional[GridAggregatorSD]):
        
        self._aggr_dict = {
            'pred': pred_aggregator,
            'target': target_aggregator,
            'original': original_aggregator,
            'recon': recon_aggregator,
            'fg_mask': fg_mask_aggregator
        }

    @property
    def pred(self) -> Optional[GridAggregatorSD]:
        return self._aggr_dict['pred']

    @property
    def target(self) -> Optional[GridAggregatorSD]:
        return self._aggr_dict['target']

    @property
    def original(self) -> Optional[GridAggregatorSD]:
        return self._aggr_dict['original']
    
    @property
    def recon(self) -> Optional[GridAggregatorSD]:
        return self._aggr_dict['recon']
    
    @property
    def fg_mask(self) -> Optional[GridAggregatorSD]:
        return self._aggr_dict['fg_mask']
    
    @property
    def aggregator_names(self) -> List[str]:
        return list(self._aggr_dict.keys())
    
    def add_aggregator(self, name: str, grid_aggr: Optional[GridAggregatorSD] = None) -> None:
        r"""Function that adds an aggregator"""
        if grid_aggr is None:
            # We can check if we can clone one of the other aggregators if they have not already been filled
            for aggr in self._aggr_dict.values():
                if isinstance(aggr, GridAggregatorSD) and aggr._output_tensor is not None:
                    grid_aggr = copy.deepcopy(aggr)
                    break
        assert grid_aggr is not None, "Provide either a valid grid aggregator or add the aggregator at the start where all aggregators are instantiated"

        self._aggr_dict[name] = grid_aggr

    
    @classmethod
    def init_from_sampler(cls, sampler: GridSamplerSD):
        from copy import deepcopy
        assert isinstance(sampler, GridSamplerSD), f"Wrong sampler type ({type(sampler)})"
        grid_aggregator = GridAggregatorSD(sampler, overlap_mode='hann')
        return cls(
            pred_aggregator=grid_aggregator, 
            target_aggregator=deepcopy(grid_aggregator),
            original_aggregator=deepcopy(grid_aggregator),
            recon_aggregator=deepcopy(grid_aggregator),
            fg_mask_aggregator=deepcopy(grid_aggregator))

    
class BraTSSubjectsSubset:
    """ Struct to contain all important information about a subject subset
    for BraTS dataloaders."""

    def __init__(self,
                 subject_list: List[Subject],
                 stage: aum.Stage):
        self._subject_list = subject_list
        self._aggregators = None
        self._stage = stage
        self._transformations = None
        self._dataset = None

    def __getitem__(self, i) -> Dict:
        return self._subject_list[i]

    def __iter__(self):
        return iter(self._subject_list)
    
    @property
    def transformations(self) -> Optional[BraTSTransforms]:
        return self._transformations

    @transformations.setter
    def transformations(self, trafo: Optional[BraTSTransforms]):
        self._transformations = trafo

    @property
    def dataset(self) -> Optional[TorchDataset]:
        return self._dataset

    @dataset.setter
    def dataset(self, ds: Optional[TorchDataset]):
        self._dataset = ds

    @property
    def subject_list(self) -> List[Subject]:
        """Create Subject List required for the Dataloader with multiprocessing support for faster processing.

        Returns
        -------
        List of Subject
        """
        return self._subject_list

    @property
    def stage(self) -> aum.Stage:
        return self._stage

    def __repr__(self):
        return repr(f"Num Subjects: {len(self._subject_list)}")

    def __len__(self):
        return len(self._subject_list)

    @property
    def aggregators(self) -> Optional[Aggregators]:
        return self._aggregators

    @aggregators.setter
    def aggregators(self, aggr: Aggregators):
        self._aggregators = aggr

    @classmethod
    def from_ckpt(cls, subjects: Dict[str, Dict[str, str]], stage: aum.Stage, root_dir: str, mri_sequences: Tuple[str, ...]):
        r"""This method creates a new instance of subjects subset by providing the option
        to have a new data_dir path as loading from a checkpoint could result in the situation of having
        two different directories of the data if executed on different servers."""
        subjects_list = list(BaseBraTS._create_subjects(dataset_dir=root_dir, file_dict=subjects, mri_sequences=mri_sequences))
        return cls(subject_list=subjects_list, stage=stage)
    
    def to_json(self, root_dir: str) -> Dict[str, Dict[str, str]]:
        r"""Extracts the sequence and the relative path of each image with respect to the base root_dir for each subject of this subset"""
        return {subject['subject_id']: {str(seq): str(image.path.relative_to(root_dir)) for seq, image in subject.get_images_dict(intensity_only=False).items()} for subject in self.subject_list}

    def resample(self, num_samples: int):
        r"""This function resamples the already instantiated dataset based on a new number of samples"""

        if num_samples >= 0:
            self._subject_list = random.sample(self.subject_list, min(len(self), num_samples))
        elif num_samples == -1:
            pass
        else:
            raise RuntimeError(f"Wrong number of samples ({num_samples})")
    
class BaseBraTS(Dataset):
    r""" Base ABC for all BraTS datasets. Implements functionality applicable to 
    each BraTS dataset.

    Parameters
    ----------
    augment: bool
        If augmentation of the data should be utilised.
    batch_size: int
        The batch size of each step
    conditional_sampling : bool
        If conditional sampling based on the label should be used
    data_dir: str
        Absolute path to the data directory.
    diffusion: bool
        If a diffusion model for the associated `task` is specified by --model. 
        It is required for downstream functions to work as the model class can 
        currently not be retrieved from the argparser
    dimensions: int
        Dimensionality of the input. Supported: 2,3
    fg_prob: int
        How much more likely a foreground class will be sampled from the entire set. 
        Only applies to non-Diffusion models.
    num_workers: int
        Number of subprocesses to use for data loading.
    multiclass_pred: bool
        Determines if multiclass prediction of multilabel should be used
    patches_per_subj: int
        Number of sampled patches/slices per subject and the associated volumes.
    patch_size : int
        The patch size sampled for each subject.
    patch_overlap: int
        Overlap between patches for the test step in the `torchio.GridAggregator`. Higher overlap
        smoothens out dissimilarities on the edge between two patches (somewhat like averaging).
    task: Task
        Which task the model performs
    num_test: int, optional
        Number of test subjects. -1 results in the usage of 10 % of the labeled dataset. 
    num_train: int, optional
        Number of training subjects. -1 results in the usage of 80% of the labeled dataset if num_val is -1
        and 100 % if num_val is 0. Any value above -1 will result in the absolute number specified
        (unless num_train > available train subjects)
    num_val: int, optional
        Number of validation subjects. -1 results in the usage of 10% of the labeled dataset if num_train is -1. 
        Any value above -1 will result in the absolute number specified (unless num_val > available train subjects). 
        Special case: specifying num_val = num_train and num_val + num_train > available train subjects 
        will result in the copy of the training set
    
    """
    LABEL_SUFFIX = '_seg.nii.gz'
    IMAGE_SUFFIX = '_image.nii.gz'
    def __init__(self,
                augment: bool,  # noqa: F841
                batch_size: int,  # noqa: F841
                conditional_sampling: bool,  # noqa: F841
                root_dir: str,  # noqa: F841
                diffusion: bool,  # noqa: F841
                dimensions: Literal[2, 3],  # noqa: F841
                in_channels: int,
                fg_prob: float,  # noqa: F841
                multiclass_pred: bool,  # noqa: F841
                num_workers: int,  # noqa: F841
                patches_per_subj: int,  # noqa: F841
                patch_overlap: int,  # noqa: F841
                patch_size: Tuple[int, ...],  # noqa: F841
                task: aum.TASK,  # noqa: F841
                num_test: int = -1,
                num_train: int = -1,
                num_val: int = -1,
                num_pred: int = -1,
                **kwargs):

        assert in_channels == 4, 'BraTS dataset only supports 4 channels'

        # Get the entire image in the 2D case
        self._patch_size = patch_size
        self._augment = augment
        self._batch_size = batch_size
        self._conditional_sampling = conditional_sampling
        self._root_dir = root_dir
        self._diffusion = diffusion
        self._fg_prob = fg_prob
        self._multiclass_pred = multiclass_pred
        self._num_workers = num_workers
        self._patches_per_subj = patches_per_subj
        self._patch_overlap = patch_overlap
        self._task = task

        self._preprocess_crop_size = None if dimensions == 3 else tuple(patch_size + (155, ))

        # Load the dataset
        self._loaded_json = self._populate_json(data_dir=root_dir, verbose=False, force=False)

        super().__init__(
            channels=in_channels, 
            dimensions=dimensions, 
            binary_min_max=(0., 32767.), 
            in_spatial_size=(240, 240, 155),
            out_spatial_size=patch_size,
            img_seq=tuple(self._loaded_json['modality'].values()),
            num_samples=4)

        # Labeled examples
        total_label = list(
            self._create_subjects(
                dataset_dir=root_dir,
                file_dict=self._loaded_json['train'],
                mri_sequences=tuple(self._loaded_json['modality'].values())))
        
        # Unlabeled examples
        total_pred = list(
            self._create_subjects(
                dataset_dir=root_dir,
                file_dict=self._loaded_json['test'],
                mri_sequences=tuple(self._loaded_json['modality'].values())))

        self._check_parsed_samples(num_total_label=len(total_label), num_total_nonlabel=len(total_pred))

        self._num_train, self._num_val, self._num_test, self._num_pred = (
            self._parse_num_samples(
                total_label=len(total_label), 
                total_unlabel=len(total_pred), 
                num_train=num_train, 
                num_val=num_val, 
                num_test=num_test, 
                num_pred=num_pred, 
                task=task))

        self._modality = self._loaded_json['modality']
        self._labels_in: Dict[str, int] = self._loaded_json.get('labels_in', {})
        self._labels_out: Dict[str, Tuple[str, ...]] = self._loaded_json.get('labels_out', {})

        if self.is_diffusion:
            self._labels_out = {'tumour': ('et', 'ncr', 'edema')}

        subsets = self._sample_subsets({'label': total_label, 'pred': total_pred})
        self._subjects_train: BraTSSubjectsSubset = subsets[aum.Stage.TRAIN]
        self._subjects_val: BraTSSubjectsSubset = subsets[aum.Stage.VAL]
        self._subjects_test: BraTSSubjectsSubset = subsets[aum.Stage.TEST]
        self._subjects_pred: BraTSSubjectsSubset = subsets[aum.Stage.PRED]

    @property
    def is_diffusion(self) -> bool:
        return self._diffusion

    @staticmethod
    def shuffle_patches(stage: aum.Stage) -> bool:
        return stage == aum.Stage.TRAIN

    @property
    def drop_last(self) -> bool:
        return True

    @property
    def subjects_train(self) -> BraTSSubjectsSubset:
        return self._subjects_train
    
    @subjects_train.setter
    def subjects_train(self, subjects: BraTSSubjectsSubset):
        assert isinstance(subjects, BraTSSubjectsSubset), f'Wrong type {type(subjects)}'
        self._subjects_train = subjects

    @property
    def subjects_val(self) -> BraTSSubjectsSubset:
        return self._subjects_val
    
    @subjects_val.setter
    def subjects_val(self, subjects: BraTSSubjectsSubset):
        assert isinstance(subjects, BraTSSubjectsSubset), f'Wrong type {type(subjects)}'
        self._subjects_val = subjects

    @property
    def subjects_test(self) -> BraTSSubjectsSubset:
        return self._subjects_test
    
    @subjects_test.setter
    def subjects_test(self, subjects: BraTSSubjectsSubset):
        assert isinstance(subjects, BraTSSubjectsSubset), f'Wrong type {type(subjects)}'
        self._subjects_test = subjects

    @property
    def subjects_pred(self) -> BraTSSubjectsSubset:
        return self._subjects_pred
    
    @subjects_pred.setter
    def subjects_pred(self, subjects: BraTSSubjectsSubset):
        assert isinstance(subjects, BraTSSubjectsSubset), f'Wrong type {type(subjects)}'
        self._subjects_pred = subjects

    @abc.abstractmethod
    def _check_parsed_samples(self, num_total_label: int, num_total_nonlabel: int)-> None:
        r"""Checks if all samples are part of the json dataset.
        Parameters
        ----------
        num_total_label : int
            The number of total labeled samples. Used for training, validation and testing
        num_total_nonlabel : int
            The number of total unlabeled samples. Used for inference in the prediction step
        """
        pass

    @abc.abstractmethod
    def is_incomplete(self, num_label_loaded: Optional[int], num_nonlabel_loaded: Optional[int]) -> bool:
        r"""Checks if the json file is incomplete."""
        pass

    def train_dataset(self, is_distributed: bool) -> Optional[TorchDataset]:
        if self._subjects_train.dataset is None:
            self._subjects_train.dataset, self._subjects_train.transformations = self._create_queue_and_trafo(
                stage=aum.Stage.TRAIN, 
                is_distributed=is_distributed)
        return self._subjects_train.dataset

    def val_dataset(self,  is_distributed: bool) -> Optional[TorchDataset]:
        if self._subjects_val.dataset is None:
            self._subjects_val.dataset, self._subjects_val.transformations = self._create_queue_and_trafo(
                stage=aum.Stage.VAL, is_distributed=is_distributed)
        return self._subjects_val.dataset

    def test_dataset(self,  is_distributed: bool, use_queue: bool = False) -> Optional[TorchDataset]:
        if self._subjects_test.dataset is None:
            self._subjects_test.dataset, self._subjects_test.transformations = self._create_queue_and_trafo(
                stage=aum.Stage.TEST, is_distributed=is_distributed)
        return self._subjects_test.dataset
    
    def pred_dataset(self,  is_distributed: bool, use_queue: bool = False) -> Optional[TorchDataset]:
        if self._subjects_pred.dataset is None:
            self._subjects_pred.dataset, self._subjects_pred.transformations = self._create_queue_and_trafo(
                stage=aum.Stage.PRED, is_distributed=is_distributed)
        return self._subjects_pred.dataset

    def state_dict(self) -> Dict[str, Any]:
        state_dict = {}

        # Subjects
        for stage in aum.Stage:
            subjects_subset: BraTSSubjectsSubset = getattr(self, f"subjects_{stage.value}")
            subjects_dict = {"stage": stage.value,
                             "subjects": subjects_subset.to_json(root_dir=self._root_dir)
                             }
            state_dict[f"subjects_{stage.value}"] = subjects_dict

        # Json Parameters
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> "BaseBraTS":
        #ckpt_hparams = state_dict.get('hparams')  # Currently not sure if I need them as they are loaded from the config.json
        state_dict_has_pred = "subjects_pred" in state_dict.keys()

        for stage in aum.Stage:
            subject_state_dict = state_dict.pop(f"subjects_{stage.value}")
            assert subject_state_dict['stage'] == stage.value, f"Incompatible stage value ({subject_state_dict['stage']} and {stage.value})"
            subject_subset: BraTSSubjectsSubset = BraTSSubjectsSubset.from_ckpt(
                subjects=subject_state_dict['subjects'],
                stage=stage,
                root_dir=self._root_dir,
                mri_sequences=tuple(self._modality.values()))
            
            if not state_dict_has_pred:
                if stage == aum.Stage.VAL:
                    split_arrays = np.array_split(subject_subset.subject_list, 2)
                    
                    subject_subset = BraTSSubjectsSubset(split_arrays[0].tolist(), stage=stage)
                    subjects_test = BraTSSubjectsSubset(subject_list=split_arrays[1].tolist(), stage=aum.Stage.TEST)
                    assert isinstance(subjects_test, list), "Subjects test is not a list"
                    setattr(self,  f"subjects_{aum.Stage.TEST.value}", subjects_test)
                    
                elif stage == aum.Stage.TEST:
                    subjects_val = subject_subset[:len(subject_subset) // 2]
                    assert isinstance(subjects_val, list), "Subjects val is not a list"
                    subjects_pred = BraTSSubjectsSubset(subject_list=subjects_val, stage=aum.Stage.PRED)
                    subjects_pred.resample(num_samples=self._num_pred)
                    setattr(
                        self, 
                        f"subjects_{aum.Stage.PRED.value}",
                        subjects_pred)
                    continue

            num_samples = getattr(self, f"_num_{stage.value}")
            subject_subset.resample(num_samples)
            setattr(self, f"subjects_{stage.value}", subject_subset)
        return self

    @staticmethod
    def _create_subjects(
            dataset_dir: str,
            file_dict: Dict[str, Dict[str, str]],
            mri_sequences: Tuple[str, ...]) -> Generator[Subject, None, None]:
        r"""Creates BraTSDiffusionSubjects based on the file_dict
        obtained from the loaded json."""

        for subject_id, json_subject_dict in file_dict.items():
            subject_dict = {}
            for k, v in json_subject_dict.items():
                if k in mri_sequences:
                    subject_dict[k] = tio.ScalarImage(pathlib.Path(dataset_dir, v))
                elif k  == constants.TARGET_SEG_DESCR:
                    subject_dict[k] = tio.LabelMap(pathlib.Path(dataset_dir, v))
                else:
                    if v is not None and k != NUM_SAMPLES:
                        subject_dict[k] = v
            subject_dict['subject_id'] = subject_id
            yield Subject(**subject_dict)
 
    def _sample_subsets(
        self, 
        subjects_total: Dict[str, List[Subject]]
        ) -> Dict[aum.Stage, BraTSSubjectsSubset]:
        r"""Subsamples the subjects for each of the stages.
        
        Parameters
        ----------
        subjects_total : dict
            The dictionary containing all subjects for each of the two possible stages ('train' and 'test').
        """
        ret = {}

        for stage in [st for st in aum.Stage]:
            num_samples = getattr(self, f"_num_{stage.value}")
            subset_specifier = 'label' if stage != aum.Stage.PRED else 'pred'

            if len(subjects_total[subset_specifier]) < num_samples:
                assert self._task == aum.TASK.GEN and stage in [aum.Stage.VAL, aum.Stage.TEST], f"Too many samples selected for the {stage.value} set and the task {self._task.value}"
                ret[aum.Stage.VAL] = copy.deepcopy(ret[aum.Stage.TRAIN])
                ret[aum.Stage.TEST] = copy.deepcopy(ret[aum.Stage.TRAIN])

            else:
                elements = random.sample(subjects_total[subset_specifier], num_samples)
                ret[stage] = BraTSSubjectsSubset(elements, stage)
                for el in elements:
                    subjects_total[subset_specifier].pop(subjects_total[subset_specifier].index(el))

                current_set_ids = set([el['subject_id'] for el in ret[stage]])
                sampled_ids = [set([el['subject_id'] for el in value]) if key != stage else set() for key, value in ret.items()]
                intersection = [len(current_set_ids.intersection(sampled_ids[i])) for i in range(len(sampled_ids))]
                assert all([el == 0 for el in intersection]), f"Sampled overlapping samples, which is not permitted"
        return ret

    def _create_queue_and_trafo(
            self,
            stage: aum.Stage,
            is_distributed: bool,
            num_classes: int = -1,
            ) -> Tuple[Optional[TorchDataset], Optional[BraTSTransforms]]:
        r"""Instantiates the queue and the transformations for the subjects.
        In the case of use_queue = False, a wrapper for the dataset is provided that only returns a single patch for each subject.
        This should circumvent the DDP issues."""

        transforms = BraTSTransforms(
            augment=self._augment,
            diffusion=self.is_diffusion,
            labels_in=self._labels_in,
            labels_out=self._labels_out,
            multiclass_pred=self._multiclass_pred,
            binary_min_max=self.binary_min_max,
            output_size=self._patch_size,
            resample=False,
            stage=stage,
            num_classes=num_classes)

        subjects_subset: BraTSSubjectsSubset = getattr(self, f"subjects_{stage.value}")
        subjects_list = subjects_subset.subject_list

        subjects_dataset = tio.SubjectsDataset(subjects_list,
                                               transform=transforms.transformations)

        if len(subjects_list) == 0:
            return None, None

        # Instantiate the sampler
        if stage in [aum.Stage.TRAIN, aum.Stage.VAL]:
            spatial_size = None
            patch_overlap = None

        else:
            spatial_size = self.in_spatial_size
            patch_overlap = self._patch_overlap

        sampler = self._create_sampler(dimensions=self.dimensions,
                                       dummy_subject=tio.Subject(subjects_list[0]),
                                       patches_per_subject=self._patches_per_subj,
                                       diffusion=self.is_diffusion,
                                       spatial_size=spatial_size,
                                       patch_size=self._patch_size,
                                       patch_overlap=patch_overlap,
                                       stage=stage,
                                       task=self._task)

        if isinstance(sampler, GridSamplerSD):
            grid_sampled_subjects: BraTSSubjectsSubset = getattr(self, f"subjects_{stage.value}")
            setattr(grid_sampled_subjects, "aggregators", Aggregators.init_from_sampler(sampler))
            max_length = samples_per_volume = sampler.num_samples
        elif isinstance(sampler, (
            BraTSDiffusionSampler, 
            tio.sampler.LabelSampler,
            tio.sampler.UniformSampler)):
            num_workers = self._num_workers if self._num_workers > 0 else 1  
            max_length = num_workers * self._patches_per_subj * 2
            samples_per_volume = self._patches_per_subj
        else:
            raise RuntimeError(f"Not supported sampler type for BraTS dataset ({type(sampler)})")

        # Instantiate the Queue
        if is_distributed:
            subjects_sampler = DistributedSampler(subjects_dataset,
                                                shuffle=True,
                                                drop_last=self.drop_last)
            shuffle_subjects = False
        else:
            subjects_sampler = None
            shuffle_subjects = stage == aum.Stage.TRAIN

        queue = tio.Queue(
            subjects_dataset=subjects_dataset,
            max_length=max_length,
            samples_per_volume=samples_per_volume,
            sampler=sampler,
            subject_sampler=subjects_sampler,
            num_workers=self._num_workers,
            shuffle_subjects=shuffle_subjects,
            shuffle_patches=self.shuffle_patches(stage),
            start_background=True
        )

        return queue, transforms

    def _create_sampler(
            self,
            dimensions: int,
            stage: aum.Stage,
            patches_per_subject: int,
            task: aum.TASK,
            diffusion: Optional[bool] = None,
            patch_size: Optional[TypeSpatialShape] = None,
            dummy_subject: Optional[tio.Subject] = None,
            patch_overlap: Optional[Union[int, tuple]] = None,
            spatial_size: Optional[tuple] = None) -> Union[tio.sampler.LabelSampler,
                                                           tio.sampler.UniformSampler,
                                                           GridSamplerSD, 
                                                           BraTSDiffusionSampler]:
        r"""This function determines the sampler object required for the Queue.
        It unifies the sampler creation for all cases and is therefore unfortunately
        a rather complex function.

        Parameters
        ----------
        dimensions : int
            The dimensions of the following operations. Can be either 2 or 3 and determines the sampler
        stage : Stage
            The current stage the setup is in
        label_prob : dict, optional
            The optional dictionary of label probabilities required for the various samplers.
            Is required for 2D case and in 3D for train and val
        diffusion : bool, optional
            Whether we have a diffusion process here. Required for 3D case as it determines the
            sampler
        patch_size : optional
            The patch size required for all 3D cases
        dummy_subject : tio.Subject, optional
            GridSampler only. A dummy subject required to initialise 
            the GridSampler in the test case 3D
        patch_overlap : optional
            The patch overlap for the GridSampler in the test case 3D
        spatial_size : tuple, optional
            GridSampler only. The respective spatial size of all subjects. 
        """
        test_diff_sampler = stage in (aum.Stage.TEST, aum.Stage.PRED) and dimensions == 2
        if stage in (aum.Stage.TRAIN, aum.Stage.VAL) or test_diff_sampler:
            if task == aum.TASK.AE and dimensions == 3:
                assert patch_size is not None
                sampler = tio.data.UniformSampler(patch_size)
            else:
                label_prob = self.calc_label_prob(
                    labels_in=tuple(self._labels_in.keys()),
                    fg_probability=int(self._fg_prob),
                    stage=stage,
                    task=self._task,
                    diffusion=self.is_diffusion,
                    conditional_sampling=self._conditional_sampling,
                    dimensions=self.dimensions
                )

                assert diffusion is not None
                assert patch_size is not None
                if diffusion or test_diff_sampler:
                    sampler = BraTSDiffusionSampler(
                        in_dim=3,
                        out_dim=dimensions,
                        patch_size=patch_size,
                        samples_per_volume=patches_per_subject,
                        stage=stage,
                        prob_map_name=constants.TARGET_SEG_DESCR,
                        label_probabilities=label_prob)
                else:
                    sampler = tio.data.LabelSampler(patch_size, constants.TARGET_SEG_DESCR,
                                                    label_probabilities=label_prob)

        elif stage in [aum.Stage.TEST, aum.Stage.PRED] and dimensions == 3:
            assert patch_overlap is not None
            assert spatial_size is not None
            assert dummy_subject is not None
            assert patch_size is not None
            sampler = GridSamplerSD(
                dimensions=dimensions,
                dummy_subject=dummy_subject,
                patch_size=patch_size,
                patch_overlap=patch_overlap,
                spatial_size=spatial_size)
        else:
            raise AttributeError(f"Wrong stage ({stage.value}) selected. Available stages are:\n"
                                f"[train, val, test]")

        return sampler
    
    @staticmethod
    def calc_label_prob(
            labels_in: Tuple[str],
            fg_probability: int,
            diffusion: bool,
            task: aum.TASK,
            stage: aum.Stage,
            conditional_sampling: bool,
            dimensions: int) -> Dict[int, float]:
        """Calculates the label probabilities depending on the current model and a
        predefined fg_prob. In the case of a diffusion model, the fg probability is the same as the bg
        probability as I want to have the same amount of sample from both in the LabelSampler.
        This only has impacts on the specific sampler. In the diffusion case, only the train and the validation
        rely on the label probabilities, whereas all other stages do Grid sampling in 3D

        Parameters
        ----------
        labels_in : tuple
            Tuple of input labels as strings
        fg_probability : int
            How much more likely eah foreground class will be sampled compared to the background. Is useless in the
            case of diffusion at the moment
        task : CoS
            Whether to utilise classification or segmentation
        diffusion: bool
            If diffusion is utilised or not
        stage: aum.Stage
            Train, val or test
        conditional_sampling : bool
            If conditional sampling is used in the current iteration. This has direct implications
            on the respective probability depending on the model being used.
        is_brats: bool
            Whether it is a BraTS dataset, which is currently the only dataset with multiple labels
            (healthy and diseased)
        Returns
        -------
        dict
            Dictionary with the probability for each int class as required by torchio.LabelSampler
        """
        if diffusion:
            if (
                task in (aum.TASK.CLASS, aum.TASK.AE) or
                (conditional_sampling and task == aum.TASK.SEG) or 
                stage == aum.Stage.VAL or
                (stage in (aum.Stage.TEST, aum.Stage.PRED) and dimensions == 2)
                ):
                return {0: 0.5, 1: 0.5}
            else:
                # NOTE: If conditional sampling is off in the UNet case, we can not use any diseased samples
                #  in the training as I SHOULD only learn the healthy distribution
                return {0: 1., 1: 0.}
        else:
            # Evenly important foreground classes
            num_labels = len(labels_in)
            lbl_ind = [idx for idx in range(num_labels)]
            bg_index = aum.get_background_index(labels_in)
            lbl_ind.remove(bg_index)

            # Normalize to percent
            bg_prob = 1 / (fg_probability * len(lbl_ind) + 1)
            fg_prob = (1 - bg_prob) / len(lbl_ind)

            ret_dic = {idx: fg_prob for idx in lbl_ind}
            ret_dic[bg_index] = bg_prob

            # Sort the dict
            return {lbl: prob for (lbl, prob) in sorted(ret_dic.items())}
        
    @staticmethod
    def _save_json(base_dir: pathlib.Path, json_dict:Dict[str, Any], json_filename: str = 'dataset.json'):
        import json
        with open(str(base_dir / json_filename), "w") as json_file:
            json.dump(json_dict, json_file, indent=4, sort_keys=True)

    @staticmethod
    def _populate_json(
        data_dir: str, 
        additional_info: bool = False, 
        json_filename: str = 'dataset.json',
        verbose: bool = False,
        force: bool = False) -> Dict[str, Any]:
        r"""This method adds entries to the json file. This includes the respective general information about the dataset
        and the individual file paths to each sequence.
        
        Parameters
        ----------
        data_dir : str
            The path to the data directory. Required to find all the files.
        additional_info : bool, optional
            Whether additional information should be added to the json file. This includes the grade and survival information.
        json_filename : str
            The name of the json file. Defaults to 'dataset.json'
        verbose : bool
            Whether the tqdm progress bar should be displayed. Default: False
        force : bool
            Whether all entries within the json file should be redone. Default: False
        """
        import tqdm

        def get_year(dataset: str) -> str:
            import re
            year = re.findall('(?<=BraTS_).*\\d{4}', dataset)
            assert year
            return year[0]

        def get_mapping_dict(year: str, path_to_mapping_csv: Union[str, pathlib.Path]) -> Optional[Dict[str, str]]:
            r"""Function to create the mapping dictionary mapping from 2021 to all iterations of the
            BraTS challenge from 2021 to 2017. It however only creates the mapping currently from 2021 to 2020

            Parameters
            ----------
            path_to_mapping_csv : Union[str, pl.Path]
                Path to the folder containing the file 'BraTS21-17_Mapping.csv'.

            Returns
            -------
            mapping_dict  : Dict[str, str]
                Dictionary containing the 2021 file specifier as name and the 2020 name as value.
            """
            if year == '2021':
                path_to_mapping_csv = pathlib.Path(path_to_mapping_csv)
                filename = 'BraTS21-17_Mapping.csv'
                try:
                    mapping_dict = {}
                    mapping_csv = pd.read_csv(str(path_to_mapping_csv / filename))
                    for idx, name in enumerate(mapping_csv.BraTS2021):
                        mapped_2020_val = mapping_csv.BraTS2020[idx]
                        if not isinstance(mapped_2020_val, str):
                            # This is for all the N/A values
                            mapped_2020_val = None
                        mapping_dict[name] = mapped_2020_val
                    return mapping_dict

                except FileNotFoundError as err:
                    raise FileNotFoundError(f"No file '{filename}' in base dir. Please provide it.\n{err}")
            else:
                return None

        data_dir_pl = pathlib.Path(data_dir)
        json_dict = parse_data.load_json(data_dir=data_dir)
        year = get_year(data_dir)

        if json_dict is None:
            if year == '2023':
                # .name provides the subtask as it is a subfolder
                dataset_name = f'BraTS 2023 {data_dir_pl.name}'
            else:
                raise NotImplementedError(f'Dataset {year} not supported')
            json_dict = BaseBraTS._init_json_dict(year=year,
                                              root_dir=data_dir,
                                              dataset_name=dataset_name,
                                              json_filename=json_filename)

        mapping_dict = get_mapping_dict(year=year, path_to_mapping_csv=data_dir)
        file_suffix = json_dict['file_suffix']
        mri_sequence = json_dict['modality'].values()
        for subset in ('train', 'test'):
            subset_dir = data_dir_pl / subset

            progress_bar = tqdm.tqdm(sorted([dir_ for dir_ in subset_dir.iterdir() if dir_.is_dir()]), disable=not verbose)

            # Process Images, Labels, etc.
            for patient_dir in progress_bar:
                data_dict = {}
                # Folder name is equal to the filename in the folder,
                # e.g. BraTS21_Training_059/BraTS20_Training_059_seq.nii
                patient_id = patient_dir.parts[-1]
                progress_bar.set_description(patient_id)

                if patient_id in json_dict[subset] and not force:
                    continue

                # READ IMAGE
                img_sequence_dict =  BaseBraTS._process_image(
                    patient_dir=patient_dir, 
                    patient_id=patient_id, 
                    file_suffix=file_suffix, 
                    root_dir=data_dir_pl, 
                    mri_seq_specifier=mri_sequence)
                if img_sequence_dict is None:
                    raise RuntimeError(f"No samples found for patient {patient_id}")
                else:
                    data_dict.update(img_sequence_dict)

                # READ LABEL
                if subset != 'test':
                    # Read ground-truth annotation
                    lbl_sequence_dict = BaseBraTS._process_label(patient_dir, patient_id, file_suffix, data_dir_pl)
                    if lbl_sequence_dict is None:
                        raise RuntimeError(f"No labels found for patient {patient_id}")
                    else:
                        data_dict.update(lbl_sequence_dict)

                if additional_info:
                    # GRADE
                    data_dict["grade"] = BaseBraTS._get_grade(patient_id=patient_id,
                                                              year=year, root_dir=data_dir,
                                                              mapping_dict=mapping_dict)

                    # Survival
                    for survival_key in constants.SURVIVAL_KEYS.keys():
                        data_dict[survival_key] = BaseBraTS._get_survival_info(
                            patient_id=patient_id,
                            survival_key=survival_key,
                            root_dir=data_dir, mapping_dict=mapping_dict, year=year)

                
                json_dict[subset][patient_id] = data_dict

                BaseBraTS._save_json(base_dir=data_dir_pl, json_dict=json_dict, json_filename=json_filename)

            json_dict[f"num{subset.capitalize()}"] = len(json_dict[subset])
        
        if ".gz" not in file_suffix:
            # After successful processing, we can overwrite the respective indicator in the json
            json_dict['file_suffix'] = f"{file_suffix}.gz"
        BaseBraTS._save_json(base_dir=data_dir_pl, json_dict=json_dict, json_filename=json_filename)
        return json_dict
    
    @staticmethod
    def _init_json_dict(
        year: str, root_dir: str, dataset_name: str, json_filename: str
        ) -> Dict[str, Any]:
        r"""Helper function for the generation of the json_dict. Tries first to load the json file and generates one from a template if it does not exist.

        Contains the reference,initial file_suffix (i.e. the suffix of the files), the mri_sequence/modality, in and out labels and a potential mapping between different BraTS datasets.

        Returns
        -------
        dictionary with reference, file_suffix, mapping and loading_sequence as keys
        """
        import json

        try:
            base_path = pathlib.Path(root_dir)
            with open(str(base_path / json_filename), "r") as preprocessed_json:
                json_dict = json.load(preprocessed_json)

            # Now replace some of the fields
            try:
                # First try to assign the old labels to the new key "labels_in"
                json_dict['labels_in'] = json_dict.pop("labels")
            except KeyError:
                pass

            # Add new labels as labels out representing the combination of the individual classes
            json_dict["labels_out"] = constants.BRAIN_CLASSES

        except FileNotFoundError:
            json_dict = {
                "name": dataset_name,
                "note": "Labels and segmentations are given as a relative path to this file. Please do not change.",
                "labels_in": {"bg": 0, "ncr": 1, "edema": 2, "et": 3},
                "labels_out": constants.BRAIN_CLASSES,
                "train": {},
                "test": {},
            }

        else:
            raise AttributeError(f"Year {year} not supported.")

        # Add dataset specific stuff
        if year == "2023":
            reference = 'https://www.synapse.org/#!Synapse:syn51156910/wiki/622351'
            file_suffix = ".nii.gz"
            mri_sequence = ('t1c', 't1n', 't2f', 't2w')
        else:
            raise AttributeError(f"Year {year} not supported.")

        json_dict.update({
            'reference': reference,
            'file_suffix': file_suffix,
            'modality': {i: seq for i, seq in enumerate(mri_sequence)}})

        return json_dict    

    @staticmethod
    def _process_image(
        patient_dir: pathlib.Path, 
        patient_id: str,
        file_suffix: str, 
        root_dir: pathlib.Path,
        mri_seq_specifier: Tuple[str, ...]) -> Optional[Dict[str, str]]:

        """Function to process the individual MRI sequences and return them as a dict.
        If the entry can not be found, an error is thrown

        Returns
        -------
        Dictionary with key indicating the sequence and the value showing the relative path to the base directory.
        """

        sequence_paths = {seq: [str(pth.relative_to(root_dir)) for pth in patient_dir.glob(f"*{patient_id}*{seq}{file_suffix}*")][0] for seq in mri_seq_specifier}
        for seq, seq_path in sequence_paths.items():
            if pathlib.Path(seq_path).suffix == ".gz":
                # If the sequence is already compressed, we can just return it
                sequence_paths[seq] = str(seq_path)
            else:
                file_path = pathlib.Path(root_dir / seq_path)
                img_nifty = sitk.ReadImage(str(file_path), imageIO="NiftiImageIO")
                sitk.WriteImage(img_nifty, f'{str(file_path)}.gz', imageIO="NiftiImageIO")
                file_path.unlink()  # Delete the initial uncompressed file
                sequence_paths[seq] = f'{str(seq_path)}.gz'

        # Check if compressed version or not 
        return sequence_paths

    @staticmethod
    def _process_label(
        patient_dir: pathlib.Path, 
        patient_id: str,
        file_suffix: str, 
        root_dir: pathlib.Path) -> Optional[Dict[str, str]]:
        r"""Function that reads the label, alters the label and stores it into a newly compressed .gz file.
        The old file is deleted in this process.

        Notes
        -----

        Altering the label is required, as the labels are given as [1: "ncr", 2: "edema", 4: "et"]. One hot encoding
        will produce one entire tensor with no information (the label 3 one). As a result, I am altering
        label 4 to label 3.

        Returns
        -------

        relative_path : str
            Path of the newly created label relative to the base_dir

        patient_id_in_json : Optional[int]
            Integer if the entry of the processed label can be found in the dataset.json, None otherwise.
            The latter is only applicable if the initial label is found.
            Otherwise, the function searching in the json returns an error.
        """

        lbl_paths = list(patient_dir.glob(f"*{patient_id}*seg{file_suffix}*"))
        if lbl_paths:
            for lbl_path in lbl_paths:
                lbl_nifty = sitk.ReadImage(str(lbl_path), imageIO="NiftiImageIO")
                lbl_np = sitk.GetArrayFromImage(lbl_nifty)
                
                lbl_val = np.unique(lbl_np).tolist()
                skipped_labels = [lbl for lbl in range(4) if lbl not in lbl_val]
                if skipped_labels:
                    wrong_index_labels = [lbl for lbl in lbl_val if lbl not in range(4)]
                    
                    for wrong_index, correct_index in zip(wrong_index_labels, skipped_labels):
                        lbl_np = np.where(lbl_np == wrong_index, correct_index, lbl_np).astype(int)
                    
                lbl_processed = sitk.GetImageFromArray(lbl_np)
                lbl_processed.CopyInformation(lbl_nifty)

                output_path = patient_dir / (patient_id + BaseBraTS.LABEL_SUFFIX)
                if output_path != lbl_path:
                    lbl_path.unlink()
                sitk.WriteImage(lbl_processed, str(output_path), imageIO="NiftiImageIO")
                return {constants.TARGET_SEG_DESCR: str(output_path.relative_to(root_dir))}
        else:
            raise RuntimeError("No lbl paths provided.")

    @staticmethod
    def _get_grade(
        patient_id: str, year: str, root_dir: str,
        mapping_dict: Optional[Dict[str, str]]) -> Optional[str]:
        r"""Function that gets the grade information of the patient. This information
        is only obtained for the 2020 iteration. As a result, the 2021 dataset requires double
        mapping. Once from 2021 to 2020 and then from there to the grade.

        Returns
        -------

        grade : Optional[str]
            grade of the patient if information is available, None otherwise
        """

        def get_grade_dict(path_to_grade_csv: Union[str, pathlib.Path]) -> Dict[str, str]:
            r"""Function to create the grade dictionary. This is included in the 2020 dataset

            Parameters
            ----------
            path_to_grade_csv : Union[str, pl.Path]
                Path to the folder containing the file 'name_mapping.csv'.

            Returns
            -------
            grade_dict  : Dict[str, str]
                Dictionary containing the 2020 file specifier as name and the grade of the glioma.
            """
            path_to_grade_csv = pathlib.Path(path_to_grade_csv)
            filename = 'name_mapping.csv'
            try:
                grade_dict = {}
                grade_csv = pd.read_csv(str(path_to_grade_csv / filename))
                for idx, name in enumerate(grade_csv.BraTS_2020_subject_ID):
                    grade_dict[name] = grade_csv.Grade[idx]
                return grade_dict
            except FileNotFoundError as err:
                raise FileNotFoundError(f"No file '{filename}' in base dir. Please provide it.\n{err}")

        grade_dict = get_grade_dict(root_dir)
        try:
            if year == "2021":
                # requires double mapping
                assert mapping_dict is not None
                return grade_dict[mapping_dict[patient_id]]
            elif year == "2020":
                return grade_dict[patient_id]
            else:
                raise AttributeError(f"Year {year} not supported.")
        except KeyError:
            return None

    @staticmethod
    def _get_survival_info(patient_id: str, survival_key: str,
                          root_dir: str, mapping_dict: Optional[Dict[str, str]], year: str) -> Optional[Union[str, int, float]]:

        def get_survival_dict(path_to_survival_csv: Union[str, pathlib.Path],
                              filename: str = 'survival_info.csv') -> Dict[str, dict]:
            r"""Function to create the survival dictionary. This is included in the 2020 dataset
            and provides the information about the Age, Days of Survival and the Treatment

            Parameters
            ----------
            path_to_survival_csv : Union[str, pl.Path]
                Path to the folder containing the file specified by filename containing
                the survival info.
            filename : str
                Filename containing the survival info. 'survival_info.csv' by default.

            Returns
            -------
            grade_dict  : Dict[str, str]
                Dictionary containing the 2020 file specifier as name and the grade of the glioma.
            """
            path_to_survival_csv = pathlib.Path(path_to_survival_csv)
            survival_dict = {}
            try:
                survival_csv = pd.read_csv(str(path_to_survival_csv / filename))
                for idx, brats20id in enumerate(survival_csv.Brats20ID):
                    # Get the row of the entire entry containing all survival specific attributes
                    row = survival_csv.iloc[[idx]]
                    tmp = {}  # temp dict which aggregates all the information for each patient

                    # Iterate over all necessary columns and get the respective formatting type_ back
                    for key, type_ in constants.SURVIVAL_KEYS.items():
                        csv_val = row[key].values[0]  # extract the value stored in the pandas column
                        try:
                            format_val = type_(csv_val)
                            if format_val == "nan":
                                format_val = None
                        except ValueError:
                            extracted_number = "".join([str(s) for s in csv_val if s.isdigit()])
                            format_val = type_(extracted_number)
                        tmp[key] = format_val
                    survival_dict[brats20id] = tmp
                return survival_dict

            except FileNotFoundError as err:
                raise FileNotFoundError(f"No file '{filename}' in base dir. Please provide it.\n{err}")

        survival_dict = get_survival_dict(root_dir)

        try:
            if year == "2021":
                # requires double mapping
                assert mapping_dict is not None
                return survival_dict[mapping_dict[patient_id]][survival_key]
            elif year == "2020":
                return survival_dict[patient_id][survival_key]
            else:
                raise AttributeError(f"Year {year} not supported.")
        except KeyError:
            return None


    def _data_distribution(self, search_index: str = 'grade'):
        ret_dict = {}
        for subset_specifier in [aum.Stage.TRAIN, aum.Stage.VAL, aum.Stage.TEST]:
            subset = getattr(self, f"_subjects_{subset_specifier.value}")
            if search_index == 'grade':
                _grade = [subj['grade'] for subj in subset.subject_info]
                _dict = {str(i): _grade.count(i) for i in _grade}
                ret_dict[subset_specifier.value] = sorted(_dict.items())
        return ret_dict


class BraTS2023(BaseBraTS):
    r"""Dataset for the 2023 Iteration of the BraTS Challenge obtained through 
    https://www.synapse.org/#!Synapse:syn51156910/wiki/621282

    Notes
    -----

    Returns
    -------

    """
    MAX_LABEL = {'Glioma': 1251, 
                 'Pediatric': 99}
    MAX_NONLABEL = {'Glioma': 219, 
                    'Pediatric': 45}
    BASE_FOLDER = 'BraTS_2023'
    def __init__(self,
                augment: bool,  # noqa: F841
                batch_size: int,  # noqa: F841
                conditional_sampling: bool,  # noqa: F841
                datasets_root: str,  # noqa: F841
                diffusion: bool,  # noqa: F841
                dimensions: Literal[2, 3],  # noqa: F841
                fg_prob: float,  # noqa: F841
                in_channels: int,
                multiclass_pred: bool,  # noqa: F841
                num_workers: int,  # noqa: F841
                patches_per_subj: int,  # noqa: F841
                patch_overlap: int,  # noqa: F841
                patch_size: Tuple[int, int, int],  # noqa: F841
                task: aum.TASK,  # noqa: F841
                num_test: int = -1,
                num_train: int = -1,
                num_val: int = -1,
                num_pred: int = -1,
                subtask: Literal['Glioma', 'Pediatric'] = 'Glioma',
                **kwargs):

        assert subtask in ['Glioma', 'Pediatric'], "brats_2023_subtask must be either 'Glioma' or 'Pediatric'"

        root_dir = pathlib.Path(datasets_root, self.BASE_FOLDER) / subtask
        assert root_dir.is_dir(), f"Directory '{root_dir}' does not exist"
        self._subtask = subtask

        super().__init__(augment=augment,
                        batch_size=batch_size,
                        conditional_sampling=conditional_sampling,
                        root_dir=str(root_dir),
                        dataset='BraTS_2023',
                        diffusion=diffusion,
                        dimensions=dimensions,
                        fg_prob=fg_prob,
                        in_channels=in_channels,
                        multiclass_pred=multiclass_pred,
                        num_workers=num_workers,
                        patches_per_subj=patches_per_subj,
                        patch_overlap=patch_overlap,
                        patch_size=patch_size,
                        task=task,
                        num_test=num_test,
                        num_train=num_train,
                        num_val=num_val,
                        num_pred=num_pred)

    def __str__(self) -> str:
        return "BraTS_2023"
    
    def _check_parsed_samples(self, num_total_label: int, num_total_nonlabel: int) -> None:
        max_label = self.MAX_LABEL[self._subtask]
        max_non_label = self.MAX_NONLABEL[self._subtask]

        assert num_total_label == max_label, f"Number of training subjects ({num_total_label}) does not match the expected number ({max_label})"
        assert num_total_nonlabel == max_non_label, f"Number of prediction/inference subjects ({num_total_nonlabel}) does not match the expected number ({max_non_label})"

    def is_incomplete(self, num_label_loaded: Optional[int], num_nonlabel_loaded: Optional[int]) -> bool:
        if num_label_loaded is None or num_nonlabel_loaded is None:
            return True
        else:
            max_label = self.MAX_LABEL[self._subtask]
            max_non_label = self.MAX_NONLABEL[self._subtask]
            return num_label_loaded != max_label or num_nonlabel_loaded != max_non_label

        