import abc
from typing import Any, Dict, Tuple, List, Optional

import torch
import torch.distributed as dist
import torchio
import pytorch_lightning as pl
import torchmetrics
from torch.utils.data import DataLoader, DistributedSampler
from pytorch_lightning.utilities import rank_zero_only
import numpy as np

from patchldmseg.utils.misc import Stage, SoE
from patchldmseg.utils import constants
from patchldmseg.input.sampler import GridAggregatorSD
from patchldmseg.input.datasets.base_dataset import Dataset
from patchldmseg.input.datasets import BaseBraTS
from patchldmseg.utils.visualization import to_wandb


class BaseModel(pl.LightningModule, abc.ABC):
    """Abstract baseclass for all lightning modules by providing a general interface.

    Parameters
    ----------
    batch_size: int
        The batch size of each step
    """
    def __init__(self,
                 batch_size: int,
                 num_samples_to_log: int = 0,
                 **kwargs: Any,
                 ) -> None:

        super().__init__()

        # Get the attributes
        self._batch_size = batch_size
        
        # Sampling container
        self._samples = []
        self._num_samples_to_log = num_samples_to_log
        self._has_logged_samples = False

    def parse_input_args(
            self, 
            dropout: int,
            *args, **kwargs) -> None:
        r"""Parses the input args and checks them for correctness"""
        assert dropout in range(0,101)


    @abc.abstractmethod
    def forward(self, **kwargs) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def validation_step(self, batch, batch_idx, dataloader_idx) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def test_step(self, batch, batch_idx, dataloader_idx) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def configure_optimizers(self) -> Any:
        return
    
    @property
    def is_ddp(self) -> bool:
        return dist.is_available() and dist.is_initialized()
    
    def on_train_start(self) -> None:
        torch.cuda.empty_cache()
        return super().on_train_start()

    def on_train_epoch_start(self) -> None:
        train_dataloader = self.trainer.train_dataloader
        val_dataloaders = self.trainer.val_dataloaders
        if not isinstance(val_dataloaders, list):
            val_dataloaders = [val_dataloaders]
        epoch = self.trainer.current_epoch
        self._set_epoch_ddp([train_dataloader], epoch)
        self._set_epoch_ddp(val_dataloaders, epoch)

    @staticmethod
    def _set_epoch_ddp(dl_list: List[DataLoader], epoch: int):
        for dl in dl_list:
            dataset = getattr(dl, 'dataset', None)
            if isinstance(dataset, torchio.Queue):
                ddp_sampler = dataset.subject_sampler
                if isinstance(ddp_sampler, DistributedSampler):
                    ddp_sampler.set_epoch(epoch)

    def log_loss(self, loss: torch.Tensor, stage: Stage, soe: SoE):
        loss_name = self.format_metric_log_name("loss", stage=stage, soe=soe, append_soe=False)
        self.log(loss_name, loss, on_step=True, on_epoch=True, batch_size=self._batch_size,
                 sync_dist=True)

    def log_metrics(self,
                    prediction: torch.Tensor,
                    target: torch.Tensor,
                    stage: Stage,
                    batch_size: int,
                    on_step: Optional[bool] = True,
                    on_epoch: Optional[bool] = True,
):
        dl_metric_fn = self._get_and_check_metric_fn(stage=stage)
        metrics_out = dl_metric_fn(preds=prediction, target=target)

        with torch.no_grad():
            # NOTE: Currently unknown how this behaves for multiclass logging but currently irrelevant
            self.log_dict(metrics_out, on_epoch=on_epoch, on_step=on_step, batch_size=batch_size)

    @staticmethod
    def get_input_tensor_from_batch(batch: Dict[str, Any],
                                    sequences: Tuple[str, ...]) -> torch.Tensor:
        r"""Returns the input tensor from the batch by concatenating the individual
        images along the Channel dimension (dim=1). Only applicable
        to medical images (torchio.Subjects/Images) as they are the only ones having
        multiple images per subject.

        Parameters
        ----------
        batch : Dict[str, Any]
            The batch from the dataloader

        Returns
        -------
        torch.Tensor
            The input tensor from the batch
        """ 
        tensor_list = []
        for seq in sequences:
            tensor = BaseModel._get_tensor_from_batch(batch, seq)
            if tensor is not None:
                tensor_list.append(tensor)

        return torch.cat(tensor_list, dim=1)
    
    @staticmethod
    def get_target_class_tensor_from_batch(
        batch: Dict[str, Any], 
        dataset: Dataset,
        stage: Stage) -> Optional[torch.Tensor]:
        r"""Returns the target class tensor from the batch

        Parameters
        ----------
        batch : Dict[str, Any]
            The batch from the dataloader

        Returns
        -------
        torch.Tensor, optional
            The target tensor from the batch (if it exists)
        """ 
        tensor = BaseModel.seg_to_class(
            BaseModel._get_tensor_from_batch(batch, constants.TARGET_SEG_DESCR))

        if tensor is None:
            _error_msg = (
                f"Dataset {dataset} should contain labels in the current stage {stage.value}.\n"
                f"Check dataloader and sampler."
                )
            raise RuntimeError(_error_msg)

        return tensor
             
        
    @staticmethod
    def get_location_tensor_from_batch(batch: Dict[str, Any]) -> torch.Tensor:
        r"""Returns the location tensor from the batch. Only available for samplers that provide the location information (mostly torchio patch samplers.)

        Parameters
        ----------
        batch : Dict[str, Any]
            The batch from the dataloader

        Returns
        -------
        torch.Tensor
            The location tensor from the batch
        """ 

        tensor = BaseModel._get_tensor_from_batch(batch, constants.LOCATION_TENSOR_DESCR)
        if tensor is not None:
            return tensor
        else:
            raise RuntimeError(f"Tensor not available in batch. Available keys are {[key for key in batch.keys()]}")

    @staticmethod
    def get_target_seg_tensor_from_batch(batch: Dict[str, Any]) -> torch.Tensor:
        r"""Returns the target segmentation tensor from the batch

        Notes
        -----
        This should only be used in models relying on segmentation and will therefore be overwritten in 
        classifier and generator models.

        Parameters
        ----------
        batch : Dict[str, Any]
            The batch from the dataloader

        Returns
        -------
        torch.Tensor
            The target tensor from the batch
        """ 

        tensor = BaseModel._get_tensor_from_batch(batch, constants.TARGET_SEG_DESCR)
        if tensor is not None:
            return tensor
        else:
            raise RuntimeError(f"Tensor not available in batch. Available keys are {[key for key in batch.keys()]}")

    @staticmethod
    def _get_tensor_from_batch(batch: Dict[str, Any], key: str) -> Optional[torch.Tensor]:
        r"""Returns the tensor from the batch

        Parameters
        ----------
        batch : Dict[str, Any]
            The batch from the dataloader
        key : str
            The key of the tensor in the batch

        Returns
        -------
        torch.Tensor
            The tensor from the batch
        """

        tensor = batch.get(key, None)
        if tensor is not None:
            if isinstance(tensor, dict) and 'data' in tensor:
                # This is to support torchio.Image
                tensor= tensor['data']
            tensor = BaseModel._squeeze_spatial(tensor)
        return tensor

    @staticmethod
    def _squeeze_spatial(tensor: torch.Tensor) -> torch.Tensor:
        r"""Squeeze all singleton spatial dimensions.
        This is required to make the 2D version compatible with the 3D framework.

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor of size BxCxSpat
        """
        if tensor is None:
            raise RuntimeError('Not possible for NoneType tensor')
        spatial_dim = torch.as_tensor(tensor.shape[2:])
        singleton_dims = torch.eq(spatial_dim, 1)

        if torch.any(singleton_dims):
            index = singleton_dims.tolist().index(True) + 2  # To offset B and C
            tensor = BaseModel._squeeze_spatial(tensor.squeeze(index))
        return tensor

    @staticmethod
    def seg_to_class(seg_label: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        r"""Function that processes a segmentation label and returns the associated
        classification label (if the segmentation is present). 
        If a foreground pixel is present in the segmentation, that specific batch will 
        have the respective 1 classification label."""

        if isinstance(seg_label, torch.Tensor):
            # Now check if there is a label in the target
            b, c, *spat = seg_label.shape

            # Flatten the spatial dimension
            target = seg_label.reshape(b, c, -1).contiguous()

            # Check if any pixel has a label other than the background in the spatial dimension
            # This is the new target: 1 for diseased, 0 for healthy
            target = torch.any(target != 0, dim=-1, keepdim=True).long()
            target = torch.squeeze(target, -1)  # Get rid of the spatial dimensions

            assert isinstance(target, torch.Tensor)

            return target
        else:
            return None

    @staticmethod
    def format_metric_log_name(metric: str,
                               stage: Stage,
                               soe: SoE,
                               dl: Optional[str] = None,
                               append_soe: bool = False) -> str:
        r"""Determines the default formatting for all logged metrics in a concise manner.
        Current format is depending on the ptl_auto_log string

        metric/stage_soe_dl

        with _dl only being present if there is a dataloader

        Parameters
        ----------
        metric : str
            The metric to log e.g. 'loss', 'f1', ...
        stage : Stage
            The stage the current model is in
        soe : SoE
            Whether step or epoch is used
        dl : str, optional
            The dataloader name in the case of multiple
        append_soe : bool
            If the step (_step) or epoch (_epoch) should be attached to the metric name.
            This is not required if the auto-logging of pytorch lightning is utilised
            self.log(on_epoch=True, on_step=True) as the respective SoE will be appended anyways
        """
        assert stage in [member for member in Stage], \
            f"Wrong state utilised `{stage.value}`.\nAvailable options are {Stage}"

        assert soe in [member for member in SoE], \
            f"Wrong state utilised `{soe.value}`.\nAvailable options are {SoE}"

        rhs = [stage.value, dl]
        if append_soe:
            rhs.append(soe.value)
        rhs_filtered = list(filter(lambda item: item is not None, rhs))
        out_list = [metric, "_".join(rhs_filtered)]

        return constants.LOGGER_SEPARATOR.join(out_list)

    def _get_and_check_metric_fn(self, stage: Stage) -> torchmetrics.MetricCollection:
        metric_fn_name = f'{stage.value}_metrics'
        metric_fn: torchmetrics.MetricCollection = getattr(self, metric_fn_name)

        try:
            assert stage.value == metric_fn.postfix.strip("/")
        except AttributeError:
            assert stage.value == metric_fn.postfix

        return metric_fn

    @staticmethod
    def _add_to_aggr(list_aggregators: List[Optional[GridAggregatorSD]],
                     list_tensors: List[torch.Tensor],
                     tensor_names: List[str],
                     location: torch.Tensor):
        dct = {}

        for aggr, tensor, name in zip(list_aggregators, list_tensors, tensor_names):
            assert isinstance(aggr, GridAggregatorSD)
            dct[name] = aggr.add_batch(tensor, location)

        # Check that all elements are having the same type, either torch.Tensor or None
        assert len(set(map(type, dct.values()))) == 1
        return dct

    def get_hparam(self, param: str) -> Any:
        try:
            return getattr(self.hparams, param)
        except AttributeError:
            raise RuntimeError(f'No hparam {param} found in model. Make sure you called `save_hyperparameters in __init__` of {self.__class__.__name__}')
        
    def get_trainer_param(self, param: str):
        assert self.trainer is not None
        try:
            return getattr(self.trainer, param)
        except AttributeError:
            raise RuntimeError(f'No param {param} found in trainer')
        
    @rank_zero_only
    def log_images(self, tensors: Dict[str, torch.Tensor]):
        r"""Logs an image in combination with the target ground-truth.
        The slice is determined to be coming from the target with the most values.

        Parameters
        ----------
        tensors: dict of torch.Tensor
            Images as torch.Tensor to be logged. The key is the description in the plot
        """
        if self.logger is None:
            # No logging object results in no logging
            return
        
        if self.get_hparam('dimensions') == 3:
            gt_tensor = tensors.get('gt_seg', None)
            if gt_tensor is not None:
                if gt_tensor.dim() == self.get_hparam('dimensions') + 2:
                    gt_tensor = gt_tensor.squeeze(1)  # Remove channel_dim

                assert gt_tensor.dim() == self.get_hparam('dimensions') + 1, "Ground-truth tensor cannot have a channel dimension"
                # As I can only log 2D images, I am logging the slice of the last dimension with the most amount of fg
                idx = torch.max(
                    torch.sum(
                        # Offset for batch_size and do all spatial dimensions except the last
                        gt_tensor, dim=tuple(1 + ps for ps in range(self.get_hparam('dimensions') - 1))),
                    dim=-1).indices
            else:
                # Get the middle slice base on the last dimension
                idx = tensors[list(tensors.keys())[0]].shape[-1] // 2
        else:
            # 2D case - I do not need an index
            idx = None

        log_dict = {"Samples": []}

        for descr, tensor in tensors.items():
            if isinstance(self.dataset, BaseBraTS):
                # Reshape the tensor
                b, c, *spat = tensor.shape
                tensor = tensor.reshape(b * c, 1, *spat).contiguous()
                images_per_row = 4
            else:
                b = tensor.shape[0]
                images_per_row = int(np.ceil(np.sqrt(b)))
            
            image = tensor[..., idx].squeeze(-1)
            log_dict['Samples'].append(to_wandb(image.float(), caption=descr, images_per_row=images_per_row))

        self.logger.experiment.log(log_dict)

    def accumulate_samples_and_log(self, samples: Dict[str, torch.Tensor]):
        if not self._has_logged_samples:
            if self.is_ddp:
                dist.barrier()

            # In the case of stitching multiple 3D patches together, the batch size does no
            # longer work
            samples_batch_size = set([sample.shape[0] for sample in samples.values()])
            assert len(samples_batch_size) == 1, "Batch size mismatch of all samples"
            samples_batch_size = list(samples_batch_size)[0]

            # Append samples to the temporary container list
            if samples and len(self._samples) * samples_batch_size < self._num_samples_to_log:
                self._samples.append(samples)

            # At least sample at the end (or when I acquired enough samples)
            if (samples and len(self._samples) * samples_batch_size >= self._num_samples_to_log) or self.trainer.is_last_batch:
                merged_samples = self._merge_samples()

                if self.trainer.global_rank == 0:
                    # Only log tensors 
                    output = {key: tensor[:min(tensor.shape[0], self._num_samples_to_log), ...] for key, tensor in merged_samples.items()}
                    self.log_images(output)
                
                # Set the flag to True to prevent logging multiple times
                self._has_logged_samples = True
                self._samples.clear()  # No need to hold them longer in memory

    def _merge_samples(self) -> dict:
        r"""This function merges samples together by combining the sampled tensors from multiple GPUs
        stored in the self._samples list. The samples are concatenated along the batch dimension and then
        gathered across all processes. The gathered samples are then concatenated along the batch dimension
        and returned for logging"""

        if not self.samples:
            raise RuntimeError("self.samples is empty. Make sure to append the samples to the list.")
    
        aggregate_dict = {key: [] for key in self._samples[0].keys()}

        for batch_sample in self._samples:
            for key, value in batch_sample.items():
                aggregate_dict[key].append(value)

        # Concatenate along the batch dimension
        merged_samples = {key: torch.cat(value, dim=0) for key, value in aggregate_dict.items()}


        # Now gather them across processes
        if self.is_ddp:
            output = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(output, merged_samples)

            # Now I need to combine them together
            if self.trainer.global_rank == 0:
                # to combine them, we want them on the CPU (should be any)
                output = {key: torch.cat([val.cpu() for val in values]) for key, values in zip(output[0].keys(), zip(*[dct.values() for dct in output]))}
        else:
            output = merged_samples

        return output

    def on_train_epoch_end(self) -> None:
        self._has_logged_samples = False
        self._samples.clear()

    def on_validation_epoch_end(self) -> None:        
        self._has_logged_samples = False
        self._samples.clear()

    def on_test_end(self) -> None:
        self._has_logged_samples = False
        self._samples.clear()

    def _add_batch_to_aggr(self,
                           initial: torch.Tensor,
                           healthy: torch.Tensor,
                           target: torch.Tensor,
                           location: torch.Tensor,
                           recon: Optional[torch.Tensor] = None) -> Dict[str, Optional[torch.Tensor]]:
        r"""Add batch to the respective aggregators for training. Based on the Aggr.State,
        the entire filled aggregator will be returned after adding the batch

        Parameters
        ----------
        initial : torch.Tensor
            The input tensor, typically the raw input
        healthy : torch.Tensor
            The 'healthified' version of the input tensor after feeding it through the Diffusion Net
        target : torch.Tensor
            The target ground-truth segmentation map
        location : torch.Tensor
            The location information of the current batch obtained automatically if the GridSampler is utilised
            to generate the patches
        recon : torch.Tensor, optional
            The reconstruction of the input tensor. If not specified, it will not be in the output dict

        Returns
        -------
        dict
            A dictionary with str indicating the data (healthy, initial, target) and the associated optional tensor
        """
        from patchldmseg.input.datasets.brats import Aggregators

        aggregators: Aggregators = getattr(getattr(getattr(getattr(getattr(self, "trainer"), "datamodule"), "dataset"), "subjects_test"), "aggregators")

        list_aggregators = [aggregators.pred, aggregators.original, aggregators.target]
        list_tensors = [healthy, initial, target]
        tensor_names = ['healthy', 'init', 'target']

        if recon is not None:
            list_aggregators.append(aggregators.recon)
            list_tensors.append(recon)
            tensor_names.append('recon')

        return self._add_to_aggr(list_aggregators, list_tensors, tensor_names, location)