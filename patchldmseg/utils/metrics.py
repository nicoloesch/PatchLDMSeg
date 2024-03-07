import torch
from typing import Tuple, Optional, Literal

import torchmetrics
from torchmetrics.image.fid import FrechetInceptionDistance
import torchmetrics.classification

from skimage.filters.thresholding import (
    threshold_otsu, 
    threshold_li
)
from skimage.morphology import binary_opening

from patchldmseg.utils.misc import TASK



def get_metrics_collection(task: TASK,
                           class_task: Literal["binary", "multiclass", "multilabel"],
                           num_classes: int,
                           threshold: float = 0.5) -> torchmetrics.MetricCollection:
    
    metrics = {
        TASK.SEG: {
            'metrics': (
                torchmetrics.classification.Accuracy,
                torchmetrics.classification.F1Score,
                torchmetrics.classification.Precision,
                torchmetrics.classification.Specificity,
                torchmetrics.classification.Recall,
                torchmetrics.classification.JaccardIndex),
            'kwargs': {'task': class_task,
                       'threshold': threshold,
                       'num_classes': num_classes}}
    }[task]

    return torchmetrics.MetricCollection([
        metric(**metrics.get('kwargs'))
        for metric in metrics.get('metrics')])


def process_diffusion_tensors(
    input_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    healthy_tensor: torch.Tensor,
    dimensions: int,
    recon_tensor: Optional[torch.Tensor] = None,
):
    # Make the target tensor a single dimension tensor in the channel domain
    target_tensor = preprocess_target(target_tensor)

    if recon_tensor is None:
        recon_tensor = input_tensor.clone()

    # Calculate the anomaly map
    anomaly_map = calc_anomaly_map(recon_tensor, healthy_tensor)

    # Process the anomaly map
    anomaly_map_combined = torch.sum(anomaly_map, 1, keepdim=False)
    anomaly_map_binary = binarise_tensor(
        anomaly_map_combined, 
        dimensions=dimensions)

    return anomaly_map_binary

def calc_anomaly_map(t1: torch.Tensor, t2: torch.Tensor):
    r"""Calculates the anomaly map between two tensors"""
    return torch.abs(t1 - t2)

def preprocess_target(tensor: torch.Tensor, 
                      task: Optional[TASK] = None, 
                      diffusion: Optional[bool] = None) -> torch.LongTensor:
    r"""Preprocessing for target tensor"""

    is_single_channel = tensor.shape[1] == 1

    if not is_single_channel:
        tensor = torch.argmax(tensor, dim=1, keepdim=True)
    return tensor.long()


def binarise_tensor(
        input_tensor: torch.Tensor, 
        dimensions: int) -> torch.Tensor:
    r"""Average Otsu threshold across the batch domain to find the optimal threshold for binarisation"""
    assert input_tensor.dim() == dimensions + 1, f"Expects a channel flattened tensor of size B,Spat with Spat being of size {dimensions}"

    threshold_fn = threshold_otsu

    nan_amount = torch.sum(torch.isnan(input_tensor.detach().cpu()))

    if nan_amount:
        raise RuntimeError(f" {nan_amount} NaN Elements found. Terminating")

    binary = []
    for batch_tensor in input_tensor:
        assert batch_tensor.dim() == dimensions, f"Expects just the spatial tensor. Got {batch_tensor.dim()}"

        binary_tensor = torch.where(
            torch.gt(
                batch_tensor, threshold_fn(batch_tensor.detach().cpu().numpy())
            ), 1, 0)
        
        cleaned_tensor = torch.from_numpy(
            binary_opening(binary_tensor.detach().cpu().numpy())).to(
                device=input_tensor.device,
                dtype=input_tensor.dtype)

        binary.append(cleaned_tensor.unsqueeze(dim=0))

    binary = torch.stack(binary, dim=0)
    return binary.to(input_tensor.device)
