from typing import Tuple, Union, Optional
import pathlib
from enum import Enum

import torch
import torch.nn as nn
from pytorch_lightning.core import saving
import SimpleITK as SITK
import numpy as np

from patchldmseg.utils import constants


class Stage(Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'
    PRED = 'pred'


class TASK(Enum):
    r"""Enum to switch between different modes.
    Utilisation: --task=SEG"""
    CLASS = 'classification'
    SEG = 'segmentation'
    GEN = 'generation'
    AE = 'autoencoder'


class SoE(Enum):
    r"""Step or Epoch Enum"""
    STEP = 'step'
    EPOCH = 'epoch'


def get_background_index(label_names: Tuple[str]) -> int:
    bg_index = []
    for bg_string in constants.BACKGROUND_STRINGS:
        try:
            bg_index.append(label_names.index(bg_string))
        except ValueError:
            continue

    # Means there is only one entry matching to one of the background strings
    if len(bg_index) == 1:
        return bg_index[0]
    elif len(bg_index) == 0:
        raise AttributeError(f"No background label present in label_names. Make sure you include "
                             f"{constants.BACKGROUND_STRINGS}")
    else:
        raise AttributeError(f"More than one background label in label_names."
                             f"Make sure you onl include one of {constants.BACKGROUND_STRINGS}")

def expand_dims(tensor: Union[torch.Tensor, np.ndarray],
                out_dim: int,
                device: Optional[torch.device] = None,
                dtype: torch.dtype = torch.float32) -> torch.Tensor:
    r"""This function expands a singleton vector to have n dimensions specified
    by dimensions. E.g. a vector of [128,] and out_dim 4 will look like [128,1,1,1].
    This allows the multiplication with multidimensional vectors along the first (batch) dimension.

    Parameters
    ----------
    tensor : torch.Tensor
        The input tensor to be modified
    out_dim: int
        The number of output dimensions of the vector
    device: torch.device, optional
        If specified, the tensor is also pushed to the correct device.
    """
    if isinstance(tensor, torch.Tensor):
        init_dim = tensor.dim()
        assert init_dim <= out_dim
        if device is None:
            device = tensor.device
        return tensor[(...,) + (None,) * (out_dim - init_dim)].to(device, dtype)
    elif isinstance(tensor, np.ndarray):
        init_dim = len(tensor.shape)
        assert init_dim <= out_dim
        return tensor[(...,) + (None,) * (out_dim - init_dim)]
    else:
        raise RuntimeError("Unsupported type")


def non_batch_mean(tensor: torch.Tensor) -> torch.Tensor:
    r"""Function that calculates the mean over all dimensions
    EXCEPT the batch dimension. The batch dimension is assumed
    to be at 0 position."""
    return torch.mean(tensor, dim=tuple(range(1, tensor.dim())))

def calc_distance_map(lbl_map: torch.Tensor) -> torch.Tensor:
    r"""This function finds the distance map from the surface of a label map. The function calculates the
    closest distance from the surface of elements that are 1 to everything that is zero.

    Parameters
    ----------
        lbl_map: torch.Tensor
            Label map with the pixels from which the distance should be calculated from are 1 and their
            counterpart 0.
    """

    assert lbl_map.dim() <= 3, "Only maximal 3 dimensional tensors are supported"

    lbl_np = lbl_map.numpy().astype(int)

    sitk_bg = SITK.Cast(SITK.GetImageFromArray(lbl_np), SITK.sitkUInt32)
    distance_map = SITK.Abs(SITK.SignedMaurerDistanceMap(sitk_bg, insideIsPositive=False,
                                                         squaredDistance=False, useImageSpacing=True))

    reshaped_distance = SITK.GetArrayViewFromImage(distance_map)
    writable = np.copy(reshaped_distance)
    return torch.from_numpy(writable)


def zero_init(module: nn.Module) -> nn.Module:
    r"""Initialises all parameters within the module as 0"""
    for p in module.parameters():
        p.detach().zero_()
    return module


def create_experiment_name(pid: int, task: TASK, diffusion: bool) -> str:
    from datetime import date
    today = date.today().strftime("%Y%m%d")
    if diffusion:
        if task == TASK.SEG:
            model = "DiffSeg"
        elif task == TASK.CLASS:
            model = "DiffClass"
        elif task == TASK.GEN:
            model = "DiffGen"
        elif task == TASK.AE:
            model = "VQGAN"
        else:
            raise AttributeError
    else:
        model = str(task.value).capitalize()

    return "_".join([today, str(pid), model])


def create_logging_dir(
        logging_dir: str,
        project_name: str,
        experiment_name: str) -> str:
    r"""Collates AND creates (if necessary) the logging directory
    specified based on the input arguments

    Parameters
    ----------
    logging_dir: str
        Absolute path to the logging directory.
        In order to discern different experiments, the `project_name` will be appended to the path.
    project_name: str
        Suffix (folder in the base logging_dir) to have a separation between the different experiment.
    experiment_name: str
        Second suffix that is appended to the directory path and presents the final directory.
    """
    ld_pl = pathlib.Path(logging_dir)
    full_path = ld_pl / project_name / experiment_name

    # Check the logging_dir path first if it in fact does exist
    if ld_pl.is_dir():
        if not full_path.is_dir():
            # Means the last one is missing (i.e. the experiment name) and we need to create directories
            if full_path.parent.is_dir():
                # Means project_name exists. Create only the last one
                parents = False
            else:
                # Only logging_dir exists and we create both subfolders (project_name and experiment_name)
                parents = True
            full_path.mkdir(parents=parents)
    else:
        if ld_pl.parent.is_dir():
            # I only accept that one dir is missing
            ld_pl.mkdir(parents=False)
            full_path = create_logging_dir(logging_dir, project_name, experiment_name)
        else:
            raise RuntimeError(f"Specified directory `logging_dir` {str(ld_pl)} does not exist and "
                               f"its parent directory does not exist as well.")

    return str(full_path)


def compare_tensors(t1: torch.Tensor, t2: torch.Tensor, threshold: float = 1e-7):
    assert torch.allclose(t1, t2, atol=threshold)

