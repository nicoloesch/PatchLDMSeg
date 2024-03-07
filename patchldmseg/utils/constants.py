# Diffusion Specific
DIFFUSION_UNET = 'DiffSeg'


PRINT_PARAMETERS = ["alpha", "activation", "augmentation", "batch_size", "conditional_sampling", 
                    "datasets_root", "dataset",
                    "dimensions", "dropout", "hidden_channels", "learning_rate", "loss_type", "model",
                    "num_max_epochs", "num_test", "num_train", "num_val", "patch_overlap", "patch_size",
                    "patches_per_subj", "upsampling"]

ADAPTIVE_GROUP_NORM = 'ada_gn'

BACKGROUND_STRINGS = ("bg", "background")

# Description how the segmentation is called in the subject and in the dataset json
TARGET_SEG_DESCR = "target_seg"
INPUT_TENSOR_DESCR = "inputs"
LOCATION_TENSOR_DESCR = "location"
SURVIVAL_KEYS = {"Age": float,
                 "Survival_days": int,
                 "Extent_of_Resection": str}

BRAIN_CLASSES = {"bg": ("bg", ),
                 "et": ("et", ),
                 "tc": ("et", "ncr"),
                 "wt": ("et", "ncr", "edema")}

LOGGER_SEPARATOR: str = '/'


# Typing Definitions
from typing import (
    TypeAlias,
    Union,
    List,
    Tuple
)
import torch

SAMPLE_TYPE: TypeAlias = Union[
    torch.Tensor, 
    List[torch.Tensor], 
    Tuple[torch.Tensor, torch.Tensor]]