import torch

from typing import Optional

from PIL import Image
import torchvision
import wandb

VERSION_DELIMITER = "|"


def tensor_to_pil(tensor: torch.Tensor, images_per_row: int) -> Image.Image:

    if hasattr(tensor, "requires_grad") and tensor.requires_grad:
        tensor = tensor.detach()  # type: ignore

    data = torchvision.utils.make_grid(tensor, normalize=True, nrow=images_per_row, value_range=(-1, 1))
    image = Image.fromarray(
        data.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy())
    
    return image

def to_wandb(tensor: torch.Tensor, images_per_row: int, caption: Optional[str] = None) -> wandb.Image:
    image = tensor_to_pil(tensor, images_per_row=images_per_row)
    return wandb.Image(image, caption=caption)

def rescale(tensor: torch.Tensor, min_: float, max_: float) -> torch.Tensor:
    return min_ + ((tensor - torch.min(tensor))*(max_ - min_) / (torch.max(tensor) - torch.min(tensor)))