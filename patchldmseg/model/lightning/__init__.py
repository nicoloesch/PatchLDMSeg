# These are required for lightning_cli and the subclass mode - Don't know why but otherwise the code breaks
from .base_model import BaseModel
from .diffbase import DiffBase
from .diffseg import DiffSeg
from .vqgan import VQGAN