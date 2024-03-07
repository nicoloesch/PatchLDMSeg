import pytorch_lightning
from typing import Optional
import pytorch_lightning.loggers
from pytorch_lightning.callbacks.stochastic_weight_avg import StochasticWeightAveraging
from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from patchldmseg.utils.misc import create_experiment_name, TASK, create_logging_dir

import sys
from torch import nn


class Logger(pytorch_lightning.loggers.WandbLogger):
    r"""Custom Logger required to make the necessary
    under the hood adjustments for pytorch_lightning v.2.x

    Parameters
    ----------

    """

    def __init__(self,
                 logging_dir: str,
                 pid: int,
                 task: TASK,
                 diffusion: bool,
                 dataset_str: str,
                 project_name: Optional[str] = None,
                 experiment_name: Optional[str] = None
                 ):
        if experiment_name is None:
            experiment_name = create_experiment_name(pid, task, diffusion)
        if project_name is None:
            project_name = f"Diffusion_{dataset_str}" if diffusion else f"NoDiffusion_{dataset_str}"
        logging_dir = create_logging_dir(logging_dir, project_name, experiment_name)
        super().__init__(name=experiment_name,
                         save_dir=logging_dir,
                         project=project_name,
                         log_model=False)
        print(self.experiment.id)

class ModelCkpt(ModelCheckpoint):
    def __init__(self,
                 logging_dir: str,
                 pid: int,
                 task: TASK,
                 diffusion: bool,
                 dataset_str: str,
                 project_name: Optional[str] = None,
                 dirpath: Optional[str] = None,
                 filename: str = '{epoch}',
                 monitor: Optional[str] = None,
                 save_top_k: int = 1,
                 save_last: Optional[bool] = True,
                 mode: str = 'min',
                 experiment_name: Optional[str] = None,
                 ):
        if experiment_name is None:
            experiment_name = create_experiment_name(pid, task, diffusion)
        if project_name is None:
            project_name = f"Diffusion_{dataset_str}" if diffusion else f"NoDiffusion_{dataset_str}"
        logging_dir = dirpath or create_logging_dir(logging_dir, project_name, experiment_name)

        super(ModelCkpt, self).__init__(dirpath=logging_dir,
                                        filename=filename,
                                        monitor=monitor,
                                        save_top_k=save_top_k,
                                        mode=mode,
                                        save_last=save_last)


class SWA(StochasticWeightAveraging):
    r"""SWA Model that can be used within a with statement"""

    def update(self):
        pass 

    def __enter__(self) -> nn.Module:
        assert isinstance(self._average_model, nn.Module)
        return self._average_model.eval()

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Nothing to do here as there is no cleanup necessary atm
        pass
