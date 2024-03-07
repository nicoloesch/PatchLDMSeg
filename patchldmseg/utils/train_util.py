from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.cli import LightningCLI, LightningArgumentParser

from patchldmseg.utils.callbacks import ModelCkpt, SWA
from patchldmseg.utils.misc import TASK



class DiffLightningCLI(LightningCLI):
    r"""Wrapper for LightningCLI to implement own methods and linking out of the box"""
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        # General all-purpose arguments
        self._configure_general_params(parser)

        # Add Callbacks
        self._add_callbacks(parser)

        # Link args
        self._link_args(parser)

    @staticmethod
    def _add_callbacks(parser: LightningArgumentParser):
        r"""Add the respective callbacks so they can be configured
        from the command-line or from the .yaml in a better way
        https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_expert.html#configure-forced-callbacks

        """
        activate = {'es': True,
                    'mckpt': True,
                    'swa': False}

        # Early Stopping
        if activate.get('es'):
            parser.add_lightning_class_args(EarlyStopping, 'early_stopping')
            parser.set_defaults({"early_stopping.monitor": 'loss/val_epoch',
                                 "early_stopping.mode": 'min',
                                 "early_stopping.patience": 5,
                                 "early_stopping.min_delta": 0.0})

        # Model Checkpoint
        if activate.get('mckpt'):
            parser.add_lightning_class_args(ModelCkpt, 'model_checkpoint')
            parser.set_defaults({"model_checkpoint.monitor": 'loss/train_epoch',
                                 "model_checkpoint.filename": '{epoch}',
                                 "model_checkpoint.save_top_k": 1,
                                 "model_checkpoint.mode": 'min'})

        # Stochastic Weighted Average
        if activate.get('swa'):
            parser.add_lightning_class_args(SWA, 'swa')
            parser.set_defaults({"swa.swa_lrs": 0.01,
                                 "swa.swa_epoch_start": 100,
                                 "swa.annealing_epochs": 50,
                                 "swa.annealing_strategy": 'cos'})

    @staticmethod
    def _configure_general_params(parser: LightningArgumentParser):
        r"""These are general arguments that are not bound to trainer, model or data
        but exist outside. These are somewhat `global` arguments."""
        
        parser.add_argument(
            '--num_workers', default=4, type=int,
            help='Number of subprocesses to use for data loading.')

        parser.add_argument(
            '--diffusion', type=bool, default=True,
            help='If a diffusion model for the associated `task` is specified by --model. It is required '
                 'for downstream functions to work as the model class can currently not be retrieved '
                 'from the argparser')
        
        parser.add_argument('--pid', type=int, default=0,
                            help="The process ID to identify experiments run on the server.")

        parser.add_argument('--task', type=TASK, default=TASK.SEG,
                            help="The task of the current model.")

        parser.add_argument('--datasets_root', type=str, default='/',
                            help='Absolute path to where all datasets are stored.')

        parser.add_argument('--logging_dir', type=str,
                            default=f"/lightning_logs",
                            help='Absolute path to the logging directory.'
                                 'In order to discern different experiments, the `project_name` will be'
                                 'appended to the path.')

    @staticmethod
    def _link_args(parser: LightningArgumentParser):
        r"""Linking arguments as some args are duplicates amongst the different modules.

        Notes
        -----
        `apply_on='instantiate'` is used if some args are only known AFTER __init__ is called.

        More information can be found here
        https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_expert.html#argument-linking
        """
        # NOTE: instantiate because we only know them after __init__ of the datamodule has run
        parser.link_arguments("data.patch_size", "model.init_args.patch_size", apply_on='instantiate')
        parser.link_arguments("data.batch_size", "model.init_args.batch_size")
        parser.link_arguments("data.dimensions", "model.init_args.dimensions")
        parser.link_arguments("data.conditional_sampling", "model.init_args.conditional_sampling")
        parser.link_arguments("data.dataset_str", "trainer.logger.init_args.dataset_str")
        parser.link_arguments("data.dataset_str", "model_checkpoint.dataset_str")
        parser.link_arguments("data.in_channels", "model.init_args.in_channels")
        parser.link_arguments("data.dataset", "model.init_args.dataset", apply_on="instantiate")

        # General Args
        parser.link_arguments('datasets_root', 'data.datasets_root')
        parser.link_arguments("diffusion", "data.diffusion")
        parser.link_arguments("diffusion", "model.init_args.diffusion")
        parser.link_arguments("diffusion", "trainer.logger.init_args.diffusion")
        parser.link_arguments("diffusion", "model_checkpoint.diffusion")
        parser.link_arguments("logging_dir", "trainer.logger.init_args.logging_dir")
        parser.link_arguments("logging_dir", "trainer.callbacks.init_args.logging_dir")
        parser.link_arguments("logging_dir", "model.init_args.logging_dir")
        parser.link_arguments("logging_dir", "model_checkpoint.logging_dir")
        parser.link_arguments("num_workers", "data.num_workers")
        parser.link_arguments("pid", "trainer.logger.init_args.pid")
        parser.link_arguments("pid", "trainer.callbacks.init_args.pid")
        parser.link_arguments("pid", "model.init_args.pid")
        parser.link_arguments("pid", "model_checkpoint.pid")
        parser.link_arguments("task", "data.task")
        parser.link_arguments("task", "trainer.logger.init_args.task")
        parser.link_arguments("task", "trainer.callbacks.init_args.task")
        parser.link_arguments("task", "model_checkpoint.task")
        parser.link_arguments("task", "model.init_args.task")
