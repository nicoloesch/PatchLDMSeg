from patchldmseg.utils import train_util
import patchldmseg.input.data_loader
from patchldmseg.model.lightning.base_model import BaseModel
from patchldmseg.input.data_loader import DataModule


def cli_main():
    # Start Training/Testing based on the args
    run = True

    cli = train_util.DiffLightningCLI(
            model_class=BaseModel,
            datamodule_class=DataModule,
            subclass_mode_model=True,
            run=run,
            seed_everything_default=123,
            save_config_kwargs={"overwrite": True,
                                "multifile": True},
            args=None)


if __name__ == "__main__":
    import warnings
    from lightning_fabric.utilities.warnings import PossibleUserWarning

    with warnings.catch_warnings():
        warnings.filterwarnings(
            action='ignore',
            category=PossibleUserWarning,
            message='The dataloader,')
        cli_main()
