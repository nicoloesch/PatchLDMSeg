from ema_pytorch import EMA
from torch import nn


class EMAContextManager(EMA):
    r"""Extend the EMA implementation with context manager magic methods so they can be used
    easier and without overhang.
    """
    def __enter__(self) -> nn.Module:
        assert isinstance(self.ema_model, nn.Module)
        return self.ema_model

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Nothing to do here as there is no cleanup necessary atm
        pass
