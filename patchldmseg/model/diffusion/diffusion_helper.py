# Implements all model building blocks for diffusion models for easy access

import torch
import torch.distributed as td
import enum
import abc
from typing import Tuple, List, Optional
import pathlib


class AbstractDiffusionSampler(abc.ABC):
    r"""Abstract base class to sample timsteps of the diffusion process in a predefined manner
    to reduce variance of the objective.
    """

    def __init__(self, diffusion_steps: int):
        assert diffusion_steps > 0, "Diffusion steps must be greater than 0"
        self._diffusion_steps = diffusion_steps

    @property
    def diffusion_steps(self) -> int:
        return self._diffusion_steps

    @abc.abstractmethod
    def weights(self) -> torch.Tensor:
        r"""Return Tensor with associated weights for each timestep.
        Weights don't need to be normalised but must be positive.
        These are used to rescale the loss accordingly."""

    @abc.abstractmethod
    def update_with_local_loss(self,
                               is_single_device: bool,
                               local_ts: torch.Tensor,
                               local_losses: torch.Tensor):
        """
        Update the reweighting using losses from a model.

        Call this method from each rank with a batch of timesteps and the
        corresponding losses for each of those timesteps.
        This method will perform synchronization to make sure all of the ranks
        maintain the exact same reweighting.

        Parameters
        ----------
        is_single_device: bool
            Whether DDP is used or not. Only DDP requires syncing
        local_ts: torch.Tensor
            an integer Tensor of timesteps.
        local_losses: torch.Tensor
            a 1D Tensor of losses.
        """

    def sample(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Importance-sample timesteps for a batch

        Parameters
        ----------
        batch_size : int
            Number of batches
        device : torch.device
            The torch device where the tensor should be on

        Returns
        -------
        timesteps : torch.Tensor
            The randomly sampled timesteps
        weights : torch.Tensor
            The associated weights for the loss reweighing
        """
        import numpy as np

    
        w = self.weights()
        p = w / torch.sum(w)  # probability

        # Has to happen in numpy as it allows the provision of a probability
        ts_np = np.random.choice(self.diffusion_steps, size=(batch_size,), p=p.numpy())
        timesteps = torch.from_numpy(ts_np).long()
        weights = torch.div(1, self.diffusion_steps * p[timesteps]).float().to(device)
        # Indexing only works if both are on the same device. Therefore, I push timesteps to the device here
        return timesteps.to(device), weights.to(device)


class UniformSampler(AbstractDiffusionSampler):
    def weights(self) -> torch.Tensor:
        return torch.ones(self.diffusion_steps)

    def update_with_local_loss(self, is_single_device: bool, local_ts: torch.Tensor, local_losses: torch.Tensor):
        r"""Does not require syncing between devices as it is uniform sampling"""
        return


class ImportanceSampler(AbstractDiffusionSampler):
    def __init__(self,
                 diffusion_steps: int,
                 history=10,
                 uniform_prob=0.001):
        super().__init__(diffusion_steps)

        # How many sampled steps are kept as history
        self._history = history
        self._uniform_prob = uniform_prob
        self._loss_history = torch.zeros((diffusion_steps, history), dtype=torch.float)
        self._loss_counts = torch.zeros(diffusion_steps, dtype=torch.int)

    def weights(self) -> torch.Tensor:
        if not self._warmed_up():
            return torch.ones(self.diffusion_steps, dtype=torch.int)
        # Mean over the history of losses for each diffusion step
        weights = torch.sqrt(torch.mean(self._loss_history ** 2, dim=-1))
        weights /= torch.sum(weights)  # Normalize to a max of 1
        # weights *= 1. - self._uniform_prob
        # weights += self._uniform_prob / len(weights)
        return weights

    def update_with_local_loss(self,
                               is_single_device: bool,
                               local_ts: torch.Tensor,
                               local_losses: torch.Tensor):
        if not is_single_device:
            batch_sizes = [
                torch.tensor([0], dtype=torch.int32, device=local_ts.device)
                for _ in range(td.get_world_size())
            ]
            td.all_gather(
                batch_sizes,
                torch.tensor([len(local_ts)], dtype=torch.int32, device=local_ts.device),
            )

            # Pad all_gather batches to be the maximum batch size.
            batch_sizes = [x.item() for x in batch_sizes]
            max_bs = max(batch_sizes)

            timestep_batches = [torch.zeros(max_bs).to(local_ts) for bs in batch_sizes]
            loss_batches = [torch.zeros(max_bs).to(local_losses) for bs in batch_sizes]
            td.all_gather(timestep_batches, local_ts)
            td.all_gather(loss_batches, local_losses)
            timesteps = [
                x.item() for y, bs in zip(timestep_batches, batch_sizes) for x in y[:bs]
            ]
            losses = [x.item() for y, bs in zip(loss_batches, batch_sizes) for x in y[:bs]]
        else:
            timesteps = [local_ts]
            losses = [local_losses]
        self.update_with_loss(timesteps, losses)

    def update_with_loss(self, ts: List[torch.Tensor], losses: List[torch.Tensor]):
        r"""Update the reweighting using losses from a model.

        This method directly updates the reweighting without synchronizing
        between workers. It is called by update_with_local_losses from all
        ranks with identical arguments. Thus, it should have deterministic
        behavior to maintain state across workers.

        Parameters
        ----------

        ts: list of torch.Tensor
            a list of int timesteps.
        losses: list of torch.Tensor
            a list of float losses, one per timestep.
        """
        for t, loss in zip(ts, losses):
            for b_t, b_loss in zip(t.cpu(), loss.cpu()):
                if self._loss_counts[b_t] == self._history:
                    # Shift out the oldest loss term.
                    self._loss_history[b_t, :-1] = self._loss_history[b_t, 1:]
                    self._loss_history[b_t, -1] = b_loss
                else:
                    self._loss_history[b_t, self._loss_counts[b_t]] = b_loss
                    self._loss_counts[b_t] += 1

    def _warmed_up(self) -> bool:
        r"""Checks if the sampling process is warmed up depending on the history size."""
        return torch.all(torch.eq(self._loss_counts, self._history))


class MeanVarX0:
    r"""Easy access for the returns of each functions with predefined attributes opposed to a dictionary"""

    def __init__(self,
                 mean: torch.Tensor,
                 var: torch.Tensor,
                 log_var: torch.Tensor,
                 x_0: torch.Tensor,
                 eps: torch.Tensor):
        self._mean = mean
        self._var = var
        self._log_var = log_var
        self._x_0 = x_0
        self._eps = eps

    def __dict__(self):
        variables = [var for var in dir(self) if not var.startswith('_') and not var.startswith('save')]
        return {var_name: getattr(self, var_name) for var_name in variables}

    @property
    def mean(self) -> torch.Tensor:
        r"""Returns :math:`\tilde{\mu}(x_t, x_0)`"""
        if self._mean is not None:
            return self._mean
        else:
            raise AttributeError(self._error_message('mean'))

    @mean.setter
    def mean(self, val: torch.Tensor):
        self._mean = val

    @property
    def var(self) -> torch.Tensor:
        r"""Returns :math:`\Sigma(x_t, t)^2`, which is either learned or fixed to :math:`\tilde{\beta}`"""
        if self._var is not None:
            return self._var
        elif self._log_var is not None:
            return torch.exp(self._log_var)
        else:
            raise AttributeError(self._error_message('var'))

    @var.setter
    def var(self, val: torch.Tensor):
        self._var = val

    @property
    def log_var(self) -> torch.Tensor:
        if self._log_var is not None:
            return self._log_var
        elif self._var is not None and torch.all(torch.gt(self._var, 0.0)):
            self._log_var = torch.log(self._var)
            return self._log_var
        else:
            raise AttributeError(self._error_message('log_var'))

    @log_var.setter
    def log_var(self, val: torch.Tensor):
        self._log_var = val

    @property
    def x_0(self) -> torch.Tensor:
        r"""Returns :math:`x_0`"""
        if self._x_0 is not None:
            return self._x_0
        else:
            raise AttributeError(self._error_message('x_0'))

    @x_0.setter
    def x_0(self, val: torch.Tensor):
        self._x_0 = val

    @property
    def eps(self) -> torch.Tensor:
        r"""Returns :math:`\epsilon`"""
        if self._eps is not None:
            return self._eps
        else:
            raise AttributeError(self._error_message('eps'))

    @eps.setter
    def eps(self, val: torch.Tensor):
        self._eps = val

    @property
    def sigma(self) -> torch.Tensor:
        r"""Returns :math:`\sigma = \sqrt{\sigma_2} = \sqrt{var}`"""
        if self._log_var is not None:
            return torch.mul(0.5, self._log_var).exp()
        elif self._var is not None:
            return torch.sqrt(self._var)
        else:
            raise AttributeError(self._error_message('sigma'))

    @staticmethod
    def _error_message(prop: str):
        r"""Error message when an attribute is None. This should not happen as it will lead to errors in the code.
        This therefore serves as a precautionary measure to prevent errors further down the line"""
        return f"Property {prop} is None and therefore unavailable. Make sure it is instantiated correctly"


class DiffusionType(enum.Enum):
    DDIM = 'ddim'
    DDPM = 'ddpm'
    EDICT = 'edict'


class VarianceType(enum.Enum):
    LEARNED = 'learned'
    FIXED_SMALL = 'fixed_small'
    FIXED_LARGE = 'fixed_large'
    LEARNED_RANGE = 'learned_range'


class MeanType(enum.Enum):
    MU = 'mu'  # predicting the forward posterior mean \mu_t
    X_0 = 'x_0'  # predicting mean of x_0
    EPSILON = 'epsilon'  # predicting mean of \epsilon (the noise)


class LossType(enum.Enum):
    KL = 'kl'  # KL divergence loss
    SIMPLE = 'simple'  # The simple loss of 2020 - Ho, i.e. the reweighted MSE of the noise prediction (mean)
    HYBRID = 'hybrid'  # The hybrid loss of 2021 - Nichol as a combination of MSE for mean and VLB for var
    MSE = 'mse'  # The initial MSE loss with weighting factors

    def is_vb(self):
        return self in [LossType.KL, LossType.HYBRID]


def process_x0(x0: torch.Tensor, clip_denoised: bool):
    if clip_denoised:
        return x0.clamp(-1., 1.)
    return x0


def split_mean_var(
        model_output: torch.Tensor, 
        x_t: torch.Tensor, 
        t: torch.Tensor, 
        variance_type: VarianceType) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    r"""In the case of learned variance, the output channels of the model are doubled.
    As a result, one needs to split them, which is performed by this function.

    Parameters
    ----------
    model_output : torch.Tensor
        The output of the model i.e. the prediction for the current timestep
    x_t : torch.Tensor
        The noisy version of the input at timestep t
    t : torch.Tensor
        The timestep t

    Returns
    -------
    mean : torch.Tensor
        The mean prediction
    var : torch.Tensor
        The variance prediction
    """
    if variance_type in [VarianceType.LEARNED, VarianceType.LEARNED_RANGE]:
        b, c = x_t.shape[:2]
        assert t.shape == (b,)
        error_msg = f"Make sure that the variance is learned and that the output channels " \
                    f"of the model are doubled for this case"
        assert model_output.shape == (b, 2 * c, *x_t.shape[2:]), error_msg

        # We now split into mean and learned variance along the channel dimension.
        model_mean, model_var = torch.split(model_output, c, dim=1)

        return model_mean, model_var
    else:
        # No split necessary as there is no variance
        return model_output, None


def save_on_crash(log_dir: Optional[str], experiment: Optional[str], **kwargs) -> str:
    r"""Logs all tensors of current scope and returns the logging directory"""
    if log_dir is not None and experiment is not None:
        crash_base_dir = pathlib.Path(log_dir, 'crash_reports')
        crash_dir = pathlib.Path(crash_base_dir, experiment)

        if not crash_dir.exists():
            crash_dir.mkdir(parents=True)

        for name, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                torch.save(value.detach().cpu(), str(pathlib.Path(crash_dir, f"{name}.pt")))

            elif isinstance(value, MeanVarX0):
                save_on_crash(log_dir, experiment, **value.__dict__())
            else:
                continue
    else:
        crash_dir = "Not provided in __init__ of Diffusion. Therefore no logging possible."

    return str(crash_dir)