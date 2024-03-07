import torch
from typing import Literal


def generate_subsequence(timesteps: int, sequence_length: int):
    # Option 1: Do also the first step and the last - probs not necessary and adds additional step
    # NOTE: Reversed already included due to start & stop reverse
    # timesteps_tensor = torch.linspace(
    #     start=self.num_timesteps-1, end=0, steps=subsequence_step_num+1, dtype=torch.int16)

    # Option 2: Skip already from the start
    return torch.arange(0, timesteps, step=timesteps // sequence_length)


def beta_schedule(schedule: Literal['linear', 'cosine', 'linear_scaled'], timesteps: int):
    if schedule == 'linear':
        return linear_schedule(timesteps)
    elif schedule == 'cosine':
        return cosine_schedule(timesteps)
    elif schedule == 'linear_scaled':
        # LDM specific schedule
        return linear_scaled_schedule(timesteps)
    else:
        raise NotImplementedError(f"Schedule {schedule} is not implemented.")


def linear_scaled_schedule(timesteps: int,
                          start: float = 0.00085,
                          end: float = 0.012,
                          dtype: torch.dtype = torch.float64) -> torch.Tensor:
    
    return torch.linspace(start=start**0.5, end=end**0.5, steps=timesteps, dtype=dtype)**2


def linear_schedule(timesteps: int,
                    start: float = 1e-4,
                    end: float = 2e-2,
                    dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """This is the beta schedule for the forward noising process. According to the linear schedule
    of Ho et al., the respective start and end values are chosen due to the following reason:

    ... to be small relative to data scaled to [-1,1] ensuring that reverse and forward processes
    have approximately the same functional form while keeping the signal-to-noise ratio at x_T
    (the end of the diffusion process) as small as possible
    """
    return torch.linspace(start=start, end=end, steps=timesteps, dtype=dtype)


def cosine_schedule(timesteps: int,
                    min_max_beta: tuple = (0.001, 0.999),
                    s: float = 8e-3,
                    pixel_bin_width: float = 1. / 127.5,
                    dtype: torch.dtype = torch.float64) -> torch.Tensor:
    r"""Cosine schedule from Nichol - Improved Denoising Probabilistic Models.

    Parameters
    ----------
    timesteps : int
        Timesteps of the forward noising process

    min_max_beta : tuple
        The minimum and maximum beta for the noising process intended to prevent singularities
    s : float
        Normalising constant that assures that ..math:: \sqrt(\beta_0) is smaller than the pixel bin width.
        Note: It appears that this is arbitrary
    pixel_bin_width : float
        The width of each pixel bin. As we are scaling down from [0.255] to [-1,1] in the original paper,
        each of the two pixels [-1,0)/(0,1] covers 255/2=127.5 pixel values
    dtype: torch.dtype
        The data type of the output

    Returns
    -------
    beta : torch.Tensor
        Returns beta_t for all t clipped to min_max_beat
    """

    # s = DiffusionProcess._estimate_s(self._max_ts, pixel_bin_width)

    # I am extending it with +1 to get also the alpha_t_minus1 into it
    timestep = torch.arange(start=0, end=timesteps+1, dtype=dtype)

    f_t = torch.cos((timestep / timesteps + s) / (s + 1) * (torch.pi / 2.)) ** 2
    f_0 = f_t[0]

    # As I am going to max + 1, I removed the first index
    alpha_bar_t = (f_t / f_0)[1:]

    # The minus one removes the last index as I am going to max + 1
    alpha_bar_t_minus1 = (f_t / f_0)[:-1]

    min_beta, max_beta = min_max_beta

    beta_t = torch.clip(1. - alpha_bar_t / alpha_bar_t_minus1, min_beta, max_beta)

    return beta_t


def estimate_s(max_ts: int, pixel_bin_width: float = 1. / 127.5):
    r""" This function estimates based on the equation 17 of Nichol2021
    based on the maximum time steps T and the desired \sqrt{beta_0},
    which according to the authors should be slightly smaller than the pixel_bin_width"""
    # IDEA: Brute Froce estimation based on pixel_bin_width.
    min_sqrt_beta_0 = pixel_bin_width - 1e-4
    acos_sqrt_beta_0 = torch.acos(torch.as_tensor(min_sqrt_beta_0))
    s = -acos_sqrt_beta_0 / (max_ts * (1. - acos_sqrt_beta_0))

    # Round to first non-zero decimal
    return float(f'{s:.1g}')