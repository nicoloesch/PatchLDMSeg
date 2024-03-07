import torch
from torch.amp.autocast_mode import autocast

import torch.nn.functional as F
from typing import Tuple, Optional, Literal
import warnings
import tqdm
from pytorch_lightning import LightningModule

import patchldmseg.utils.misc as pum
import patchldmseg.model.diffusion.diffusion_helper as dh
import patchldmseg.model.diffusion.diffusion_noising as dn
import patchldmseg.model.diffusion.diffusion_losses as dl
from patchldmseg.utils.constants import SAMPLE_TYPE


class DDPMDiffusion:
    r"""
    Class for diffusion process.

    Parameters
    ----------
    gradient_scale: float
        In the case of classifier-guided sampling, how much the gradient of the classifier
        effects the prediction
    loss_type: patchldmseg.model.diffusion.diffusion_helper.LossType
        The type of the loss specified by LossType (SIMPLE, KL or HYBRID)
    num_timesteps: int
        How many forward diffusion steps should be carried out
    max_in_val: float
        Value required for decoder rescaling (i.e. bin width of L0)
    mean_type: patchldmseg.model.diffusion.diffusion_helper.MeanType
        The mean prediction of the diffusion model (X_PREV, X_0, EPSILON)
    noise_schedule: str
        The noising schedule utilised for the diffusion process
    var_type: patchldmseg.model.diffusion.diffusion_helper.VarianceType
        The type of the predicted variance (LEARNED, FIXED, LEARNED_RANGE)
    verbose: bool
        If the diffusion process should be displayed as a Progressbar.
        Easy for debugging to see the progress of encoding and sampling
    verbose : bool
        Whether the TQDM for encoding and decoding images should generate an output. This is useful for debugging
    """

    def __init__(self,
                 loss_type: dh.LossType,
                 mean_type: dh.MeanType,
                 var_type: dh.VarianceType,
                 max_in_val: float,
                 gradient_scale: float = 1.0,
                 num_timesteps: int = 4000,
                 noise_schedule: Literal["linear", "cosine", "linear_scaled"] = 'linear',
                 verbose: bool = False,
                 logging_dir: Optional[str] = None,
                 experiment_name: Optional[str] = None,
                 p_unconditional: float = 0.,
                 **kwargs
                 ):
        super().__init__()

        self._diffusion_type: dh.DiffusionType = dh.DiffusionType.DDPM
 

        # If less diffusion steps are calculated, the image is not noisy enough
        if num_timesteps < 1000:
            warnings.warn(f"Your diffusion step size is set to {num_timesteps}.\n"
                          f"Having a step size of more than 1000 assures that the noise after "
                          f"T steps is isotropic and centred at a mean of 0.\n"
                          f"Diffusion process might not work as intended in the current configuration")

        self._max_in_val = max_in_val

        self._noise_schedule = noise_schedule
        self._num_ts = num_timesteps
        self._p_unconditional = p_unconditional

        self._mean_type = mean_type
        self._var_type = var_type
        self._loss_type = loss_type
        self._gradient_scale = gradient_scale

        # Storage for diffusion steps
        self._sampled_ts = torch.zeros((num_timesteps,))

        self._beta = dn.beta_schedule(schedule=self._noise_schedule,
                                      timesteps=self._num_ts)

        self._alpha = 1. - self._beta
        self._alpha_bar = torch.cumprod(self._alpha, dim=0)

        # Appending these values as the start/end value for the respective \bar{\alpha}
        alpha_prev = torch.as_tensor([1.0])  # NOTE: Defined by Song
        self._alpha_bar_prev = torch.cat((alpha_prev, self._alpha_bar[:-1]))

        self._sqrt_alpha_bar = torch.sqrt(self._alpha_bar)
        self._sqrt_one_minus_alpha_bar = torch.sqrt(1. - self._alpha_bar)
        self._sqrt_recip_alpha_bar_m1 = torch.sqrt(1. / self._alpha_bar - 1.)
        self._log_one_minus_alpha_bar = torch.log(1. - self._alpha_bar)
        self._sqrt_recip_alpha_bar = torch.sqrt(1. / self._alpha_bar)

        # Values for Posterior q(x_{t-1} | x_t, x_0)
        # Posterior Variance
        self._beta_tilde = self._beta * (1. - self._alpha_bar_prev) / (1. - self._alpha_bar)  # Eq.10 Nichol
        # As beta_tilde is 0 for t=0, we clamp it to the lowest non-zero value
        self._log_beta_tilde = torch.log(torch.clamp(self._beta_tilde, min=self._beta_tilde[1]))

        # Posterior Mean Coefficients: Eq.11 Nichol for \mu_tilde
        self._post_mean_coeff1 = torch.sqrt(self._alpha_bar_prev) * self._beta / (1. - self._alpha_bar)
        self._post_mean_coeff2 = torch.sqrt(self._alpha) * (1. - self._alpha_bar_prev) / (1. - self._alpha_bar)

        # time step sampler
        if self._loss_type == dh.LossType.KL:
            # Importance sampling for VLB objective
            sampler = dh.ImportanceSampler
        else:
            sampler = dh.UniformSampler

        self._ts_sampler = sampler(diffusion_steps=num_timesteps)

        # Logging for the tqdm bar
        self._verbose = verbose

        # Logging for crashes
        self._logging_dir = logging_dir
        self._experiment_name = experiment_name

        self._validate_configuration()

    @property
    def num_timesteps(self):
        return self._num_ts
    
    @property
    def p_unconditional(self) -> float:
        return self._p_unconditional

    @property
    def diffusion_type(self) -> dh.DiffusionType:
        return self._diffusion_type

    @property
    def noise_schedule(self) -> str:
        return self._noise_schedule

    @property
    def loss_type(self) -> dh.LossType:
        return self._loss_type

    @property
    def mean_type(self) -> dh.MeanType:
        return self._mean_type

    @property
    def var_type(self) -> dh.VarianceType:
        return self._var_type

    @property
    def gradient_scale(self) -> float:
        return self._gradient_scale

    @property
    def verbose(self) -> bool:
        return self._verbose

    @property
    def logging_dir(self) -> Optional[str]:
        return self._logging_dir

    @property
    def experiment_name(self) -> Optional[str]:
        return self._experiment_name

    @property
    def max_in_val(self) -> float:
        return self._max_in_val

    @property
    def sampler(self) -> dh.AbstractDiffusionSampler:
        r"""Return the sampler"""
        return self._ts_sampler

    @staticmethod
    def get_rng(device: torch.device):
        r"""Has to be initialised every time from scratch as the first sampling from it
        leads then to exactly the same noise every time."""
        return torch.Generator(device).manual_seed(torch.initial_seed())

    @property
    def sampled_ts(self) -> torch.Tensor:
        r"""Returns the sampled time step distribution with a tensor of size (max_ts,),
        and a counter for each timestep"""
        return self._sampled_ts

    def reset_sampled_ts(self):
        self._sampled_ts = torch.zeros_like(self._sampled_ts)

    def _validate_configuration(self):
        r"""Validates if the configuration of loss_type, mean_type, var_type, etc. is allowed"""
        self._check_loss()

    def _check_loss(self):
        if self._loss_type in [dh.LossType.KL, dh.LossType.HYBRID]:
            if self._var_type not in [dh.VarianceType.LEARNED, dh.VarianceType.LEARNED_RANGE]:
                raise AttributeError('Hybrid and KL Loss is only available with a learned variance')
        elif self._loss_type in [dh.LossType.MSE, dh.LossType.SIMPLE]:
            if self._var_type not in [dh.VarianceType.FIXED_SMALL, dh.VarianceType.FIXED_LARGE]:
                raise AttributeError('MSE/Simple Training objective only with fixed variance. Select Hybrid for learned var')
            
    @property
    def timestep_iterator(self):
        return range(self._num_ts)

    def generate_random_timesteps(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates random timesteps of the forward noising process.
        Parameters
        ----------
        batch_size : int
            The batch size for the timestep tensor
        device : torch.device
            Device where the tensor currently lies on (CPU, CUDA)

        Returns
        -------
        timesteps: torch.Tensor
            Tensor of random timesteps of size [Batch Size,]
        weights: torch.Tensor
            Weight associated with each timestep for (optional) reweighting
        """
        return self._ts_sampler.sample(batch_size=batch_size, device=device)

    def q_sample(self,
                 x_0: torch.Tensor,
                 t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""This function resembles the forward noising process at timestep t for the
        beta schedule of noising

        Parameters
        ----------
        x_0 : torch.Tensor
            The input Tensor of shape [BxCxDim]
        t : torch.Tensor
            Timestep for each individual batch element of size [B, ] of the noising process
        noise: torch.Tensor, optional
            Optional noise

        Returns
        -------
        x_t : torch.Tensor
            Noisy version of the input image x_0 at the current time step t
        eps : torch.Tensor
            Random noise

        Notes
        -----
        Follows eq.9 of Nichol

        .. math::
            x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon
        """

        if torch.any(torch.eq(t, -1)):
            # Diffusion process is turned off. Make invalid output for eps
            return x_0, torch.as_tensor([-1])

        # Check if we can sample based on how we set up the noise
        assert torch.max(t) <= self._num_ts
        # Assert that the batch dimensions are the same
        assert x_0.shape[0] == t.shape[0]

        # get the respective :math:`\bar{\alpha}_t values`
        # which are required for the closed form calculation of the noise sample
        sqrt_alpha_bar_t = pum.expand_dims(self._sqrt_alpha_bar, x_0.dim(), x_0.device)[t]
        sqrt_one_minus_alpha_bar_t = pum.expand_dims(self._sqrt_one_minus_alpha_bar,
                                                     x_0.dim(),
                                                     x_0.device)[t]

        # sample noise N(0,1)
        # This is happening due to the reparametrisation trick where we are going from
        # N(\mu,\sigma^2) = \mu + \sigma * epsilon
        if noise is None:
            noise = torch.randn_like(x_0)
        return sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise, noise

    def p_mean_variance(self,
                        x_t: torch.Tensor,
                        t: torch.Tensor,
                        y: Optional[torch.Tensor] = None,
                        location: Optional[torch.Tensor] = None,
                        clip_denoised: bool = True,
                        model_mean: Optional[torch.Tensor] = None,
                        model_var: Optional[torch.Tensor] = None,
                        denoise_fn: Optional[torch.nn.Module] = None,
                        p_unconditional: Optional[float] = None
                        ) -> dh.MeanVarX0:
        r"""One backward step, i.e. apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        Decompose the model output into the prediction of mean and variance of the distribution
        of p(x_{t-1} | x_t), which can be utilised to sample a new image with the reparametrisation trick.

        Notes
        -----
        The reparametrisation trick is formulated as

        .. math::
            \mathcal{N}(\mu, \sigma^2) = \mu + \sigma * \epsilon

        Parameters
        ----------
        denoise_fn : torch.nn.Module, optional
            The model to predict the backwards step
        x_t : torch.Tensor
            Noisy input tensor before the backwards step
        t : torch.Tensor
            The current timestep(s) of the diffusion process
        clip_denoised : bool, optional
            Whether the denoised sample should be clipped to [-1, 1]
        y : torch.Tensor, optional
            Desired class to be sampled. Requires conditional sampling to work
        model_mean : torch.Tensor, optional
            The mean prediction of the model (\mu_\theta)
        model_var : torch.Tensor, optional
            The variance prediction of the model (for learned variances).

        Returns
        -------
        dh.MeanVarX0
            The mean var class with the respective attributes
        """
        if model_mean is None and model_var is None:
            assert denoise_fn is not None

            model_pred = self.calculate_model_pred(
                denoise_fn=denoise_fn,
                x_t=x_t,
                t=t,
                y=y,
                location=location,
                p_unconditional=p_unconditional or self.p_unconditional)

            model_mean, model_var = dh.split_mean_var(model_pred, x_t, t, self.var_type)

        assert model_mean is not None
        assert isinstance(model_var, torch.Tensor) or model_var is None, "Model var must be a tensor or None"
        
        # VARIANCE
        model_var, model_log_var = self._p_var(
            model_var=model_var,
            x_t=x_t,
            t=t)

        # MEAN
        model_mean, x0_pred, eps = self._p_mean(
            model_pred=model_mean,
            x_t=x_t,
            t=t,
            clip_denoised=clip_denoised)
        
        return dh.MeanVarX0(mean=model_mean, var=model_var, log_var=model_log_var, x_0=x0_pred, eps=eps)
    
    def _p_var(self, 
               model_var: Optional[torch.Tensor],
               x_t: torch.Tensor,
               t: torch.Tensor):
        r"""Calculates the variance of the backwards step.
        
        Returns
        -------
        model_var: torch.Tensor, optional
            The predicted (or fixed) variance of timestep t for learned variances.
        model_log_var: torch.Tensor
            The predicted (or fixed) log variance of timestep t
        """

        if self._var_type in [dh.VarianceType.LEARNED, dh.VarianceType.LEARNED_RANGE]:
            assert model_var is not None, "Model variance is None"

            if self._var_type == dh.VarianceType.LEARNED:
                assert model_var.shape == x_t.shape, "The tensors must be of same size. Make sure to split the mean and var in the learned variance case"
                model_log_var = model_var
                model_var = torch.exp(model_log_var)
            else:
                # Range Estimation
                min_log = pum.expand_dims(self._log_beta_tilde, model_var.dim(), x_t.device)[t]
                max_log = pum.expand_dims(torch.log(self._beta), model_var.dim(), x_t.device)[t]

                # The model_var is [-1, 1] for [min_var, max_var] -> rescale to [0, 1]
                # NOTE: This is not a constraint but observed by Dhariwal
                frac = (model_var + 1) / 2
                model_log_var = frac * max_log + (1 - frac) * min_log
                model_var = torch.exp(model_log_var)

        elif self._var_type in [dh.VarianceType.FIXED_SMALL, dh.VarianceType.FIXED_LARGE]:
            assert model_var is None, "Model variance must be none and can not be predicted"

            if self._var_type == dh.VarianceType.FIXED_SMALL:
                model_log_var = pum.expand_dims(self._log_beta_tilde, x_t.dim(), x_t.device)[t]
                model_var = pum.expand_dims(self._beta_tilde, x_t.dim(), x_t.device)[t]

            else:
                model_log_var = pum.expand_dims(torch.log(torch.hstack([self._beta_tilde[1], self._beta[1:]])),
                                                x_t.dim(), x_t.device)[t]
                model_var = pum.expand_dims(self._beta, x_t.dim(), x_t.device)[t]

        else:
            raise NotImplementedError(self._var_type)
        
        return model_var, model_log_var

        
    def _p_mean(self, 
                model_pred: torch.Tensor,
                x_t: torch.Tensor,
                t: torch.Tensor,
                clip_denoised: bool = True,
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Calculate the mean of the backwards step, i.e. p(x_{t-1} | x_t) by using the model prediction.
        This is a safe function as the model_prediction is cloned and not changed in place.
        
        Returns
        -------
        model_pred : torch.Tensor
            The posterior mean of timestep t. Could be either MU, Epsilon or X_0, all being converted to MU
        x0_pred : torch.Tensor
            The predicted x_0
        eps : torch.Tensor
            The predicted noise of the current timestep t
        """

        if self._mean_type == dh.MeanType.MU:
            x0_pred = dh.process_x0(
                self._predict_x0_mean(x_t=x_t,
                                      t=t,
                                      model_pred_mean=model_pred),
                clip_denoised)
            model_mean = model_pred
            eps = self._predict_eps_from_x0(x_t=x_t, t=t, x_0=x0_pred)
        elif self._mean_type in [dh.MeanType.X_0, dh.MeanType.EPSILON]:
            if self._mean_type == dh.MeanType.X_0:
                x0_pred = dh.process_x0(model_pred, clip_denoised)
                eps = self._predict_eps_from_x0(x_t=x_t, t=t, x_0=x0_pred)
            else:
                x0_pred = dh.process_x0(
                    self._predict_x0_mean(x_t=x_t,
                                          t=t,
                                          model_pred_mean=model_pred),
                    clip_denoised)
                eps = model_pred

            model_mean = self.q_posterior_mean_variance(x_t=x_t, x_0=x0_pred, t=t, eps=eps).mean

        else:
            raise RuntimeError(self._mean_type)
        
        return model_mean, x0_pred, eps

    def q_posterior_mean_variance(
            self,
            x_t: torch.Tensor,
            t: torch.Tensor,
            x_0: torch.Tensor,
            eps: torch.Tensor) -> dh.MeanVarX0:
        r"""Calculates the posteriors conditioned on the (predicted) x_0
        in order to reduce the variance of the prediction (given a noised version
        in combination with the original x_0 makes it easier for the model to predict the
        noise contribution).

        Notes
        -----
        The posteriors are defined in Eq. (6) and (7) of 2020 - Ho and Eq. (10) - (12) of
        2021 - Nichol

        .. math::
            q(x_{t-1} \vert x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}(x_t, x_0), \tilde{\beta}_t I)

            \tilde{\mu}_t(x_t, x_0) =
            \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} x_0 +
            \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t

            \tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t

        Parameters
        ----------
        x_t : torch.Tensor
            The input tensor to the backwards/denoising process at timestep(s) t
        t : torch.Tensor
            The timestep(s) for the respective batch
        x_0 : torch.Tensor
            The predicted original image
        eps: torch.Tensor
            The noise of timestep t

        Returns
        -------
        dh.MeanVarX0
            The posterior at the current timestep t

        """

        # Expand dim for safe multiplication along the batch dimension
        post_mean_coeff1_t = pum.expand_dims(self._post_mean_coeff1, x_t.dim(), x_0.device)[t]
        post_mean_coeff2_t = pum.expand_dims(self._post_mean_coeff2, x_t.dim(), x_0.device)[t]
        mean = post_mean_coeff1_t * x_0 + post_mean_coeff2_t * x_t

        log_var = pum.expand_dims(self._log_beta_tilde, x_t.dim(), x_t.device)[t]
        var = pum.expand_dims(self._beta_tilde, x_t.dim(), x_t.device)[t]

        return dh.MeanVarX0(mean=mean, var=var, log_var=log_var, eps=eps, x_0=x_0)

    def _predict_x0_mean(self,
                         x_t: torch.Tensor,
                         t: torch.Tensor,
                         model_pred_mean: torch.Tensor) -> torch.Tensor:
        r"""Predicts the mean of x_0 depending on the model mean type. This is decided upon as
        2 of 3 ways to predict the mean of the noising process depend on x_0, and we can therefore use
        the same functions for sampling regardless of the mean prediction type

        Notes
        -----
        In order to predict the mean of the backwards step 3 methods are available:
            1. Predict :math:`\mu` directly with a neural network
            2. Predict the mean of :math:`x_0` and utilise the posterior formulation to calculate :math:`\mu`
             (Eq. 11 of 2021 - Nichol)
            3. Predict the mean of the noise :math:`\epsilon` and utilise it to calculate :math:`x_0`
             with the reparametrisation trick (Eq. 9 of 2021 - Nichol), which can be used then similar to method 2

        Estimation of mean (and variance) are used for the KL-divergence between the estimation of the
        mean and variance and the forward  process posterior tractable if conditioned on x_0

        Parameters
        ----------
        x_t : torch.Tensor
            The input tensor to the backwards/denoising process at timestep(s) t
        t : torch.Tensor
            The timestep(s) for the respective batch
        model_pred_mean : torch.Tensor
            The prediction of the model depending on the model_pred_type

        Returns
        -------
        torch.Tensor
            The predicted x_0
        """

        if self._mean_type == dh.MeanType.MU:
            # First case: the model predicted the mean directly.
            # NOTE: In order to support the rest of the functions in an easy way, we estimate x_0 by using the
            #   Eq. 11 of Nichol 2021 reformulated to x_0
            return self._predict_x0_from_mu(mu=model_pred_mean, t=t, x_t=x_t)
            
        elif self._mean_type == dh.MeanType.X_0:
            # This is the second case: the model predicted the mean of x_0
            return model_pred_mean

        elif self._mean_type == dh.MeanType.EPSILON:
            # The third case: the model predicted the mean of the noise
            # NOTE: Utilise reparam trick (Eq. 9 - Nichol - to estimate x_0)
            return self._predict_x0_from_eps(eps=model_pred_mean, t=t, x_t=x_t)

        else:
            raise AttributeError(self._mean_type)
        
    def _predict_x0_from_mu(
        self,
        mu: torch.Tensor,
        t: torch.Tensor,
        x_t: torch.Tensor) -> torch.Tensor:       
        r"""Convenience function to calculate x_0 from the predicted mean :math:`\mu_\theta`.
        
        Notes
        -----
        The model predicted the mean of the posterior :math:`\mu_\theta`. We utilise the posterior formulation
        to calculate :math:`x_0` from the predicted mean :math:`\mu_\theta` (Eq. 11 of 2021 - Nichol)

        .. math::
            x_0 = \mu_\theta / coeff1 - coeff2/coeff1 x_t
        """
        post_mean_coeff1_t = pum.expand_dims(self._post_mean_coeff1, x_t.dim(), x_t.device)[t]
        post_mean_coeff2_t = pum.expand_dims(self._post_mean_coeff2, x_t.dim(), x_t.device)[t]

        return mu / post_mean_coeff1_t - post_mean_coeff2_t / post_mean_coeff1_t * x_t
    
    def _predict_mu_from_x0(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        x_t: torch.Tensor) -> torch.Tensor:

        r"""Convenience function to calculate mu from the predicted :math:`x_0`.

        Notes
        -----
        Utilises Eq. 11 of Nichol

        .. math::
            \mu = coeff1 * x_0 + coeff2 * x_t

        """

        post_mean_coeff1_t = pum.expand_dims(self._post_mean_coeff1, x_t.dim(), x_t.device)[t]
        post_mean_coeff2_t = pum.expand_dims(self._post_mean_coeff2, x_t.dim(), x_t.device)[t]

        return post_mean_coeff1_t * x_0 + post_mean_coeff2_t * x_t
        
        
    def _predict_x0_from_eps(
            self,
            eps: torch.Tensor,
            t: torch.Tensor,
            x_t: torch.Tensor) -> torch.Tensor:
        r"""Convenience function to calculate x_0 from the predicted noise :math:`\epsilon_\theta`.
        
        Notes
        -----
        The model predicted the mean of the noise :math:`\epsilon_\theta`. We utilise the reparametrisation 
        trick to calculate :math:`x_0` from the predicted noise :math:`\epsilon_\theta` (Eq. 9 of 2021 - Nichol)

        .. math::
            x_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_\theta}{\sqrt{\bar{\alpha}_t}}
        """

        calculate_old = False
        # NOTE: sqrt(1/alpha) * sqrt(1-alpha) == sqrt(1/alpha - 1) should be the same but floats get different
        #   results at around 1.8e-5 as the max difference. For comparison, calculate in the new way!
        sqrt_recip_alpha_bar_t = pum.expand_dims(self._sqrt_recip_alpha_bar, x_t.dim(), x_t.device)[t]

        if calculate_old:
            sqrt_one_minus_alpha_bar_t = pum.expand_dims(self._sqrt_one_minus_alpha_bar, x_t.dim(), x_t.device)[t]
            return sqrt_recip_alpha_bar_t * (x_t - sqrt_one_minus_alpha_bar_t * eps)
        else:
            sqrt_recip_alpha_bar_m1 = pum.expand_dims(self._sqrt_recip_alpha_bar_m1, x_t.dim(), x_t.device)[t]
            return sqrt_recip_alpha_bar_t * x_t - sqrt_recip_alpha_bar_m1 * eps

    def _predict_eps_from_x0(self,
                             x_t: torch.Tensor,
                             t: torch.Tensor,
                             x_0: torch.Tensor) -> torch.Tensor:
        r"""Calculate :math:`\epsilon_\theta` by using the predicted :math:`x_0`.

        Notes
        -----
        Utilises Eq. 9 of Nichol reformulated to \epsilon

        .. math::
            \epsilon = (x_t - \sqrt{\bar{\alpha}_t} x_0) / \sqrt{1 - \bar{\alpha}_t}

        Parameters
        ----------
        x_t : torch.Tensor
            The input tensor to the backwards/denoising process at timestep(s) t
        t : torch.Tensor
            The timestep(s) for the respective batch
        x_0 : torch.Tensor
            The predicted original image

        Returns
        -------
        torch.Tensor
            The predicted :math:`\epsilon_\theta` based on the predicted math:`x_0`
        """
        # Expand dim for safe multiplication along the batch dimension
        sqrt_alpha_bar_t = pum.expand_dims(self._sqrt_alpha_bar, x_0.dim(), x_0.device)[t]
        sqrt_one_minus_alpha_bar_t = pum.expand_dims(self._sqrt_one_minus_alpha_bar, x_0.dim(), x_0.device)[t]

        return (x_t - x_0 * sqrt_alpha_bar_t) / sqrt_one_minus_alpha_bar_t

    def condition(self,
                  denoise_fn: torch.nn.Module,
                  mean_var_x0: dh.MeanVarX0,
                  x_t: torch.Tensor,
                  y: Optional[torch.Tensor],
                  t: torch.Tensor,
                  classifier: Optional[LightningModule] = None,
                  location: Optional[torch.Tensor] = None) -> dh.MeanVarX0:
        r"""Convenience function to potentially condition the input mean_var_x0.
        Conditioning is based on the model being used and the respective implementation
        of _class_free_guidance and _class_guidance. Conditioning also only takes place
        if the classifier or self.p_unconditional are not None/ larger than 0."""

        if self._gradient_scale == 0.:
            return mean_var_x0

        if self.p_unconditional > 0.:
            #assert y is not None, "Ground-truth tensor y is None but is required for conditioning"
            assert classifier is None, "Classifier is not allowed if unconditional sampling is performed"
            return self._class_free_guidance(
                denoise_fn=denoise_fn,
                x_t=x_t,
                y=y,
                t=t,
                location=location,
                mean_var_x0_conditional=mean_var_x0)

        elif classifier is not None:
            assert y is not None, "Ground-truth tensor y is None but is required for conditioning"
            assert self.p_unconditional == 0., "Unconditional sampling is not allowed if classifier is used"
            return self._class_guidance(
                mean_var_x0=mean_var_x0,
                x_t=x_t,
                y=y,
                t=t,
                classifier=classifier,
                location=location
            )
        else:
            return mean_var_x0

        
    def _class_guidance(
            self,
            mean_var_x0: dh.MeanVarX0,
            x_t: torch.Tensor,
            y: torch.Tensor,
            t: torch.Tensor,
            classifier: LightningModule,
            location: Optional[torch.Tensor] = None) -> dh.MeanVarX0:
        r"""Conditions the sampling process based on the mean 
        (conditional reverse noising process) as it is a DDPM model.

        Notes
        -----
        Conditioning mechanism outlined in Section 4.1 of Dhariwal et al. 2021
        or in Algorithm 1 of Dhariwal et al. 2021

        Parameters
        ----------
        diffusion_type : dh.DiffusionType
            The type of the diffusion process (for sampling). One of [DDIM, DDPM]
        mean_var_x0 : dh.MeanVarX0
            The predicted mean, variance and x0
        x_t : torch.Tensor
            The input tensor to the backwards/denoising process at timestep(s) t. REQUIRES GRAD
        y : torch.Tensor
            Desired class to be conditioned on
        t : torch.Tensor
            The timestep(s) for the respective batch
        classifier: torch.nn.Module
            The classifier to predict the label of the current input tensor
        location: torch.Tensor, optional
            The location of the patch in the case of patch_based sampling.

        Returns
        -------
        dh.MeanVarX0
            The new predictions conditioned on the input label y
        """

        # Gradient calculation
        gradient = self._gradient(x_t=x_t, y=y, t=t, classifier=classifier, location=location)

        mean_var_x0.mean = mean_var_x0.mean + mean_var_x0.var * gradient * self._gradient_scale
        return mean_var_x0
    
    def _class_free_guidance(
            self,
            denoise_fn: torch.nn.Module,
            x_t: torch.Tensor,
            y: torch.Tensor,
            t: torch.Tensor,
            location: Optional[torch.Tensor] = None,
            mean_var_x0_conditional: Optional[dh.MeanVarX0] = None):
        r"""Classifier free guidance for the sampling process. The model is required 
        to be trained with p_unconditional > 0 in order to use this function.
        """

        assert self.p_unconditional > 0., "Classifier free guidance is only available if p_unconditional > 0"
        assert denoise_fn.p_unconditional > 0., "Classifier free guidance is only available if p_unconditional > 0"  # type: ignore

        if mean_var_x0_conditional is None:
            mean_var_x0_conditional = self.p_mean_variance(
                x_t=x_t,
                t=t,
                denoise_fn=denoise_fn,
                y=y,
                location=location,
                p_unconditional=0.
            )

        mean_var_x0_unconditional = self.p_mean_variance(
            x_t=x_t,
            t=t,
            denoise_fn=denoise_fn,
            y=y,
            location=location,
            p_unconditional=1.
        )

        # Mean
        eps_mixed = (1+self.gradient_scale)*mean_var_x0_conditional.eps - self.gradient_scale*mean_var_x0_unconditional.eps
        x_0_mixed = self._predict_x0_from_eps(eps=eps_mixed, t=t, x_t=x_t)
        mu_mixed = self._predict_mu_from_x0(x_0=x_0_mixed, t=t, x_t=x_t)

        var_mixed = (1+self.gradient_scale)*mean_var_x0_conditional.var - self.gradient_scale*mean_var_x0_unconditional.var
        log_var_mixed = torch.log(var_mixed)

        return dh.MeanVarX0(mean=mu_mixed, var=var_mixed, log_var=log_var_mixed, eps=eps_mixed, x_0=x_0_mixed)


    @staticmethod
    def _gradient(x_t: torch.Tensor, 
                  t: torch.Tensor,
                  y: torch.Tensor,
                  classifier: LightningModule,
                  location: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""Calculate the gradient of log(p(y|x)) based on the prediction of a classifier.

        Parameters
        ----------
        y : torch.Tensor
            Desired class to be conditioned on
        x_t : torch.Tensor
            Input image to the classifier at timestep t
        t : torch.Tensor
            The timestep(s) for the respective batch

        Returns
        -------
        torch.Tensor
            The gradient of log(p(y|x))

        """
        assert y is not None
        assert y.dim() == 1, "y needs to be a single dimensional tensor of length batch size with a class idx at each position"
        assert not torch.is_inference_mode_enabled(), "Gradient calculation requires inference mode to be disabled. --trainer.inference_mode=False"

        with torch.enable_grad():
            x_in = x_t.detach().requires_grad_(True)
            # We require full precision here as the classifier returns otherwise NaNs
            with autocast(classifier.device.type, enabled=False):
                y_hat = classifier(x_t=x_in, t=t, y=None, location=location)
                assert len(y_hat) == len(y), "Batch size mismatch between y and y_hat"
                log_probs = F.log_softmax(y_hat, dim=-1)
                selected = log_probs[:, y]
                grad = torch.autograd.grad(selected.sum(), x_in)[0]
            return grad

    @torch.no_grad()   
    def sample_healthy(
            self,
            denoise_fn: torch.nn.Module,
            x_0: torch.Tensor,
            location: Optional[torch.Tensor] = None,
            first_n_timesteps: Optional[int] = None,
            classifier: Optional[LightningModule] = None,
            *args,
            **kwargs) -> torch.Tensor:
        
        r"""Convenience function to sample healthy images. This includes the encoding
        and the decoding of the image. This is a wrapper around sample_image and
        has almost similar functionality."""
        
        num_images = x_0.shape[0]
        y = torch.zeros((num_images), device=x_0.device, dtype=torch.long)

        encoded_image, _ = self.q_sample(
            x_0=x_0,
            t=torch.full(size=(num_images,), 
                         fill_value=(first_n_timesteps or self._num_ts) - 1, # -1 as we start indexing at 0
                         device=x_0.device),
        )

        return self.sample_image(
            denoise_fn=denoise_fn,
            noisy_img=encoded_image,
            y=y,
            first_n_timesteps=first_n_timesteps,
            classifier=classifier,
            conditional_sampling=True,
            location=location
        )

    @torch.no_grad()
    def sample_image(
            self,
            denoise_fn: torch.nn.Module,
            num_images: Optional[int] = None,
            inputs: Optional[torch.Tensor] = None,
            noisy_img: Optional[SAMPLE_TYPE] = None,
            y: Optional[torch.Tensor] = None,
            location: Optional[torch.Tensor] = None,
            first_n_timesteps: Optional[int] = None,
            classifier: Optional[LightningModule] = None,
            conditional_sampling=False,
            eta: Optional[float] = None) -> torch.Tensor:
        r"""Function to sample an image after the model has learnt to estimate
        the noise. This is the backwards process in its entirety and calculates
        x_0 starting from random noise at x_T.

        Parameters
        ----------
        denoise_fn : torch.nn.Module
            The model to predict the backwards step
        num_images: int, optional
            Number of images to be sampled
        y : torch.Tensor, optional
            Desired class to be sampled
        location : torch.Tensor, optional
            The location of the patch to be sampled
        classifier : nn.Module, optional
            Classifier for conditional sampling
        conditional_sampling : bool
            If conditional sampling based on the ground truth image scale weak labels should be used.
            Make sure the diffusion model was trained with the same parameter
        inputs: torch.Tensor, optional
            The input tensor. Is required to get the spatial dimensions, dtype and device right
        noisy_img: torch.Tensor, optional
            A potential noisy version of inputs. This is important for encoding the image.
        first_n_timesteps : int 
            The number of timesteps to sample. If None, the full schedule is sampled

        Notes
        -----
        Subclasses may overwrite this function and the respective _sample method is then called
        for the subclass

        Returns
        -------
        torch.Tensor
            A sampled tensor representing a generated image
        """
        if noisy_img is not None:
            x_t = noisy_img
            rng = None
            if isinstance(x_t, (list, tuple)):
                assert self.diffusion_type == dh.DiffusionType.EDICT, "Only EDICT supports tuple inputs for x_t"
                num_images = x_t[0].shape[0]
            elif isinstance(x_t, torch.Tensor):
                num_images = x_t.shape[0]
            else:
                raise RuntimeError(f"noisy_img must be either a tensor or a tuple of tensors. Got {type(noisy_img)}")
        else:
            assert inputs is not None and num_images is not None
            shape = (num_images,) + inputs.shape[1:]
            # Note: This generates the same noise over and over again
            rng = self.get_rng(inputs.device)
            x_t = torch.empty(
                size=shape,
                device=inputs.device, dtype=inputs.dtype).normal_(generator=rng)

        device = x_t.device if isinstance(x_t, torch.Tensor) else x_t[0].device  

        if conditional_sampling:
            if y is None:
                raise RuntimeError(f"y is None but conditional sampling is active. Make sure to provide"
                                    f"the target labels for the sampling process.")
            elif not denoise_fn.conditional_sampling:
                raise RuntimeError(f"The model has not been trained with conditional sampling,"
                                    f"which prevents the utilisation of conditional sampling now")

            
        timesteps_iterator = tuple(self.timestep_iterator)
        if first_n_timesteps is not None:
            assert first_n_timesteps <= len(timesteps_iterator), "first_n_timesteps must be smaller than the number of timesteps"
            timesteps_iterator = timesteps_iterator[:first_n_timesteps]

        for i, t_iterator in enumerate(
            tqdm.tqdm(
                reversed(timesteps_iterator),
                disable=not self._verbose,
                total=len(timesteps_iterator),
                desc=f"{self.diffusion_type.value} Sampling",
                position=1,  # to not interfere with the one from pytorch lightning
                leave=False)
        ):
            t = torch.full(
                size=(num_images,), 
                fill_value=t_iterator, 
                device=device)
            
            x_t = self._sample_single_step(
                x_t=x_t,
                t=t,
                denoise_fn=denoise_fn,
                y=y,
                location=location,
                classifier=classifier,
                eta=eta,
                num_images=num_images,
                rng=rng,
                total_iterations=len(timesteps_iterator),
                current_iteration=i
            )

        self._check_sampling_return_type(x_t)
        return x_t  # type: ignore
    
    
    def _check_sampling_return_type(self, x_t: SAMPLE_TYPE):
        r"""This function is called at the end of sample_image. As `sample_image` unifies the different
        sampling functions, different return types are to be expected and need to be verified before returning
        them to the user."""
        
        _class_name = self.__class__.__name__
        _error_msg = (f"Wrong return type. {_class_name} expects torch.Tensor as return type of sampling. Got {type(x_t)}.")
        assert isinstance(x_t, torch.Tensor), _error_msg
        
    def _sample_single_step(
            self,
            x_t: SAMPLE_TYPE,
            t: torch.Tensor,
            denoise_fn: torch.nn.Module,
            y: Optional[torch.Tensor] = None,
            location: Optional[torch.Tensor] = None,
            classifier: Optional[LightningModule] = None,
            eta: Optional[float] = None,
            num_images: Optional[int] = None,
            rng: Optional[torch.Generator] = None,
            total_iterations: Optional[int] = None,
            current_iteration: Optional[int] = None
    ) -> SAMPLE_TYPE:
        r"""Convenience function to united the sampling of a single step 
        and facilitate the usage of the same sample function for all
        diffusion models."""
        
        assert isinstance(x_t, torch.Tensor), f"{self.diffusion_type} does not support tuple inputs for x_t. This is reserved for EDICT."
        num_images = num_images or x_t.shape[0]

        p_mean_var_x0 = self.p_mean_variance(
            x_t=x_t, 
            t=t,
            denoise_fn=denoise_fn,
            y=y,
            location=location,
            clip_denoised=True,
            p_unconditional=0.) # Fully conditional sampling

        # This conditions only based on the attributes from above/how the class was initialised
        p_mean_var_x0 = self.condition(
            mean_var_x0=p_mean_var_x0,
            denoise_fn=denoise_fn,
            x_t=x_t,
            t=t,
            y=y,
            classifier=classifier,
            location=location)

        # Sample x_{t-1} (but override x_t, so I can use it in the next step with the same variable)
        x_t = self._sampling_fn(
            p_mean_var_x0=p_mean_var_x0,
            t=t,
            x_t=x_t,
            generator=rng,
            eta=eta)
    
        return x_t

    def _sampling_fn(
            self,
            t: torch.Tensor,
            x_t: torch.Tensor,
            p_mean_var_x0: Optional[dh.MeanVarX0] = None,
            model_pred: Optional[torch.Tensor] = None,
            generator: Optional[torch.Generator] = None,
            noise: Optional[torch.Tensor] = None,
            *args,
            **kwargs) -> torch.Tensor:
        r"""Sampling of x_{t-1} of the DDPM (denoising diffusion probabilistic model.)

        Notes
        -----
        Sampling takes place following the reparametrisation trick:

        .. math::
            x_{t-1} = \mu_\theta(x_t,t) + \sigma_t * z, z \in \mathcal(N)(0,1)

        Both :math:`\mu` and :math:`\sigma` are obtained through p_mean_var_x0. This formulation
        then follows the posterior variance formulation (Eq. 11 Nichol) for the mean in combination
        with Eq. 9 reformulated to :math:`x_0`. Combining the reparam trick with the aforementioned
        formulas, one gets to the formulation by Ho et. al (Algorithm 2, Line 4 Ho ):

        .. math::
            x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left(x_t - \frac{\beta_t}{\sqrt{1- \bar{\alpha}_t}}
            \epsilon_\theta(x_t, t)\right) + \sigma_t*z

        Parameters
        ----------
        p_mean_var_x0 : dh.MeanVarX0, optional
            The predicted mean, variance and x0 of the model decomposed by p_mean_variance(). If not 
            provided, it is calculated internally using the model_pred
        x_t: torch.Tensor
            The input tensor, which needs to be fed through the backwards process
        t : torch.Tensor
            The current timestep
        generator: torch.Generator, optional
            A random-number generator. If specified, it results in sampling the same noise over and over
            again. If not, arbitrary noise is sampled.
        model_pred: torch.Tensor, optional
            The prediction of the model. Is designed only for unittests and should not be used in real
            applications (refer to p_mean_var_x0).
        noise: torch.Tensor, optional
            Random sampled noise. Is used for testing and should not be used in production.

        Returns
        -------
        torch.Tensor
            The predicted sample x_{t-1}
        """
        if p_mean_var_x0 is None:
            assert model_pred is not None, "Provide either p_mean_var_x0 or model_pred (preferred)"
            # Just for testing! p_mean_var should be specified!
            model_mean, model_var = dh.split_mean_var(
                model_output=model_pred,
                x_t=x_t,
                t=t,
                variance_type=self.var_type)
            p_mean_var_x0 = self.p_mean_variance(
                x_t=x_t, 
                t=t,
                model_mean=model_mean,
                model_var=model_var,
                clip_denoised=True,
                p_unconditional=0.) # Fully conditional sampling
            
        assert p_mean_var_x0 is not None, "p_mean_var_x0 is None"

        if noise is None:
            noise = torch.empty_like(x_t).normal_(generator=generator)

        # Mask noise depending on where t!=0 as the initial DDPM learns this step intrinsically
        noise_mask = pum.expand_dims(torch.ne(t, 0), x_t.dim(), x_t.device)

        # Expand dim for safe multiplication along the batch dimension
        mean = pum.expand_dims(p_mean_var_x0.mean, x_t.dim(), x_t.device)
        sigma = pum.expand_dims(p_mean_var_x0.sigma, x_t.dim(), x_t.device)  # sigma not var (= sigma**2)

        x_t_m1 = mean + sigma * noise * noise_mask

        if torch.any(torch.isnan(x_t_m1)):
            crash_dir = dh.save_on_crash(self._logging_dir, experiment=self._experiment_name, **locals())
            raise RuntimeError(f"NaN values encountered. Artefacts saved to {crash_dir}")

        return x_t_m1

    def calculate_loss(self,
                       x_0: torch.Tensor,
                       t: torch.Tensor,
                       denoise_fn: Optional[torch.nn.Module] = None,
                       noise: Optional[torch.Tensor] = None,
                       y: Optional[torch.Tensor] = None,
                       location: Optional[torch.Tensor] = None,
                       model_pred: Optional[torch.Tensor] = None,
                       x_t: Optional[torch.Tensor] = None,
                       p_unconditional: Optional[float] = None) -> torch.Tensor:
        r"""Calculate the loss of the diffusion step

        Parameters
        ----------
        x_0 : torch.Tensor
            The initial starting image obtained from the sampler.
        noise : torch.Tensor, optional
            Random noise \epsilon following Eq. 9 of 2021 - Nichol obtained through q_sample
        t : torch.Tensor
            The current timestep(s)
        denoise_fn : torch.nn.Module, optional
            The model to predict the backwards step. Despite being optional, it is the main
            component of this function and should be provided instead of model_pred.
        y : torch.Tensor, optional
            Desired class to be sampled. Requires conditional sampling to work
        model_pred: torch.Tensor, optional
            Prediction of the denoise_fn. Is designed only for unittests and should not be used in real
            applications (refer to denoise_fn).
        location: torch.Tensor, optional
            Patch location for positional embedding
        Returns
        -------
        torch.Tensor
            The loss as (B,...)
        """
        if x_t is None:
            x_t, noise = self.q_sample(x_0=x_0, t=t, noise=noise)
        else:
            assert noise is not None, 'Provide x_t and noise from q_sample'

        _error_msg = (f"Selected loss {self._loss_type} is not supported.\n"
                      f"Select a loss of {list(dh.LossType)}")

        assert self._loss_type in [
            dh.LossType.KL, dh.LossType.HYBRID, dh.LossType.SIMPLE, dh.LossType.MSE], _error_msg

        loss_mse = torch.zeros(x_0.shape[0], device=x_0.device)
        loss_vlb = torch.zeros(x_0.shape[0], device=x_0.device)

        # Real probability distribution, x0 and noise -> the target
        q_mean_var_x0 = self.q_posterior_mean_variance(x_t=x_t, x_0=x_0, t=t, eps=noise)

        if model_pred is None:
            assert denoise_fn is not None
            model_pred = self.calculate_model_pred(
                denoise_fn=denoise_fn,
                x_t=x_t,
                t=t,
                y=y,
                location=location,
                p_unconditional=p_unconditional or self.p_unconditional)  # We are in training therefore we need to mask

        assert model_pred is not None

        # L_VLB including L_0
        if self._loss_type in [dh.LossType.KL, dh.LossType.HYBRID]:
            assert self._var_type in [dh.VarianceType.LEARNED, dh.VarianceType.LEARNED_RANGE], "KL and Hybrid loss only supports learned variance"
             # NOTE: Overwrite model_pred as it is used further down in the MEAN calculation.
            model_pred, model_var = dh.split_mean_var(
                model_output=model_pred,
                x_t=x_t,
                t=t,
                variance_type=self.var_type)

            if self._loss_type == dh.LossType.HYBRID:
                model_mean = model_pred.detach()  # Freeze the mean
            else:
                model_mean = model_pred
               
            p_mean_var_x0_vlb = self.p_mean_variance(
                denoise_fn=denoise_fn,
                x_t=x_t,
                t=t,
                y=y,
                location=location,
                clip_denoised=False,
                model_var=model_var,
                model_mean=model_mean,
                p_unconditional=self.p_unconditional)

            loss_vlb = dl.loss_vlb(variance_type=self._var_type,
                                   t=t,
                                   p_mean_var_x0=p_mean_var_x0_vlb,
                                   q_mean_var_x0=q_mean_var_x0,
                                   max_in_val=self._max_in_val)
            if self._loss_type == dh.LossType.HYBRID:
                lambda_ = 0.001  # Rescaling factor of lambda for the L_{VLB}
                loss_vlb = loss_vlb * lambda_

        # L_{t-1} (mean) between model_output and target
        if self._loss_type in [dh.LossType.HYBRID, dh.LossType.SIMPLE, dh.LossType.MSE]:
            target = {
                dh.MeanType.X_0: q_mean_var_x0.x_0,
                dh.MeanType.MU: q_mean_var_x0.mean,
                dh.MeanType.EPSILON: q_mean_var_x0.eps
            }[self._mean_type]

            loss_mse = pum.non_batch_mean(torch.pow(torch.sub(model_pred, target), 2))

            if self._loss_type == dh.LossType.MSE:
                # MSE without dropping the weights
                weight = {
                    # 2020 - Ho - Eq. 8
                    dh.MeanType.MU: 1. / (2. * q_mean_var_x0.var),
                    # 2020 - Ho - Eq. 8 and reformulation of reparam trick
                    # NOTE: Not the same as Rombach! Don't know how Rombach got their value
                    dh.MeanType.X_0: 0.5 * (
                            (self._alpha_bar_prev[t] * self._beta[t]) / (
                            (1. - self._alpha_bar[t]) * (1. - self._alpha_bar_prev[t]))
                    ),
                    # 2020 - Ho - Eq. 12
                    dh.MeanType.EPSILON:
                        self._beta[t].pow(2) / (
                                2. * q_mean_var_x0.var * self._alpha[t] * (1 - self._alpha_bar[t])
                        ),
                }[self._mean_type]

                loss_mse = loss_mse * pum.expand_dims(weight, loss_mse.dim(), loss_mse.device)

        return loss_vlb + loss_mse

    # IDEA: Output the different losses compartmentalised? Similar to the calc_bpd_loop of Dhariwal

    def calculate_model_pred(
            self,
            denoise_fn: torch.nn.Module,
            x_t: torch.Tensor,
            t: torch.Tensor,
            y: Optional[torch.Tensor] = None,
            location: Optional[torch.Tensor] = None,
            p_unconditional: float = 0.
    ) -> torch.Tensor:
        r"""Convenience function to calculate the model prediction.
        Required to not depend on DDIM in DDPM to prevent circular import,
        as the calculation changes due to the subsequence of DDIM."""
        return denoise_fn(x_t=x_t, t=t, y=y, location=location, p_unconditional=p_unconditional)