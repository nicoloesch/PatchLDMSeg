
import torch
import tqdm
import patchldmseg.model.diffusion.diffusion_helper as dh
import patchldmseg.model.diffusion.diffusion_noising as dn
from patchldmseg.model.diffusion.ddpm import DDPMDiffusion
from patchldmseg.utils.constants import SAMPLE_TYPE
import patchldmseg.utils.misc as pum

from pytorch_lightning import LightningModule


from typing import Literal, Optional


class DDIMDiffusion(DDPMDiffusion):
    r"""Extends the base class diffusion process by additional DDIM methods. This includes mainly
    the option for a subsequence, and the reverse encoding of an image.

    Notes
    -----
    This class is only there for generative purposes. Therefore, every training call will be suppressed immediately
    resulting in a RuntimeError. This is as the parameters will be modified to support subsequence sampling.

    Parameters
    ----------
    gradient_scale: float
        In the case of classifier-guided sampling, how much the gradient of the classifier
        effects the prediction
    loss_type: patchldmseg.model.diffusion.diffusion_helper.LossType
        The type of the loss specified by LossType (SIMPLE, KL or HYBRID)
    num_timesteps: int
        How many forward diffusion steps should be carried out
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
    subsequence_length: int, optional
        Only applicable for DDIM! The desired length of the subsequence.
        If not specified, the subsequence is the same length as the
        number of timesteps i.e. no steps are skipped
    """

    def __init__(self,
                 loss_type: dh.LossType,
                 mean_type: dh.MeanType,
                 var_type: dh.VarianceType,
                 eta: float,
                 max_in_val: float,
                 gradient_scale: float = 1.0,
                 num_timesteps: int = 4000,
                 noise_schedule: Literal["linear", "cosine"] = 'linear',
                 verbose: bool = False,
                 logging_dir: Optional[str] = None,
                 experiment_name: Optional[str] = None,
                 subsequence_length: Optional[int] = None,
                 p_unconditional: float = 0.
                 ):
        super().__init__(loss_type=loss_type,
                         mean_type=mean_type,
                         var_type=var_type,
                         gradient_scale=gradient_scale,
                         max_in_val=max_in_val,
                         num_timesteps=num_timesteps,
                         noise_schedule=noise_schedule,
                         verbose=verbose,
                         logging_dir=logging_dir,
                         experiment_name=experiment_name,
                         p_unconditional=p_unconditional)

        self._diffusion_type = dh.DiffusionType.DDIM  # Overwrite the diffusion_type
        self._eta = eta
        eta2 = eta ** 2
        subsequence_length = subsequence_length or num_timesteps
        self._subsequence = dn.generate_subsequence(num_timesteps, subsequence_length)
        self._subsequence_length = subsequence_length
        self._num_ts = subsequence_length
        self._init_num_ts = num_timesteps  # New attribute to have the initial diffusion steps
        self._sampled_ts = self._sampled_ts[self._subsequence]
        self._alpha_bar = self._alpha_bar[self._subsequence]
        self._alpha_bar_prev = torch.cat([torch.ones(1, dtype=torch.float64), self._alpha_bar[:-1]], dim=0)
        self._alpha = self._alpha_bar / self._alpha_bar_prev
        self._beta = 1. - self._alpha
        self._log_one_minus_alpha_bar = torch.log(1. - self._alpha_bar)
        self._sqrt_alpha_bar = torch.sqrt(self._alpha_bar)
        self._sqrt_one_minus_alpha_bar = torch.sqrt(1. - self._alpha_bar)

        # Encoding
        self._alpha_bar_next = torch.cat((self._alpha_bar[1:], torch.zeros(1, dtype=torch.float64)))

        # Values for Posterior q(x_{t-1} | x_t, x_0)
        # Posterior Variance
        # NOTE: One could also add eta**2 to the end if the class is initialised with that - also nice to have it
        #  in the function as I can then sample multiple different sequences with a single instance of DDIM
        self._beta_tilde = self._beta * (1. - self._alpha_bar_prev) / (1. - self._alpha_bar)  # Eq.10 Nichol
        self._log_beta_tilde = torch.log(torch.clamp(self._beta_tilde, min=self._beta_tilde[1]))

        # coefficients to recover x_0 from x_t and \epsilon_t
        self._sqrt_recip_alpha_bar = torch.sqrt(1. / self._alpha_bar)
        self._sqrt_recip_alpha_bar_m1 = torch.sqrt(1. / self._alpha_bar - 1.)

        # Posterior Mean Coefficients: Eq.11 Nichol for \mu_tilde
        # This is the generalised form with eta determining the variance
        #self._post_mean_coeff2 = (torch.sqrt(1 - self._alpha_bar - eta2 * self._beta)
        #                          * torch.sqrt(1 - self._alpha_bar_prev) / (1. - self._alpha_bar))
        #self._post_mean_coeff1 = torch.sqrt(self._alpha_bar_prev) * (1. - torch.sqrt(self._alpha) * self._post_mean_coeff2)
        
        # Posterior Mean Coefficients: Eq.11 Nichol for \mu_tilde
        self._post_mean_coeff1 = torch.sqrt(self._alpha_bar_prev) * self._beta / (1. - self._alpha_bar)
        self._post_mean_coeff2 = torch.sqrt(self._alpha) * (1. - self._alpha_bar_prev) / (1. - self._alpha_bar)


    @property
    def subsequence(self) -> torch.Tensor:
        return self._subsequence

    def ddim_t_to_ddpm_t(self, t_ddim: torch.Tensor) -> torch.Tensor:
        r"""By providing the option to shorten the sequence, the timesteps of the DDIM
        are changed and do no longer reflect the original DDPM. This function maps the
        new ddim timesteps to the old ddpm timesteps"""

        if t_ddim.device != self.subsequence.device:
            self._subsequence = self.subsequence.to(t_ddim.device)

        return self.subsequence.gather(0, t_ddim)

    @property
    def eta(self):
        return self._eta

    @classmethod
    def from_ddpm(cls,
                  ddpm_instance: DDPMDiffusion,
                  eta: float,
                  subsequence_length: Optional[int] = None):
        attribs = {attrib: getattr(ddpm_instance, attrib) for attrib in (
            'max_in_val', 'loss_type', 'mean_type', 'var_type', 'gradient_scale', 'num_timesteps','noise_schedule', 'verbose', 'logging_dir', 'experiment_name', 'p_unconditional'
            )}
        # Add new attributes
        attribs.update({
            'subsequence_length': subsequence_length,
            'eta': eta})
        return cls(**attribs)

    def calculate_loss(self,
                       x_0: torch.Tensor,
                       t: torch.Tensor,
                       denoise_fn: torch.nn.Module,
                       noise: Optional[torch.Tensor] = None,
                       y: Optional[torch.Tensor] = None,
                       model_pred: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise RuntimeError("This class can only be used for sampling in the current iteration and implementation")
    
    def calculate_model_pred(
            self,
            denoise_fn: torch.nn.Module,
            x_t: torch.Tensor,
            t: torch.Tensor,
            y: Optional[torch.Tensor] = None,
            location: Optional[torch.Tensor] = None,
            p_unconditional: float = 0.,
    ) -> torch.Tensor:
        return super().calculate_model_pred(
            denoise_fn=denoise_fn,
            x_t=x_t,
            t=self.ddim_t_to_ddpm_t(t),  # Model does not know about the subsequence
            y=y,
            location=location,
            p_unconditional=p_unconditional)

    @torch.no_grad()
    def sample_healthy(
            self,
            denoise_fn: torch.nn.Module,
            x_0: torch.Tensor,
            location: Optional[torch.Tensor] = None,
            first_n_timesteps: Optional[int] = None,
            classifier: Optional[LightningModule] = None,
            enc_class: Optional[torch.Tensor] = None,
            *args,
            **kwargs) -> torch.Tensor:
        
        # Healthy target (class 0)
        y = torch.full((x_0.shape[0], ), 0, device=x_0.device, dtype=torch.long)

        return self.encode_and_sample(
            input_tensor=x_0,
            denoise_fn=denoise_fn,
            location=location,
            classifier=classifier,
            conditional_sampling=True,
            first_n_timesteps=first_n_timesteps,
            y=y,
            enc_class=enc_class)

    @torch.no_grad()
    def encode_and_sample(
            self,
            input_tensor: torch.Tensor,
            denoise_fn: torch.nn.Module,
            location: Optional[torch.Tensor] = None,
            classifier: Optional[LightningModule] = None,
            conditional_sampling: bool = False,
            y: Optional[torch.Tensor] = None,
            enc_class: Optional[torch.Tensor] = None,
            first_n_timesteps: Optional[int] = None,
        ) -> torch.Tensor:
        r"""Function that embeds and image into the diffuse space using 
        the DDIM reverse sampling.

        Parameters
        ----------
        denoise_fn : torch.nn.Module
            The diffusion model with the pretrained weights trained to denoise an image
        classifier : torch.nn.Module, optional
            The classifier to estimate the class for the current input tensor
        location: torch.Tensor
            Patch location for positional embedding
        input_tensor : torch.Tensor
            The input tensor in question to be 'healthified'.
        conditional_sampling : bool
            If conditional sampling based on the ground truth image scale weak labels should be used.
            Make sure the diffusion model was trained with the same parameter
        first_n_timesteps : int, optional
            Number of timesteps that are carried out using both encoding and decoding.
        y : torch.Tensor, optional
            The class to be sampled. Requires either classifier or conditional_sampling

        Returns
        -------
        torch.Tensor
            The healthy version of the 'input_tensor' (could be the same as the input if the input is already healthy)
        """
        if classifier is not None or conditional_sampling:
            _error = "Provide a target class for conditional sampling or classifier-guided sampling"
            assert y is not None, _error

        if first_n_timesteps is not None:
            assert first_n_timesteps <= self._num_ts, f"first_n_timesteps ({first_n_timesteps}) must be smaller than the number of timesteps ({self._num_ts})"

        # Now we encode the anatomical information into the image by applying noise with the DDIM reverse sampling
        encoded_img = self.encode_image(
            x_0=input_tensor,
            denoise_fn=denoise_fn,
            y=enc_class,
            location=location,
            first_n_timesteps=first_n_timesteps)

        healthy = self.sample_image(
            denoise_fn=denoise_fn,
            y=y,
            conditional_sampling=conditional_sampling,
            classifier=classifier,
            noisy_img=encoded_img,
            location=location,
            first_n_timesteps=first_n_timesteps)
        return healthy

    def _sampling_fn(self,
                t: torch.Tensor,
                x_t: torch.Tensor,
                p_mean_var_x0: Optional[dh.MeanVarX0] = None,
                model_pred: Optional[torch.Tensor] = None,
                generator: Optional[torch.Generator] = None,
                noise: Optional[torch.Tensor] = None,
                eta: Optional[float] = None,
                *args,
                **kwargs) -> torch.Tensor:

        # Just for testing! p_mean_var should be specified!
        if p_mean_var_x0 is None:
            assert model_pred is not None, "Provide either p_mean_var_x0 or model_pred (preferred)"
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
                p_unconditional=0.)  # Fully conditional sampling

        # Note: Re-derive as x_0 is clipped to (-1,1), which results in a different eps than p_mean_var_x0.eps
        eps = self._predict_eps_from_x0(x_t=x_t, t=t, x_0=p_mean_var_x0.x_0)


        # Generate random noise
        if noise is None:
            noise = torch.empty_like(x_t).normal_(generator=generator)

        # No noise for t==0
        noise_mask = pum.expand_dims(torch.ne(t, 0), x_t.dim(), x_t.device)

        # Expand dim for safe multiplication along the batch dimension
        alpha_bar_prev_t = pum.expand_dims(self._alpha_bar_prev, x_t.dim(), x_t.device)[t]
        alpha_bar_t = pum.expand_dims(self._alpha_bar, x_t.dim(), x_t.device)[t]

        # For eta=1, we have the ddpm case with :math:`\sigma^2 = \tilde{\beta}`
        # This is a generalised form of Nichol Eq. 10 with eta (Eq. 15 Song)
        eta = eta or self.eta
        sigma = (
                eta
                * torch.sqrt((1 - alpha_bar_prev_t) / (1 - alpha_bar_t))
                * torch.sqrt(1 - alpha_bar_t / alpha_bar_prev_t)
        )

        # Eq. 12 of Song
        mean_pred = (
                torch.sqrt(alpha_bar_prev_t) * p_mean_var_x0.x_0 +
                torch.sqrt(1. - alpha_bar_prev_t - sigma ** 2) * eps
        )

        x_t_m1 = mean_pred + sigma * noise_mask * noise

        if torch.any(torch.isnan(x_t_m1)):
            crash_dir = dh.save_on_crash(self._logging_dir, experiment=self._experiment_name, **locals())
            raise RuntimeError(f"NaN values encountered. Artefacts saved to {crash_dir}")

        return x_t_m1

    @torch.no_grad()
    def encode_image(
        self,
        x_0: SAMPLE_TYPE,
        denoise_fn: torch.nn.Module,
        y: Optional[torch.Tensor] = None,
        location: Optional[torch.Tensor] = None,
        first_n_timesteps: Optional[int] = None) -> SAMPLE_TYPE:
        r"""This function allows the encoding of an image in the forward noising process or interpolation between images by
        embedding them in the latents.

        Parameters
        ----------
        x_0 : torch.Tensor
            The image to be encoded/interpolated
        denoise_fn : torch.nn.Module
            The model to predict the backwards step
        y : torch.Tensor, optional
            The desired class, which can already be used for conditional sampling
        first_n_timesteps : int, optional
            Whether the first n timesteps of the diffusion process should only be used to encode the image.
        """
        #assert self.eta == 0.0, "Requires deterministic process"

        timesteps_iterator = list(self.timestep_iterator)
        if first_n_timesteps is not None:
            assert first_n_timesteps <= len(timesteps_iterator), "first_n_timesteps must be smaller than the number of timesteps"
            timesteps_iterator = timesteps_iterator[:first_n_timesteps]

        x_t = x_0  # The start image
        b = x_0.shape[0] if isinstance(x_0, torch.Tensor) else x_0[0].shape[0]
        device = x_0.device if isinstance(x_0, torch.Tensor) else x_0[0].device

        # Iterate over the following time steps
        for i, _t in enumerate(tqdm.tqdm(
            timesteps_iterator,
            disable=not self._verbose,
            desc=f"{self.diffusion_type.value} Encoding",
            position=1,  # to not interfere with the one from pytorch lightning 
            leave=False)):

            t_t = torch.full((b,), _t, device=device)
            x_t = self._encode_single_step(
                denoise_fn=denoise_fn,
                x_t=x_t,
                t=t_t,
                y=y,
                location=location,
                total_iterations=len(timesteps_iterator),
                current_iteration=i)

        return x_t

    def _encode_single_step(
            self,
            denoise_fn: torch.nn.Module,
            x_t: SAMPLE_TYPE,
            t: torch.Tensor,
            y: Optional[torch.Tensor],
            location: Optional[torch.Tensor],
            total_iterations: Optional[int] = None,
            current_iteration: Optional[int] = None) -> SAMPLE_TYPE:
        r"""Encodes an input image for a single forward step.

        Parameters
        ----------
        denoise_fn : torch.nn.Module
            The model to predict the backwards step
        t : torch.Tensor
            The current time step of the iterative embedding
        x_t: torch.Tensor, tuple of two torch.Tensor or list of two torch.Tensor
            The embedded image at timestep t. In the case of EDICT, it is both x and y
            combined in a single variable, which is called here x_t for re-usability
            across different diffusion formulations.
        y : torch.Tensor, optional
            Desired class to be sampled. Requires conditional sampling to work
        location: torch.Tensor, optional
            The location of the patch of 3D images. Only required for positional embedding
        total_iterations: int, optional
            The total number of iterations. Only required for EDICT for alternate estimation
            of x_t and y_t
        current_iteration: int, optional
            The current iteration. Only required for EDICT for alternate estimation
            of x_t and y_t

        Returns
        -------
        torch.Tensor
            The new :math:`x_{t+1}`
        """

        assert isinstance(x_t, torch.Tensor), f"{self.diffusion_type} does not support tuple inputs for x_t. This is reserved for EDICT."

        p_mean_var_x0 = self.p_mean_variance(
            denoise_fn=denoise_fn,
            t=t,
            x_t=x_t,
            y=y,
            location=location,
            clip_denoised=True,
            p_unconditional=0.) # Fully conditional encoding

        return self._encoding_fn(
            model_output=p_mean_var_x0,
            x_t=x_t,
            t=t)

    def _encoding_fn(
            self,
            model_output: dh.MeanVarX0,
            x_t: torch.Tensor,
            t: torch.Tensor,
    ) -> torch.Tensor:
        r"""Encoding function for DDIM with deterministic characteristics following the ODE
        formulation. This is a reformulation of Eq. 13 of Song for the small limit
        approximation of \eps(x_t) \approx \eps(x_{t-1}). The formulation for the encoding
        step can be seen in Dhariwal Appendix F and is here reformulated for easier 
        calculation.

        Notes
        -----
        Eq. 12 of Song can be rewritten to highlight the ODE relationship as given in Appendix F of Dhariwal.
        However, one does not need to utilise that formalism but can reformulate it again back to the form of Eq. 12
        with the t+1 step

        .. math::
        x_{t+1} = \sqrt{\bar{\alpha}_{t+1}} * x_{0, pred} +
        \sqrt{1 - \bar{\alpha}_{t+1}} * \hat{\epsilon}_{\theta}^t
        """

        # Re-derive epsilon due to potential clipping
        eps_pred = self._predict_eps_from_x0(x_t=x_t, t=t, x_0=model_output.x_0)

        alpha_bar_next = pum.expand_dims(self._alpha_bar_next, x_t.dim(), x_t.device)[t]
        x_0_pred = pum.expand_dims(model_output.x_0, x_t.dim(), x_t.device)  # predicted x_0
        eps_pred = pum.expand_dims(eps_pred, x_t.dim(), x_t.device)
        return x_0_pred * torch.sqrt(alpha_bar_next) + torch.sqrt(1 - alpha_bar_next) * eps_pred
    
    def _class_guidance(
            self,
            mean_var_x0: dh.MeanVarX0,
            x_t: torch.Tensor,
            y: torch.Tensor,
            t: torch.Tensor,
            classifier: LightningModule,
            location: Optional[torch.Tensor] = None) -> dh.MeanVarX0:
        r"""Conditions the sampling process based on the score (conditional sampling) as it is a DDIM model.

        Notes
        -----
        Conditioning mechanism outlined in Section 4.2 of Dhariwal et al. 2021
        or in Algorithm 2 of Dhariwal et al. 2021

        Parameters
        ----------
        diffusion_type : dh.DiffusionType
            The type of the diffusion process (for sampling). One of [DDIM, DDPM]
        mean_var_x0 : dh.MeanVarX0
            The predicted mean, variance and x0
        x_t : torch.Tensor
            The input tensor to the backwards/denoising process at timestep(s) t
        y : torch.Tensor
            Desired class to be conditioned on
        t : torch.Tensor
            The timestep(s) for the respective batch
        classifier: torch.nn.Module
            The classifier to predict the label of the current input tensor
        location : torch.Tensor, optional
            The location information of the extracted patch. Required for 3D patch-based classifier

        Returns
        -------
        dh.MeanVarX0
            The new predictions conditioned on the input label y
        """
        # Gradient calculation
        gradient = self._gradient(
            x_t=x_t, 
            t=self.ddim_t_to_ddpm_t(t), # classifier does not know about the subsequence
            y=y, 
            classifier=classifier, 
            location=location)

        # Re-derive it from the prediction of x_0 as x_0 might be clipped to (-1,1)
        eps = self._predict_eps_from_x0(x_t=x_t, x_0=mean_var_x0.x_0, t=t)
        sqrt_one_minus_alpha_bar = pum.expand_dims(self._sqrt_one_minus_alpha_bar, eps.dim(), x_t.device)[t]

        # Conditioning Eq. 14 of Song
        eps = eps - sqrt_one_minus_alpha_bar * gradient * self._gradient_scale

        # Store the conditioning in mean_var_x0 again as the mean has been altered
        mean_var_x0.eps = eps

        # As we altered eps, we now need to get the new x_0 by re-using the x_0 prediction function
        mean_var_x0.x_0 = self._predict_x0_mean(x_t=x_t,
                                                t=t,
                                                model_pred_mean=eps)

        mean_var_x0.mean = self.q_posterior_mean_variance(x_t, t, mean_var_x0.x_0, eps=eps).mean

        return mean_var_x0