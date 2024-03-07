from patchldmseg.model.diffusion.diffusion_helper import MeanVarX0
import patchldmseg.utils.misc as pum
from pytorch_lightning import LightningModule
import torch
import patchldmseg.model.diffusion.diffusion_helper as dh
from patchldmseg.model.diffusion.ddpm import DDPMDiffusion
from patchldmseg.model.diffusion.ddim import DDIMDiffusion
from patchldmseg.utils.constants import SAMPLE_TYPE

from torch.nn.modules import Module


from typing import List, Literal, Optional, Tuple, Union


class EDICTDiffusion(DDIMDiffusion):
    r"""Extends the base class diffusion process by additional EDICT methods. This includes mainly
    the exact reconstruction of an input image. This method implements Eq. 14 of Wallace et al. in 
    the reverse direction with first estimating y_t^{inter}, using this estimate to calculate x_t^{inter}
    and then either calculate y_t or x_t depending on the input that is alternated. The method makes use 
    of the linearization assumption of DDIM i.e. 

    .. math::
        \epsilon(x_t, t) \approx \epsilon(x_{t-1}, t)

    Notes
    -----
    Implementation by Wallace et al. https://github.com/salesforce/EDICT
    https://arxiv.org/abs/2211.12446 
    """

    def __init__(
            self,
            loss_type: dh.LossType,
            mean_type: dh.MeanType,
            var_type: dh.VarianceType,
            eta: float,
            max_in_val: float,
            gradient_scale: float = 1,
            num_timesteps: int = 4000,
            noise_schedule: Literal['linear', 'cosine'] = 'linear',
            verbose: bool = False,
            logging_dir: str | None = None,
            experiment_name: str | None = None,
            subsequence_length: int | None = None,
            p_unconditional: float = 0.,
            mixing_weight: float = 0.93):
        super().__init__(
            loss_type=loss_type, 
            mean_type=mean_type, 
            var_type=var_type, 
            eta=eta, 
            max_in_val=max_in_val, 
            gradient_scale=gradient_scale, 
            num_timesteps=num_timesteps, 
            noise_schedule=noise_schedule, 
            verbose=verbose, 
            logging_dir=logging_dir, 
            experiment_name=experiment_name, 
            subsequence_length=subsequence_length,
            p_unconditional=p_unconditional)

        self._diffusion_type = dh.DiffusionType.EDICT
        self.mixing_weight = mixing_weight

    @torch.no_grad()
    def encode_image(
        self,
        x_0: SAMPLE_TYPE,
        denoise_fn: torch.nn.Module,
        y: Optional[torch.Tensor] = None,
        location: Optional[torch.Tensor] = None,
        first_n_timesteps: Optional[int] = None) -> SAMPLE_TYPE:

        latent_pair = super().encode_image(
            x_0=x_0,
            denoise_fn=denoise_fn,
            y=y,
            location=location,
            first_n_timesteps=first_n_timesteps)

        assert len(latent_pair) == 2, "Can only process a latent pair"
        return tuple(latent_pair)  # type: ignore

    def _encode_single_step(
            self,
            denoise_fn: torch.nn.Module,
            x_t: SAMPLE_TYPE,
            t: torch.Tensor,
            y: Optional[torch.Tensor],
            location: Optional[torch.Tensor],
            total_iterations: Optional[int] = None,
            current_iteration: Optional[int] = None,
    ) -> SAMPLE_TYPE:
        if isinstance(x_t, (tuple, list)):
            latent_pair = list(x_t)
        elif isinstance(x_t, torch.Tensor):
            latent_pair = [x_t.clone(), x_t.clone()]
        else:
            raise AttributeError(f"Unknown input type {type(x_t)}")

        assert total_iterations is not None
        assert current_iteration is not None

        # Mixing for intermediate - Eq. 14 of Wallace et al. 
        new_latents = [l.clone() for l in latent_pair]
        new_latents[1] = (new_latents[1].clone() - (1-self.mixing_weight)*new_latents[0].clone()) / self.mixing_weight # y_t^inter
        new_latents[0] = (new_latents[0].clone() - (1-self.mixing_weight)*new_latents[1].clone()) / self.mixing_weight # x_t^inter
        latent_pair = new_latents

        # In-place - maybe not memory safe
        #latent_pair[1] = (latent_pair[1] - (1 - self.mixing_weight) * latent_pair[0])/self.mixing_weight  # y_t^inter
        #latent_pair[0] = (latent_pair[0] - (1 - self.mixing_weight) * latent_pair[1])/self.mixing_weight  # x_t^inter

        for latent_i in range(2):
            latent_i, latent_j = self.alternate_steps(
                latent_i=latent_i,
                current_iter=current_iteration,
                total_steps=total_iterations)

            # See Eq. 11 of Wallace et al.
            model_input = latent_pair[latent_j]  # either x_{t-1}/x_t^{iter} or y_t
            fixed = latent_pair[latent_i]  # either y_{t-1}/y_t^{iter} or x_{t-1}/x_t^{iter}

            # Prediction of the model output based on y_t 
            model_output_cond = self.p_mean_variance(
                x_t=model_input,
                t=t,
                y=y,
                location=location,
                denoise_fn=denoise_fn,
                p_unconditional=0.
            )

            model_output = self.condition(
                denoise_fn=denoise_fn,
                classifier=None,
                x_t=model_input,
                y=y,
                location=location,
                t=t,
                mean_var_x0=model_output_cond
            )

            latent_pair[latent_i] = self._encode_fn(
                model_output=model_output,
                x_or_y=fixed,
                t=t)

        return latent_pair

    def _encode_fn(self,
                   model_output: dh.MeanVarX0,
                   x_or_y: torch.Tensor,
                   t: torch.Tensor) -> torch.Tensor:
        r"""Encoding with the EDICT method. This method follows Eq. 11 of Wallace et al.
        and predicts either x_t or y_t based on the input x_or_y.

        Parameters
        ----------
        model_output : dh.MeanVarX0
            The predicted mean, variance and x0 of the model with either x_{t-1} or y_t as input
        x_or_y: torch.Tensor
            The respective other tensor not being used for model predictions. Either y_{t-1} or x_{t-1}
        t : torch.Tensor
            The current timestep

        Returns
        -------
        torch.Tensor
            Either y_{t+1} or x_{t+1} depending on the input y_t or x_t  (could also define it as t and t-1)

        """
        alpha_bar_t = pum.expand_dims(self._alpha_bar, x_or_y.dim(), x_or_y.device)[t]
        alpha_bar_prev_t = pum.expand_dims(self._alpha_bar_prev, x_or_y.dim(), x_or_y.device)[t]

        a_t, b_t = self._get_at_bt(alpha_bar_t=alpha_bar_t, 
                                   alpha_bar_prev_t=alpha_bar_prev_t)

        x_or_y_p1 = (x_or_y - b_t * model_output.eps)/ a_t
        return x_or_y_p1

    @staticmethod
    def _get_at_bt(alpha_bar_t: torch.Tensor, 
                   alpha_bar_prev_t: torch.Tensor
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Calculate a_t and b_t of Eq. 3 and 4 of Wallace et al.
        
        Notes
        -----

        .. math::
            a_t = \sqrt{\frac{\alpha_{t-1}}{\alpha_t}}
            b_t = \sqrt{1 - \alpha_{t-1}} - \sqrt{\frac{\alpha_{t-1}(1-\alpha_t)}{\alpha_t}}
                = \sqrt{1 - \alpha_{t-1}} - a_t \sqrt{1 - \alpha_t}

        """
        a_t = torch.sqrt(alpha_bar_prev_t/alpha_bar_t)
        b_t_first_term = torch.sqrt(1. - alpha_bar_t) * a_t
        b_t_second_term = torch.sqrt(1. - alpha_bar_prev_t)
        #  Swapped the order because of the negative sign
        b_t = b_t_second_term - b_t_first_term
        return a_t, b_t

    @staticmethod
    def alternate_steps(
        latent_i: int, 
        total_steps: int, 
        current_iter: int,
        leapfrog_steps: bool = True) -> Tuple[int, int]:
        r""" Alternates the order in which the x and y series are computed.
        
        Notes
        -----
        Taken from Wallace et al.
        """
        if leapfrog_steps:
            # what i would be from going other way ?
            orig_i = total_steps - (current_iter+1)
            offset = (orig_i+1) % 2
            latent_i = (latent_i + offset) % 2
        else:
            # Do 1 the 0
            latent_i = (latent_i + 1) % 2

        latent_j = (latent_i + 1) % 2
        return latent_i, latent_j

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
            eta: Optional[float] = None,
            *args,
            **kwargs) -> torch.Tensor:

            # Does return a tuple of tensors instead of the docstring tensor
            # as we are hijacking it.
            # Here we guarantee that we really only return a single tensor
            latent_pair = super().sample_image(
                denoise_fn=denoise_fn,
                num_images=num_images,
                inputs=inputs,
                noisy_img=noisy_img,
                y=y,
                location=location,
                first_n_timesteps=first_n_timesteps,
                classifier=classifier,
                conditional_sampling=conditional_sampling,
                eta=eta)
            
            assert len(latent_pair) == 2, "Can only process a latent pair"
            return latent_pair[0]  # type: ignore

    def _sample_single_step(
            self,
            x_t: SAMPLE_TYPE,
            t: torch.Tensor,
            denoise_fn: Module,
            y: Optional[torch.Tensor] = None,
            location: Optional[torch.Tensor] = None,
            classifier: Optional[LightningModule] = None,
            eta: Optional[float] = None,
            num_images: Optional[int] = None,
            rng: Optional[torch.Generator] = None,
            total_iterations: Optional[int] = None,
            current_iteration: Optional[int] = None) -> SAMPLE_TYPE:

        assert total_iterations is not None, (
            "total_iterations required for estimating the alternating steps between x and y estimation")
        assert current_iteration is not None, (
            "current_iteration required for estimating the alternating steps between x and y estimation"
        )

        if isinstance(x_t, (tuple, list)):
            latent_pair = list(x_t)
        elif isinstance(x_t, torch.Tensor):
            latent_pair = [x_t.clone(), x_t.clone()]
        else:
            raise RuntimeError(f"x_t must be either a tensor or a tuple of tensors. Got {type(x_t)}")

        for latent_i in range(2):
            latent_i, latent_j = self.alternate_steps(
                latent_i=latent_i,
                current_iter=current_iteration,
                total_steps=total_iterations)

            # See Eq. 11 of Wallace et al.
            model_input = latent_pair[latent_j]  # either x_{t-1}/x_t^{iter} or y_t
            fixed = latent_pair[latent_i]  # either y_{t-1}/y_t^{iter} or x_{t-1}/x_t^{iter}

            # Prediction of the model output based on y_t 
            model_output_cond = self.p_mean_variance(
                x_t=model_input,
                t=t,
                y=y,
                location=location,
                denoise_fn=denoise_fn,
                p_unconditional=0.
            )

            model_output = self.condition(
                denoise_fn=denoise_fn,
                classifier=classifier,
                x_t=model_input,
                y=y,
                location=location,
                t=t,
                mean_var_x0=model_output_cond
            )

            latent_pair[latent_i] = self._sampling_fn(
                model_output=model_output,
                x_or_y=fixed,
                t=t)
            
        # Mixing of the latents - Eq. 14 of Wallace et al.
        new_latents = [l.clone() for l in latent_pair]
        new_latents[0] = (self.mixing_weight*new_latents[0] + (1-self.mixing_weight)*new_latents[1]).clone() 
        new_latents[1] = ((1-self.mixing_weight)*new_latents[0] + (self.mixing_weight)*new_latents[1]).clone() 
        latent_pair = new_latents

        # In-place - maybe not memory safe
        # latent_pair[0] = self.mixing_weight * latent_pair[0] + (1 - self.mixing_weight) * latent_pair[1]
        # latent_pair[1] = self.mixing_weight * latent_pair[1] + (1 - self.mixing_weight) * latent_pair[0]

        return latent_pair


    def _sampling_fn(
            self,
            model_output: dh.MeanVarX0,
            x_or_y: torch.Tensor,
            t: torch.Tensor) -> torch.Tensor:
        r"""Decoding (reversing the noise) with the EDICT method. This method follows Eq. 10 of Wallace et al.
        and predicts either x_{t-1} or y_{t-1} based on the input x_or_y.

        Parameters
        ----------
        model_output : dh.MeanVarX0
            The predicted mean, variance and x0 of the model with either x_{t-1} or y_t as input
        x_or_y: torch.Tensor
            The respective other tensor not being used for model predictions. Either y_t or x_t
        t : torch.Tensor
            The current timestep

        Returns
        -------
        torch.Tensor
            Either y_t or x_t depending on the input x_or_y.

        """
        alpha_bar_t = pum.expand_dims(self._alpha_bar, x_or_y.dim(), x_or_y.device)[t]
        alpha_bar_prev_t = pum.expand_dims(self._alpha_bar_prev, x_or_y.dim(), x_or_y.device)[t]

        a_t, b_t = self._get_at_bt(alpha_bar_t=alpha_bar_t, 
                                   alpha_bar_prev_t=alpha_bar_prev_t)

        x_or_y_m1 = a_t * x_or_y + b_t * model_output.eps
        return x_or_y_m1
        
    def _class_guidance(self, *args, **kwargs) -> MeanVarX0:
        raise NotImplementedError("Class guidance not implemented for EDICT")
    
    def _check_sampling_return_type(self, x_t: SAMPLE_TYPE):
        _class_name = self.__class__.__name__
        _error_msg = (f"Wrong return type. {_class_name} expects list of torch.Tensor as return type of sampling. Got {type(x_t)}.")
        assert isinstance(x_t, list), _error_msg

    @classmethod
    def from_ddpm(cls,
                  ddpm_instance: DDPMDiffusion,
                  eta: float,
                  subsequence_length: Optional[int] = None,
                  mixing_weight: float = 0.93):
        attribs = {attrib: getattr(ddpm_instance, attrib) for attrib in (
            'max_in_val', 'loss_type', 'mean_type', 'var_type', 'gradient_scale', 'num_timesteps','noise_schedule', 'verbose', 'logging_dir', 'experiment_name', 'p_unconditional'
            )}
        # Add new attributes
        attribs.update({
            'subsequence_length': subsequence_length,
            'eta': eta,
            'mixing_weight': mixing_weight})
        return cls(**attribs)