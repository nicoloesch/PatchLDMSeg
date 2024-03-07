import torch

import patchldmseg.model.diffusion.diffusion_helper as dh
from patchldmseg.utils.misc import non_batch_mean


def loss_vlb(
        variance_type: dh.VarianceType,
        t: torch.Tensor,
        p_mean_var_x0: dh.MeanVarX0,
        q_mean_var_x0: dh.MeanVarX0,
        max_in_val: float):
    r""" Calculate the VLB term of the hybrid loss function of 2021 - Nichol (Eq. 16).
    This arises from learning the variance and as such, the term L_T of 2020 - Ho (Eq. 5)
    comes into play. It is simply the KL-divergence between the prediction and the real probability.
    Additionally, the L_0-term in the case of t=0 is also calculated, which is the decoder NLL

    Notes
    -----
    
    .. math::
        L_T = \text{KL}(q(x_{t-1} \vert x_t,x_0) \Vert p_\theta(x_{t-1} \vert x_t))

        L_0 = \log(p_\theta(x_0 \vert x_1))

    Parameters
    ----------
    variance_type : dh. VarianceType
        The respective variance type of the model. Just required for safety check
    p_mean_var_x0 : dh.MeanVarX0
        The predicted mean and variance of the probability distribution
    q_mean_var_x0 : dh.MeanVarX0
        The actual posterior mean and variance of the probability distribution
    t : torch.Tensor
        The current timestep(s) t
    max_in_val : float
        The range of input values typically from [0, 255] for RGB images but different for medical images.
    Returns
    -------
    torch.Tensor
        The KL-divergence for term L_{vlb} of the hybrid loss, which is L_T in the general loss due to the learning
        of the variance \Sigma of shape (B, ...)
    """
    assert variance_type in [dh.VarianceType.LEARNED, dh.VarianceType.LEARNED_RANGE]
    kld = kl_divergence(mean_var_1=q_mean_var_x0,
                        mean_var_2=p_mean_var_x0,
                        is_log_var=True)

    # Calculates  \log(p_\theta(x_0 \vert x_1)) with x_0 given as the real ground-truth x_0 i.e. the input image
    decoder_nll = torch.mul(-1, gaussian_log_likelihood(x_0=q_mean_var_x0.x_0,
                                                        mean_var_x0=p_mean_var_x0,
                                                        is_log_var=True,
                                                        max_in_val=max_in_val))

    # Normalise from nats to bits
    kld = non_batch_mean(kld) / torch.log(torch.as_tensor(2.))
    decoder_nll = non_batch_mean(decoder_nll) / torch.log(torch.as_tensor(2.))

    # Return the decoder NLL in the first timestep \log(p_\theta(x_0 \vert x_1)) and the KLD afterwards
    # Mean over the batch dim
    return torch.where(torch.eq(t, 0), decoder_nll, kld)


def kl_divergence(mean_var_1: dh.MeanVarX0,
                  mean_var_2: dh.MeanVarX0,
                  is_log_var: bool = True) -> torch.Tensor:
    r"""Calculate the KL divergence between two (univariate) Gaussians according to this definition on
    https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Examples

    Notes
    -----
    The KL-divergence is calculated by

    .. math::
        D_\text{KL} (p \Vert q) = \log \frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2 \sigma_2^2} -
        \frac{1}{2}

    Variance = :math:`\sigma^2`!

    Parameters
    ----------
    mean_var_1 : dh.MeanVarX0
        The container for all calculated values of the first distribution
    mean_var_2 : dh.MeanVarX0
        The container for all calculated values of the second distribution

    is_log_var : bool
        Whether to use the log-variances for calculation to prevent numerical instabilities.

    Returns
    -------
    torch.Tensor
        The KL-divergence between both distributions in NATS as the implementation uses the natural log
    """

    mean_1 = mean_var_1.mean
    mean_2 = mean_var_2.mean

    if is_log_var:
        log_var_1 = mean_var_1.log_var
        log_var_2 = mean_var_2.log_var

        # Multiplication of 0.5 results in sqrt in logarithm
        return 0.5 * (
                log_var_2 - log_var_1 + torch.exp(log_var_1 - log_var_2) +
                ((mean_1 - mean_2) ** 2) * torch.exp(-log_var_2) - 1.)

    else:
        var_1 = mean_var_1.var
        var_2 = mean_var_2.var
        message = "The variances need to be bigger than 0, as the logarithm is only defined in (0, inf)"
        assert torch.all(torch.gt(var_1, 0.0)) and torch.all(torch.gt(var_2, 0.0)), message

        return 0.5 * (torch.log(var_2 / var_1) + ((var_1 + (mean_1 - mean_2) ** 2) / var_2) - 1.)


def gaussian_log_likelihood(x_0: torch.Tensor,
                            mean_var_x0: dh.MeanVarX0,
                            is_log_var: bool,
                            max_in_val: float) -> torch.Tensor:
    r"""Compute the log-likelihood of a Gaussian distribution. This step is performed in the backwards process
    from t=1 to t=0 i.e. the decoder.

    This has been dropped by Ho et al. for simplicity but was re-introduced by Dhariwal et al. to improve
    the log-likelihood.

    Notes
    -----
    The implementation follows the definition of the decoder in Eq. 13 of Ho et al.

    Parameters
    ----------
    x_0 : torch.Tensor
        The input tensor scaled to [-1, 1] against which to compare to
    mean_var_x0 : dh.MeanVarX0
        The predicted mean and variance (x0 not required) of the model, which will be compared to the ground-truth x_t
    is_log_var : bool
        Whether the variance is provided in log-scale. Better for numerical instabilities and exploding gradients
    max_in_val : float
        The range of input values typically from [0, 255] for RGB images but different for medical images.
    """
    increment = 1. / max_in_val

    mean = mean_var_x0.mean

    # Scaling factor for integral borders
    if is_log_var:
        # 0.5 due to sqrt = ^0.5
        inv_std = torch.exp(torch.mul(mean_var_x0.var, -0.5))
    else:
        inv_std = 1. / torch.sqrt(mean_var_x0.var)

    centered_x = x_0 - mean

    delta_plus = inv_std * (centered_x + increment)
    delta_minus = inv_std * (centered_x - increment)

    # Calculate CDF
    cdf_minf_deltap = approximate_standard_normal_cdf(delta_plus).clamp(min=1e-12)  # The CDF from -\infty to \delta_+
    cdf_minf_deltam = approximate_standard_normal_cdf(delta_minus)  # The CDF from -\infty to \delta_-
    cdf_deltam_pinf = (1. - cdf_minf_deltam).clamp(min=1e-12)  # The CDF from \delta_- to \infty
    cdf_deltam_delta_p = (cdf_minf_deltap - cdf_minf_deltam).clamp(min=1e-12)  # The CDF from \delta_- to \delta_+

    # Calculate the conditions for x <= -1, x >= 1 and the one in between
    # Logarithm as I am calculating \log(p_\theta(x_0 \vert x_1))
    log_probs = torch.where(torch.le(x_0, -1.), torch.log(cdf_minf_deltap),
                            torch.where(torch.ge(x_0, 1.), torch.log(cdf_deltam_pinf), torch.log(cdf_deltam_delta_p)))

    return log_probs


def approximate_standard_normal_cdf(x: torch.Tensor):
    r"""Approximates the standard normal cumulative density function by using the approximation of
    Page - Approximations to the Cumulative Normal Functions and its inverse for use on a pocket calculator

    Notes
    -----
    .. math::
        \phi(z) \approx 0.5\left(1 + \tanh \left( \sqrt{\frac{2}{\pi}} (z + 0.044715z^3)\right)\right)

    The function calculates the integral from :math:`-\infty` to the upper bound specified by x

    Parameters
    ----------
    x : torch.Tensor
        The upper bound of the CDF

    Returns
    -------
    torch.Tensor
        The CDF calculated as :math:`\int_{-\infty}^x \text{CDF}`
    """

    return 0.5 * (
            1.0 + torch.tanh(
                torch.sqrt(torch.as_tensor(2.0 / torch.pi)) * (x + torch.as_tensor(0.044715) * torch.pow(x, 3))
                )
            )
