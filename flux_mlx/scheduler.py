"""FlowMatch Euler Discrete Scheduler for Flux2 Klein — pure numpy, no diffusers.

Replicates the exact timestep schedule from diffusers FlowMatchEulerDiscreteScheduler
with dynamic exponential shifting, matching Flux2KleinPipeline's compute_empirical_mu.
"""

import math
import numpy as np
import mlx.core as mx


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    """Compute dynamic shift parameter — matches diffusers exactly."""
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


def _time_shift(mu: float, sigma: float) -> float:
    """Apply exponential time shift: exp(mu) * sigma / (1 + (exp(mu)-1) * sigma)."""
    e_mu = math.exp(mu)
    return e_mu * sigma / (1.0 + (e_mu - 1.0) * sigma)


def get_sigmas(num_steps: int, image_seq_len: int) -> tuple[np.ndarray, float]:
    """Compute sigma schedule matching diffusers FlowMatchEulerDiscreteScheduler.

    Replicates the exact logic from Flux2KleinPipeline.__call__:
      sigmas_input = np.linspace(1.0, 1/num_steps, num_steps)
      then applies dynamic exponential shifting with mu

    Returns:
        sigmas: array [num_steps + 1] from ~1.0 to 0.0
        mu: the computed shift parameter
    """
    mu = compute_empirical_mu(image_seq_len, num_steps)

    # Input sigmas (what the pipeline passes to retrieve_timesteps)
    sigmas_input = np.linspace(1.0, 1.0 / num_steps, num_steps)

    # Apply time shift to each sigma
    shifted = np.array([_time_shift(mu, s) for s in sigmas_input])

    # Append terminal 0
    sigmas = np.append(shifted, 0.0)

    return sigmas, mu


def step(noise_pred: mx.array, sigma: float, sigma_next: float,
         latents: mx.array) -> mx.array:
    """Single Euler step: latents += (sigma_next - sigma) * noise_pred."""
    dt = mx.array(sigma_next - sigma).astype(latents.dtype)
    return latents + dt * noise_pred
