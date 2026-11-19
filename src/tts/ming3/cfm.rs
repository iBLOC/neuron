//! Flow-matching sampler for Ming-3.0-tts (`fm/CFM.py::CFM.sample` +
//! `fm/flowloss.py::FlowLoss.sample`).
//!
//! Deterministic-path details that are part of the reference behaviour:
//! - the initial noise and every step's SDE noise are drawn **up front**,
//!   in this order and shape (so a fixed RNG stream matches the reference
//!   sampler); tests inject them instead of relying on the global RNG;
//! - the SDE term is `sigma * sqrt(temperature) * sqrt(|dt|) * noise` and
//!   vanishes at `temperature = 0` (the API default);
//! - timesteps use the EPSS (Empirically Pruned Step Sampling) tables for
//!   low NFE, then the sway correction `t + c*(cos(π/2 t) - 1 + t)` with
//!   `c = -1.0`.

use super::dit::MingDiT;
use burn::prelude::Backend;
use burn::tensor::Tensor;

/// Sampler knobs with the upstream defaults (`MingTTS.synthesize`).
#[derive(Debug, Clone, PartialEq)]
pub struct CfmOptions {
    /// ODE steps (the `steps` default of `FlowLoss.sample`).
    pub steps: usize,
    /// Classifier-free guidance scale.
    pub cfg_scale: f32,
    /// SDE noise scale.
    pub sigma: f32,
    /// SDE temperature — 0 makes the sampler fully deterministic given the
    /// initial noise.
    pub temperature: f32,
    /// Sway sampling coefficient.
    pub sway_sampling_coef: f32,
}

impl Default for CfmOptions {
    fn default() -> Self {
        Self {
            steps: 10,
            cfg_scale: 2.0,
            sigma: 0.25,
            temperature: 0.0,
            sway_sampling_coef: -1.0,
        }
    }
}

/// The EPSS predefined timesteps (`get_epss_timesteps`), scaled by 1/32.
///
/// Falls back to a uniform `linspace(0, 1, steps + 1)` for step counts
/// without a table — same as upstream.
pub fn epss_timesteps(steps: usize) -> Vec<f32> {
    let tables: [(usize, &[f32]); 6] = [
        (5, &[0.0, 2.0, 4.0, 8.0, 16.0, 32.0]),
        (6, &[0.0, 2.0, 4.0, 6.0, 8.0, 16.0, 32.0]),
        (7, &[0.0, 2.0, 4.0, 6.0, 8.0, 16.0, 24.0, 32.0]),
        (
            10,
            &[0.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0],
        ),
        (
            12,
            &[
                0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 24.0, 28.0, 32.0,
            ],
        ),
        (
            16,
            &[
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 24.0,
                28.0, 32.0,
            ],
        ),
    ];
    let dt = 1.0 / 32.0;
    if let Some((_, table)) = tables.iter().find(|(n, _)| *n == steps) {
        return table.iter().map(|v| v * dt).collect();
    }
    (0..=steps)
        .map(|i| i as f32 / steps as f32)
        .collect()
}

/// One Euler flow-matching solve from `noise` to a latent patch.
///
/// `cond` is the LLM condition `z` `(B, 1, cond_dim)`; `latent_history` is
/// `(B, history, latent)`; `noise` is `(B, patch, latent)`. `sde_noise` is
/// `(steps, B, patch, latent)` — pass `None` to draw standard normals.
#[allow(clippy::too_many_arguments)]
pub fn sample<B: Backend>(
    dit: &MingDiT<B>,
    cond: Tensor<B, 3>,
    latent_history: Tensor<B, 3>,
    noise: Tensor<B, 3>,
    sde_noise: Option<Tensor<B, 4>>,
    opts: &CfmOptions,
) -> Tensor<B, 3> {
    let device = noise.device();
    let steps = opts.steps;
    let [b, patch, latent] = noise.dims();

    let sde = sde_noise.unwrap_or_else(|| {
        Tensor::random([steps, b, patch, latent], Default::default(), &device)
    });

    let timesteps: Vec<f32> = epss_timesteps(steps)
        .into_iter()
        .map(|t| t + opts.sway_sampling_coef * ((std::f32::consts::PI / 2.0 * t).cos() - 1.0 + t))
        .collect();

    let mut y = noise;
    for step in 0..steps {
        let dt = timesteps[step + 1] - timesteps[step];
        // CFG: both halves computed as one batch of 2B.
        let pred_cfg = dit.forward_with_cfg(
            y.clone(),
            timesteps[step],
            cond.clone(),
            latent_history.clone(),
        );
        let pred = pred_cfg.clone().slice([0..b, 0..patch, 0..latent]);
        let null = pred_cfg.slice([b..2 * b, 0..patch, 0..latent]);
        // v = pred + (pred - null) * cfg_scale
        let v = (pred.clone().sub(null)).mul_scalar(opts.cfg_scale).add(pred);
        y = y.add(v.mul_scalar(dt));

        // SDE term — vanishes at temperature 0 (the default).
        let scale = opts.sigma * opts.temperature.sqrt() * dt.abs().sqrt();
        if scale != 0.0 {
            let step_noise = sde
                .clone()
                .slice([step..step + 1, 0..b, 0..patch, 0..latent])
                .reshape([b, patch, latent]);
            y = y.add(step_noise.mul_scalar(scale));
        }
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;

    type TestBackend = NdArray;

    fn device() -> burn::backend::ndarray::NdArrayDevice {
        Default::default()
    }

    /// The EPSS table for the default 10 steps matches the reference list.
    #[test]
    fn epss_table_matches_upstream() {
        let t = epss_timesteps(10);
        let expected = [0.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0]
            .iter()
            .map(|v| v / 32.0)
            .collect::<Vec<_>>();
        assert_eq!(t, expected);
        // Unknown step counts fall back to a uniform grid.
        let t9 = epss_timesteps(9);
        assert_eq!(t9.len(), 10);
        assert!((t9[3] - 3.0 / 9.0).abs() < 1e-6);
    }

    /// With temperature 0 the sampler is deterministic given the initial
    /// noise: two runs with the same injected noise produce identical
    /// latents.
    #[test]
    fn temperature_zero_is_deterministic_given_noise() {
        let device = device();
        let latent = 8;
        let dit = MingDiT::init(latent, 1, 1, 16, 2, 32, 4, 4.0, &device);
        let opts = CfmOptions::default();
        let noise = Tensor::<TestBackend, 3>::from_floats(
            [[[0.1f32, -0.2, 0.3, 0.05, -0.1, 0.2, -0.3, 0.15]]],
            &device,
        );
        let cond = Tensor::<TestBackend, 3>::from_floats(
            [[[
                1.0f32, 0.5, -0.5, 1.0, 0.25, -1.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0,
            ]]],
            &device,
        );
        let history = Tensor::<TestBackend, 3>::zeros([1, 1, latent], &device);

        let out1 = sample(&dit, cond.clone(), history.clone(), noise.clone(), None, &opts);
        let out2 = sample(&dit, cond.clone(), history, noise, None, &opts);
        assert_eq!(out1.dims(), [1, 1, latent]);
        let a = out1.to_data().to_vec::<f32>().expect("read run 1");
        let b = out2.to_data().to_vec::<f32>().expect("read run 2");
        assert_eq!(a, b);
    }

    /// A nonzero temperature must change the trajectory (the SDE term is
    /// no longer vanishing).
    #[test]
    fn nonzero_temperature_changes_the_trajectory() {
        let device = device();
        let latent = 8;
        let dit = MingDiT::init(latent, 1, 1, 16, 2, 32, 4, 4.0, &device);
        let noise = Tensor::<TestBackend, 3>::from_floats(
            [[[0.1f32, -0.2, 0.3, 0.05, -0.1, 0.2, -0.3, 0.15]]],
            &device,
        );
        let cond = Tensor::<TestBackend, 3>::zeros([1, 1, 16], &device);
        let history = Tensor::<TestBackend, 3>::zeros([1, 1, latent], &device);
        // Fixed SDE noise so the only difference is the temperature scale.
        let sde = Tensor::<TestBackend, 2>::from_floats([[0.5f32; 8]; 10], &device)
            .reshape([10, 1, 1, 8]);
        let mut opts0 = CfmOptions::default();
        opts0.temperature = 0.0;
        let mut opts1 = CfmOptions::default();
        opts1.temperature = 4.0;
        let a = sample(
            &dit,
            cond.clone(),
            history.clone(),
            noise.clone(),
            Some(sde.clone()),
            &opts0,
        )
        .to_data()
        .to_vec::<f32>()
        .expect("read a");
        let b = sample(&dit, cond, history, noise, Some(sde), &opts1)
            .to_data()
            .to_vec::<f32>()
            .expect("read b");
        assert_ne!(a, b);
    }
}
