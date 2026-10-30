//! MOSS-Image sampling schedule — a faithful port of
//! `MOSSImageScheduler` (a `FlowMatchEulerDiscreteScheduler`
//! subclass) from the reference `pipeline_moss_image.py`.
//!
//! Two layers:
//!
//! 1. **Sigma grid** (`set_timesteps`): the inference sigmas are
//!    produced by the "V2" shifted grid
//!    `sigma = shift · v / (1 + (shift−1) · v)` over
//!    `v = linspace(1, shift/(total+shift−1), n)`, then each sigma is
//!    **snapped to the nearest point of the training grid** (the same
//!    grid with `training_shift`, 1000 points, ascending) and
//!    consecutive duplicates collapse. A terminal `sigma = 0` is
//!    appended by the underlying FlowMatchEuler scheduler.
//! 2. **Step** (flow-matching Euler, fp32):
//!    `x ← x + (sigma_next − sigma) · v`.
//!
//! The DiT receives `timestep = sigma · num_train_timesteps`.

/// The computed schedule: `sigmas.len() == timesteps.len() + 1`
/// (terminal 0.0 appended to sigmas only, matching FlowMatchEuler's
/// `set_timesteps` when explicit sigmas are provided).
#[derive(Debug, Clone)]
pub struct MossSchedule {
    /// Length-N sigma list the steps iterate over (descending,
    /// already snapped to the training grid).
    pub timesteps: Vec<f64>,
    /// Length-N+1: `sigmas[i]` pairs with `timesteps[i]`, and
    /// `sigmas[N] == 0.0` is the terminal target.
    pub sigmas: Vec<f64>,
}

/// Build the shifted grid `shift · v / (1 + (shift−1) · v)` over
/// `v = linspace(1, lower, count)` where `lower = shift/(total+shift−1)`.
fn grid(count: usize, shift: f64, total: usize) -> Vec<f64> {
    let lower = shift / (total as f64 + shift - 1.0);
    (0..count)
        .map(|i| {
            let t = if count == 1 {
                1.0
            } else {
                i as f64 / (count - 1) as f64
            };
            let v = 1.0 + (lower - 1.0) * t;
            shift * v / (1.0 + (shift - 1.0) * v)
        })
        .collect()
}

/// MOSS-Image scheduler configuration (defaults from the reference:
/// `num_train_timesteps=1000`, `shift=5.0`, `training_shift=1.0`).
#[derive(Debug, Clone, Copy)]
pub struct MossSchedulerConfig {
    pub num_train_timesteps: usize,
    pub shift: f64,
    pub training_shift: f64,
}

impl Default for MossSchedulerConfig {
    fn default() -> Self {
        Self {
            num_train_timesteps: 1000,
            shift: 5.0,
            training_shift: 1.0,
        }
    }
}

/// Compute the snapped sigma schedule for `num_inference_steps`
/// sampling steps.
pub fn set_timesteps(num_inference_steps: usize, config: &MossSchedulerConfig) -> MossSchedule {
    assert!(num_inference_steps >= 1, "steps must be >= 1");
    assert!(config.shift > 0.0 && config.training_shift > 0.0);
    let total = config.num_train_timesteps;

    // Inference sigmas, descending from 1.
    let sigmas = grid(num_inference_steps, config.shift, total);
    // Training grid, ascending after the flip.
    let training = {
        let mut t = grid(total, config.training_shift, total);
        t.reverse();
        t
    };

    // Snap each inference sigma to the nearest training-grid point.
    // `searchsorted` (side=left) on the ascending grid: the first
    // index whose value is >= sigma; clamp into range.
    let mut snapped: Vec<f64> = Vec::with_capacity(num_inference_steps);
    for &sigma in &sigmas {
        let upper = training.partition_point(|&t| t < sigma).min(total - 1);
        let lower = upper.saturating_sub(1);
        let nearest = if (sigma - training[lower]).abs()
            <= (training[upper] - sigma).abs()
        {
            training[lower]
        } else {
            training[upper]
        };
        // unique_consecutive
        if snapped.last() != Some(&nearest) {
            snapped.push(nearest);
        }
    }

    let timesteps: Vec<f64> = snapped
        .iter()
        .map(|&s| s * config.num_train_timesteps as f64)
        .collect();
    // FlowMatchEuler appends the terminal 0 to sigmas only.
    let mut all_sigmas = snapped;
    all_sigmas.push(0.0);
    MossSchedule {
        timesteps,
        sigmas: all_sigmas,
    }
}

/// One flow-matching Euler integration step, fp32:
/// `x_{i+1} = x_i + (sigma_{i+1} − sigma_i) · velocity`.
///
/// `velocity` is the DiT prediction already converted to f32 (CFG
/// combinations also happen in f32, matching the reference).
pub fn euler_step(
    schedule: &MossSchedule,
    step_index: usize,
    sample: &candle::Tensor,
    velocity: &candle::Tensor,
) -> candle::Result<candle::Tensor> {
    let sigma = schedule.sigmas[step_index];
    let sigma_next = schedule.sigmas[step_index + 1];
    let dt = sigma_next - sigma;
    // sample + dt * velocity, all in f32.
    let sample = sample.to_dtype(candle::DType::F32)?;
    let velocity = velocity.to_dtype(candle::DType::F32)?;
    velocity.affine(dt, 0.0)? + sample
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sigma grid formula against hand-derived values:
    /// shift=1 collapses to identity over v, and the training grid
    /// with total=1000, shift=1 is [0.001, ..., 1.0] ascending.
    #[test]
    fn test_training_grid_is_ascending_unit_shift() {
        let mut training = grid(1000, 1.0, 1000);
        training.reverse();
        assert!((training[0] - 0.001).abs() < 1e-12, "first {}", training[0]);
        assert!((training[999] - 1.0).abs() < 1e-12, "last {}", training[999]);
        for w in training.windows(2) {
            assert!(w[0] <= w[1], "not ascending: {} > {}", w[0], w[1]);
        }
    }

    /// Inference grid at shift=5 spans (1.0, ~0.0244] — values
    /// verified against the reference Python implementation.
    #[test]
    fn test_inference_grid_bounds() {
        let sigmas = grid(50, 5.0, 1000);
        assert!((sigmas[0] - 1.0).abs() < 1e-12);
        assert!((sigmas[49] - 0.024_414_062_5).abs() < 1e-9, "last {}", sigmas[49]);
        for w in sigmas.windows(2) {
            assert!(w[0] > w[1], "not descending");
        }
    }

    /// Endpoints and monotonicity of the default 50-step schedule
    /// (reference values computed with the upstream Python code).
    #[test]
    fn test_default_schedule_shape() {
        let schedule = set_timesteps(50, &MossSchedulerConfig::default());
        // Snapped to the 0.001-resolution training grid.
        assert!((schedule.sigmas[0] - 1.0).abs() < 1e-12);
        assert_eq!(*schedule.sigmas.last().unwrap(), 0.0);
        assert_eq!(schedule.timesteps.len() + 1, schedule.sigmas.len());
        // 50 inference steps produce 50 distinct snapped values.
        assert_eq!(schedule.timesteps.len(), 50);
        // Reference: first five snapped sigmas.
        for (i, expected) in [1.0, 0.996, 0.992, 0.987, 0.983].iter().enumerate() {
            assert!(
                (schedule.sigmas[i] - expected).abs() < 1e-9,
                "sigmas[{i}] = {} expected {expected}",
                schedule.sigmas[i]
            );
        }
        // Reference: last snapped sigma is 0.024 (the raw grid value
        // 0.0244140625 snaps to its nearest training-grid point).
        assert!((schedule.sigmas[49] - 0.024).abs() < 1e-9);
        assert!((schedule.timesteps[49] - 24.0).abs() < 1e-9);
        // Every sigma except the terminal 0 is a multiple of 0.001
        // (snapped onto the training grid).
        for &s in &schedule.sigmas[..schedule.sigmas.len() - 1] {
            let scaled = s * 1000.0;
            assert!((scaled - scaled.round()).abs() < 1e-9, "not snapped: {s}");
        }
        // Descending with no consecutive duplicates.
        for w in schedule.sigmas.windows(2) {
            assert!(w[0] > w[1], "not strictly descending: {} {}", w[0], w[1]);
        }
        // Timesteps = sigma * 1000.
        assert!((schedule.timesteps[0] - 1000.0).abs() < 1e-9);
    }

    /// The reference dedupes snapped duplicates — verify with a step
    /// count dense enough that two inference sigmas snap onto the same
    /// training point. total=1000 training points and 900 steps
    /// guarantee collisions by pigeonhole.
    #[test]
    fn test_snap_dedupes_collisions() {
        let schedule = set_timesteps(900, &MossSchedulerConfig::default());
        assert!(
            schedule.timesteps.len() < 900,
            "expected dedupe, got {}",
            schedule.timesteps.len()
        );
        for w in schedule.sigmas.windows(2) {
            assert!(w[0] > w[1]);
        }
    }

    /// Euler step at the last index lands on the model output target:
    /// sigma_next = 0 → x = x + (0 − sigma)·v.
    #[test]
    fn test_euler_step_terminal() -> Result<(), Box<dyn std::error::Error>> {
        use candle::{DType, Device, Tensor};
        let schedule = set_timesteps(50, &MossSchedulerConfig::default());
        let last = schedule.timesteps.len() - 1;
        let dev = Device::Cpu;
        let sample = Tensor::from_vec(vec![1.0f32, 2.0], (1, 2), &dev)?;
        let velocity = Tensor::from_vec(vec![0.5f32, 1.0], (1, 2), &dev)?;
        let out = euler_step(&schedule, last, &sample, &velocity)?;
        let dt = schedule.sigmas[last + 1] - schedule.sigmas[last];
        let vals = out.to_dtype(DType::F64)?.flatten_all()?.to_vec1::<f64>()?;
        assert!((vals[0] - (1.0 + dt * 0.5)).abs() < 1e-6);
        assert!((vals[1] - (2.0 + dt * 1.0)).abs() < 1e-6);
        assert_eq!(out.dtype(), DType::F32);
        Ok(())
    }
}
