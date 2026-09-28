//! Multi-quantile output head for Chronos2.
//!
//! Chronos2 emits a per-vocab-bin distribution per output step.
//! The output head is `Linear(d_model, num_quantiles)` followed
//! by sigmoid to produce a cumulative distribution function (CDF)
//! over quantile fractions. The runtime then extracts specific
//! quantiles (mean, median, or user-specified) from the CDF.
//!
//! ## CDF to quantile conversion
//!
//! The output of the head is `[batch, seq_len, num_quantiles]`
//! where each value is in (0, 1) representing the cumulative
//! probability up to that quantile fraction. To extract a
//! specific quantile:
//!
//! 1. For median (50th percentile): take the middle quantile
//!    column.
//! 2. For mean: average all quantile columns weighted by their
//!    quantile fractions.
//! 3. For arbitrary quantile q: interpolate between the two
//!    nearest quantile columns.

use candle::{DType, Module, Result, Tensor};
use candle_nn::{linear, Linear, RmsNorm, VarBuilder};

use super::vendor::T5BlockConfig;

const DEFAULT_NUM_QUANTILES: usize = 64;

/// Interpolation method for quantile extraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationMethod {
    /// Linear interpolation between adjacent quantile values.
    Linear,
    /// Nearest neighbor - use the closest quantile value.
    Nearest,
    /// Cubic interpolation using 4 surrounding points.
    Cubic,
}

/// Multi-quantile output head configuration.
#[derive(Debug, Clone)]
pub struct OutputHeadConfig {
    pub d_model: usize,
    pub num_quantiles: usize,
    pub layer_norm_eps: f64,
}

impl OutputHeadConfig {
    pub fn new(d_model: usize, num_quantiles: usize) -> Self {
        Self {
            d_model,
            num_quantiles,
            layer_norm_eps: 1e-6,
        }
    }

    pub fn from_t5_config(cfg: &T5BlockConfig, num_quantiles: Option<usize>) -> Self {
        Self {
            d_model: cfg.d_model,
            num_quantiles: num_quantiles.unwrap_or(DEFAULT_NUM_QUANTILES),
            layer_norm_eps: cfg.layer_norm_eps,
        }
    }
}

/// Multi-quantile output head.
///
/// Projects decoder hidden states to a CDF over quantile
/// fractions. Each output step produces `num_quantiles` values
/// representing the cumulative probability at evenly-spaced
/// quantile fractions [0, 1/num_quantiles, 2/num_quantiles, ..., 1].
pub struct Chronos2OutputHead {
    final_norm: RmsNorm,
    proj: Linear,
    bias: Tensor,
    num_quantiles: usize,
}

impl Chronos2OutputHead {
    pub fn new(cfg: OutputHeadConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            final_norm: candle_nn::rms_norm(
                cfg.d_model,
                cfg.layer_norm_eps,
                vb.pp("final_norm"),
            )?,
            proj: linear(cfg.d_model, cfg.num_quantiles, vb.pp("proj"))?,
            bias: vb.get_with_hints(
                (cfg.num_quantiles,),
                "bias",
                candle_nn::init::ZERO,
            )?,
            num_quantiles: cfg.num_quantiles,
        })
    }

    /// Forward pass: project hidden states to CDF.
    ///
    /// Input: `[batch, seq_len, d_model]`
    /// Output: `[batch, seq_len, num_quantiles]` with values in (0, 1)
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.final_norm.forward(x)?;
        let proj = self.proj.forward(&x)?;
        let proj = proj.broadcast_add(&self.bias)?;
        candle_nn::ops::sigmoid(&proj)
    }

    /// Extract the median (50th percentile) from the CDF.
    ///
    /// Input: `[batch, seq_len, num_quantiles]`
    /// Output: `[batch, seq_len]`
    pub fn extract_median(&self, cdf: &Tensor) -> Result<Tensor> {
        let mid = self.num_quantiles / 2;
        cdf.get_on_dim(2, mid)
    }

    /// Extract the mean from the CDF by averaging all quantile
    /// columns weighted by their quantile fractions.
    ///
    /// Input: `[batch, seq_len, num_quantiles]`
    /// Output: `[batch, seq_len]`
    pub fn extract_mean(&self, cdf: &Tensor) -> Result<Tensor> {
        let (_batch, _seq_len, num_q) = cdf.dims3()?;
        let device = cdf.device();

        // Create quantile fraction weights: [0, 1/N, 2/N, ..., (N-1)/N]
        let weights: Vec<f32> = (0..num_q)
            .map(|i| i as f32 / num_q as f32)
            .collect();
        let weights = Tensor::from_vec(weights, (1, 1, num_q), device)?;

        // Weighted sum along quantile dimension.
        let weighted = cdf.broadcast_mul(&weights)?;
        weighted.sum_keepdim(2)?.squeeze(2)
    }

    /// Extract a specific quantile from the CDF using linear interpolation.
    ///
    /// Input: `[batch, seq_len, num_quantiles]`
    /// Output: `[batch, seq_len]`
    pub fn extract_quantile(&self, cdf: &Tensor, q: f64) -> Result<Tensor> {
        self.extract_quantile_with_method(cdf, q, InterpolationMethod::Linear)
    }

    /// Extract a specific quantile using the specified interpolation method.
    ///
    /// Input: `[batch, seq_len, num_quantiles]`
    /// Output: `[batch, seq_len]`
    pub fn extract_quantile_with_method(
        &self,
        cdf: &Tensor,
        q: f64,
        method: InterpolationMethod,
    ) -> Result<Tensor> {
        assert!((0.0..=1.0).contains(&q), "quantile must be in [0, 1]");
        let (_batch, _seq_len, num_q) = cdf.dims3()?;

        match method {
            InterpolationMethod::Linear => self.extract_quantile_linear(cdf, q, num_q),
            InterpolationMethod::Nearest => self.extract_quantile_nearest(cdf, q, num_q),
            InterpolationMethod::Cubic => self.extract_quantile_cubic(cdf, q, num_q),
        }
    }

    fn extract_quantile_linear(&self, cdf: &Tensor, q: f64, num_q: usize) -> Result<Tensor> {
        let device = cdf.device();
        let q_index = q * (num_q - 1) as f64;
        let lower = q_index.floor() as usize;
        let upper = (lower + 1).min(num_q - 1);
        let frac = q_index - lower as f64;

        if lower == upper || frac < 1e-10 {
            cdf.get_on_dim(2, lower)
        } else if frac > 1.0 - 1e-10 {
            cdf.get_on_dim(2, upper)
        } else {
            let lower_val = cdf.get_on_dim(2, lower)?;
            let upper_val = cdf.get_on_dim(2, upper)?;
            let one_minus = Tensor::full(1.0 - frac as f32, lower_val.shape(), device)?;
            let frac_tensor = Tensor::full(frac as f32, upper_val.shape(), device)?;
            lower_val.broadcast_mul(&one_minus)?.add(&upper_val.broadcast_mul(&frac_tensor)?)
        }
    }

    fn extract_quantile_nearest(&self, cdf: &Tensor, q: f64, num_q: usize) -> Result<Tensor> {
        let q_index = q * (num_q - 1) as f64;
        let nearest = q_index.round() as usize;
        let nearest = nearest.min(num_q - 1);
        cdf.get_on_dim(2, nearest)
    }

    fn extract_quantile_cubic(&self, cdf: &Tensor, q: f64, num_q: usize) -> Result<Tensor> {
        let device = cdf.device();
        let q_index = q * (num_q - 1) as f64;
        let lower = q_index.floor() as usize;

        if num_q < 4 || lower == 0 || lower >= num_q - 2 {
            return self.extract_quantile_linear(cdf, q, num_q);
        }

        let p0 = cdf.get_on_dim(2, lower - 1)?;
        let p1 = cdf.get_on_dim(2, lower)?;
        let p2 = cdf.get_on_dim(2, lower + 1)?;
        let p3 = cdf.get_on_dim(2, lower + 2)?;

        let t = (q_index - lower as f64) as f32;
        let t2 = t * t;
        let t3 = t2 * t;

        let a = Tensor::full(-0.5 * t3 + t2 - 0.5 * t, p0.shape(), device)?;
        let b = Tensor::full(1.5 * t3 - 2.5 * t2 + 1.0, p1.shape(), device)?;
        let c = Tensor::full(-1.5 * t3 + 2.0 * t2 + 0.5 * t, p2.shape(), device)?;
        let d = Tensor::full(0.5 * t3 - 0.5 * t2, p3.shape(), device)?;

        p0.broadcast_mul(&a)?
            .add(&p1.broadcast_mul(&b)?)?
            .add(&p2.broadcast_mul(&c)?)?
            .add(&p3.broadcast_mul(&d)?)
    }

    /// Extract a confidence interval from the CDF.
    ///
    /// Returns (lower, upper) tensors for the given confidence
    /// level (e.g., 0.9 for 90% CI). The lower bound is at
    /// `(1-level)/2` and the upper bound is at `1-(1-level)/2`.
    pub fn extract_confidence_interval(
        &self,
        cdf: &Tensor,
        level: f64,
    ) -> Result<(Tensor, Tensor)> {
        self.extract_confidence_interval_with_method(cdf, level, InterpolationMethod::Linear)
    }

    /// Extract a confidence interval using the specified interpolation method.
    pub fn extract_confidence_interval_with_method(
        &self,
        cdf: &Tensor,
        level: f64,
        method: InterpolationMethod,
    ) -> Result<(Tensor, Tensor)> {
        assert!((0.0..=1.0).contains(&level), "level must be in [0, 1]");
        let alpha = (1.0 - level) / 2.0;
        let lower = self.extract_quantile_with_method(cdf, alpha, method)?;
        let upper = self.extract_quantile_with_method(cdf, 1.0 - alpha, method)?;
        Ok((lower, upper))
    }

    /// Extract multiple confidence intervals at once.
    ///
    /// Returns a vector of (lower, upper) pairs for each level.
    pub fn extract_confidence_intervals(
        &self,
        cdf: &Tensor,
        levels: &[f64],
    ) -> Result<Vec<(Tensor, Tensor)>> {
        levels.iter()
            .map(|&level| self.extract_confidence_interval(cdf, level))
            .collect()
    }

    /// Get the number of quantiles.
    pub fn num_quantiles(&self) -> usize {
        self.num_quantiles
    }
}

/// Extract the last `horizon` steps from a sequence.
///
/// Input shape: `[batch, seq_len, num_quantiles]`
/// Output shape: `[batch, horizon, num_quantiles]`
pub fn take_horizon(projections: &Tensor, horizon: u32) -> Result<Tensor> {
    let dims = projections.dims();
    if dims.len() == 3 {
        let seq_len = dims[1];
        let start = seq_len.saturating_sub(horizon as usize);
        projections.narrow(1, start, seq_len - start)
    } else {
        let seq_len = dims[dims.len() - 1];
        let start = seq_len.saturating_sub(horizon as usize);
        projections.narrow(dims.len() - 1, start, seq_len - start)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{Device, DType};

    fn make_head(num_quantiles: usize) -> Chronos2OutputHead {
        let cfg = OutputHeadConfig::new(8, num_quantiles);
        let dev = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &dev);
        Chronos2OutputHead::new(cfg, vb).expect("head")
    }

    #[test]
    fn forward_output_shape() {
        let head = make_head(64);
        let dev = Device::Cpu;
        let x = Tensor::zeros((2, 5, 8), DType::F32, &dev).unwrap();
        let y = head.forward(&x).unwrap();
        assert_eq!(y.dims3().unwrap(), (2, 5, 64));
    }

    #[test]
    fn forward_output_in_unit_interval() {
        let head = make_head(64);
        let dev = Device::Cpu;
        let x = Tensor::randn(0.0f32, 1.0f32, (1, 10, 8), &dev).unwrap();
        let y = head.forward(&x).unwrap();
        let data: Vec<f32> = y.flatten_all().unwrap().to_vec1().unwrap();
        for &v in &data {
            assert!((0.0..=1.0).contains(&v), "value {} out of [0, 1]", v);
        }
    }

    #[test]
    fn forward_output_is_sorted_along_quantiles() {
        let head = make_head(64);
        let dev = Device::Cpu;
        let x = Tensor::randn(0.0f32, 1.0f32, (1, 5, 8), &dev).unwrap();
        let y = head.forward(&x).unwrap();
        let data: Vec<f32> = y.flatten_all().unwrap().to_vec1().unwrap();

        // For each step, check that quantile values are non-decreasing.
        for step in 0..5 {
            for q in 1..64 {
                let prev = data[step * 64 + q - 1];
                let curr = data[step * 64 + q];
                // Allow small numerical errors.
                assert!(
                    curr >= prev - 1e-6,
                    "CDF not sorted at step {}: q[{}]={} > q[{}]={}",
                    step,
                    q - 1,
                    prev,
                    q,
                    curr
                );
            }
        }
    }

    #[test]
    fn extract_median_shape() {
        let head = make_head(64);
        let dev = Device::Cpu;
        let cdf = Tensor::zeros((2, 5, 64), DType::F32, &dev).unwrap();
        let median = head.extract_median(&cdf).unwrap();
        assert_eq!(median.dims2().unwrap(), (2, 5));
    }

    #[test]
    fn extract_mean_shape() {
        let head = make_head(64);
        let dev = Device::Cpu;
        let cdf = Tensor::zeros((2, 5, 64), DType::F32, &dev).unwrap();
        let mean = head.extract_mean(&cdf).unwrap();
        assert_eq!(mean.dims2().unwrap(), (2, 5));
    }

    #[test]
    fn extract_quantile_shape() {
        let head = make_head(64);
        let dev = Device::Cpu;
        let cdf = Tensor::zeros((2, 5, 64), DType::F32, &dev).unwrap();
        let q = head.extract_quantile(&cdf, 0.5).unwrap();
        assert_eq!(q.dims2().unwrap(), (2, 5));
    }

    #[test]
    fn extract_quantile_boundary_values() {
        let head = make_head(64);
        let dev = Device::Cpu;
        let cdf = Tensor::zeros((1, 3, 64), DType::F32, &dev).unwrap();

        // 0th percentile should be first column.
        let q0 = head.extract_quantile(&cdf, 0.0).unwrap();
        assert_eq!(q0.dims2().unwrap(), (1, 3));

        // 100th percentile should be last column.
        let q1 = head.extract_quantile(&cdf, 1.0).unwrap();
        assert_eq!(q1.dims2().unwrap(), (1, 3));
    }

    #[test]
    fn extract_median_uses_middle_column() {
        let head = make_head(10);
        let dev = Device::Cpu;
        // Create CDF where each column has a distinct value.
        let mut data = vec![0.0f32; 1 * 1 * 10];
        for i in 0..10 {
            data[i] = i as f32 / 10.0;
        }
        let cdf = Tensor::from_vec(data, (1, 1, 10), &dev).unwrap();
        let median = head.extract_median(&cdf).unwrap();
        let val: f32 = median.flatten_all().unwrap().to_vec1().unwrap()[0];
        // Middle column (index 5) should have value 0.5.
        assert!((val - 0.5).abs() < 1e-6, "median value {}", val);
    }

    #[test]
    fn take_horizon_basic() {
        let dev = Device::Cpu;
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], (1, 5), &dev).unwrap();
        let out = take_horizon(&x, 3).unwrap();
        assert_eq!(out.dims2().unwrap(), (1, 3));
        let data: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(data, vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn take_horizon_clamps() {
        let dev = Device::Cpu;
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (1, 3), &dev).unwrap();
        let out = take_horizon(&x, 10).unwrap();
        assert_eq!(out.dims2().unwrap(), (1, 3));
    }

    #[test]
    fn take_horizon_3d() {
        let dev = Device::Cpu;
        let x = Tensor::zeros((2, 10, 64), DType::F32, &dev).unwrap();
        let out = take_horizon(&x, 5).unwrap();
        // For 3D tensors, narrow dimension 1 (seq_len).
        assert_eq!(out.dims3().unwrap(), (2, 5, 64));
    }

    #[test]
    fn output_head_config_from_t5() {
        let t5_cfg = T5BlockConfig::new(512, 2048, 8);
        let head_cfg = OutputHeadConfig::from_t5_config(&t5_cfg, Some(32));
        assert_eq!(head_cfg.d_model, 512);
        assert_eq!(head_cfg.num_quantiles, 32);
    }

    #[test]
    fn output_head_config_default_quantiles() {
        let t5_cfg = T5BlockConfig::new(512, 2048, 8);
        let head_cfg = OutputHeadConfig::from_t5_config(&t5_cfg, None);
        assert_eq!(head_cfg.num_quantiles, DEFAULT_NUM_QUANTILES);
    }

    #[test]
    fn extract_quantile_nearest_shape() {
        let head = make_head(64);
        let dev = Device::Cpu;
        let cdf = Tensor::zeros((2, 5, 64), DType::F32, &dev).unwrap();
        let q = head.extract_quantile_with_method(&cdf, 0.5, InterpolationMethod::Nearest).unwrap();
        assert_eq!(q.dims2().unwrap(), (2, 5));
    }

    #[test]
    fn extract_quantile_cubic_shape() {
        let head = make_head(64);
        let dev = Device::Cpu;
        let cdf = Tensor::zeros((2, 5, 64), DType::F32, &dev).unwrap();
        let q = head.extract_quantile_with_method(&cdf, 0.5, InterpolationMethod::Cubic).unwrap();
        assert_eq!(q.dims2().unwrap(), (2, 5));
    }

    #[test]
    fn extract_quantile_nearest_uses_closest() {
        let head = make_head(10);
        let dev = Device::Cpu;
        let mut data = vec![0.0f32; 1 * 1 * 10];
        for i in 0..10 {
            data[i] = i as f32;
        }
        let cdf = Tensor::from_vec(data, (1, 1, 10), &dev).unwrap();

        // q=0.5 should use index 5 (nearest to 4.5).
        let q = head.extract_quantile_with_method(&cdf, 0.5, InterpolationMethod::Nearest).unwrap();
        let val: f32 = q.flatten_all().unwrap().to_vec1().unwrap()[0];
        assert!((val - 5.0).abs() < 1e-6, "nearest value {}", val);
    }

    #[test]
    fn extract_quantile_methods_produce_similar_results() {
        let head = make_head(64);
        let dev = Device::Cpu;
        let cdf = Tensor::randn(0.0f32, 1.0f32, (1, 3, 64), &dev).unwrap();

        let linear = head.extract_quantile_with_method(&cdf, 0.5, InterpolationMethod::Linear).unwrap();
        let nearest = head.extract_quantile_with_method(&cdf, 0.5, InterpolationMethod::Nearest).unwrap();
        let cubic = head.extract_quantile_with_method(&cdf, 0.5, InterpolationMethod::Cubic).unwrap();

        // All methods should produce valid shapes.
        assert_eq!(linear.dims2().unwrap(), (1, 3));
        assert_eq!(nearest.dims2().unwrap(), (1, 3));
        assert_eq!(cubic.dims2().unwrap(), (1, 3));
    }

    #[test]
    fn interpolation_method_default_is_linear() {
        let method = InterpolationMethod::Linear;
        assert_eq!(method, InterpolationMethod::Linear);
    }

    #[test]
    fn extract_confidence_interval_shape() {
        let head = make_head(64);
        let dev = Device::Cpu;
        let cdf = Tensor::zeros((2, 5, 64), DType::F32, &dev).unwrap();
        let (lower, upper) = head.extract_confidence_interval(&cdf, 0.9).unwrap();
        assert_eq!(lower.dims2().unwrap(), (2, 5));
        assert_eq!(upper.dims2().unwrap(), (2, 5));
    }

    #[test]
    fn extract_confidence_interval_90_percent() {
        let head = make_head(100);
        let dev = Device::Cpu;
        let mut data = vec![0.0f32; 1 * 1 * 100];
        for i in 0..100 {
            data[i] = i as f32 / 100.0;
        }
        let cdf = Tensor::from_vec(data, (1, 1, 100), &dev).unwrap();

        let (lower, upper) = head.extract_confidence_interval(&cdf, 0.9).unwrap();
        let lower_val: f32 = lower.flatten_all().unwrap().to_vec1().unwrap()[0];
        let upper_val: f32 = upper.flatten_all().unwrap().to_vec1().unwrap()[0];

        // 90% CI: lower at 5th percentile, upper at 95th percentile.
        assert!((lower_val - 0.05).abs() < 0.02, "lower {}", lower_val);
        assert!((upper_val - 0.95).abs() < 0.02, "upper {}", upper_val);
    }

    #[test]
    fn extract_confidence_interval_50_percent() {
        let head = make_head(100);
        let dev = Device::Cpu;
        let mut data = vec![0.0f32; 1 * 1 * 100];
        for i in 0..100 {
            data[i] = i as f32 / 100.0;
        }
        let cdf = Tensor::from_vec(data, (1, 1, 100), &dev).unwrap();

        let (lower, upper) = head.extract_confidence_interval(&cdf, 0.5).unwrap();
        let lower_val: f32 = lower.flatten_all().unwrap().to_vec1().unwrap()[0];
        let upper_val: f32 = upper.flatten_all().unwrap().to_vec1().unwrap()[0];

        // 50% CI: lower at 25th percentile, upper at 75th percentile.
        assert!((lower_val - 0.25).abs() < 0.02, "lower {}", lower_val);
        assert!((upper_val - 0.75).abs() < 0.02, "upper {}", upper_val);
    }

    #[test]
    fn extract_confidence_intervals_multiple_levels() {
        let head = make_head(64);
        let dev = Device::Cpu;
        let cdf = Tensor::zeros((1, 3, 64), DType::F32, &dev).unwrap();

        let levels = vec![0.5, 0.9, 0.95];
        let intervals = head.extract_confidence_intervals(&cdf, &levels).unwrap();
        assert_eq!(intervals.len(), 3);

        for (lower, upper) in &intervals {
            assert_eq!(lower.dims2().unwrap(), (1, 3));
            assert_eq!(upper.dims2().unwrap(), (1, 3));
        }
    }

    #[test]
    fn extract_confidence_interval_lower_less_than_upper() {
        let head = make_head(64);
        let dev = Device::Cpu;
        // Create a sorted CDF (monotonically increasing).
        let mut data = vec![0.0f32; 1 * 1 * 64];
        for i in 0..64 {
            data[i] = i as f32 / 64.0;
        }
        let cdf = Tensor::from_vec(data, (1, 1, 64), &dev).unwrap();

        let (lower, upper) = head.extract_confidence_interval(&cdf, 0.9).unwrap();
        let lower_data: Vec<f32> = lower.flatten_all().unwrap().to_vec1().unwrap();
        let upper_data: Vec<f32> = upper.flatten_all().unwrap().to_vec1().unwrap();

        for (l, u) in lower_data.iter().zip(upper_data.iter()) {
            // Lower should be <= upper for a proper CDF.
            assert!(*l <= u + 1e-6, "lower {} > upper {}", l, u);
        }
    }
}
