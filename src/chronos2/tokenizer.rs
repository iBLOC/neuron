//! Chronos2 quantile tokenizer.
//!
//! Amazon's Chronos2 uses a quantile-based bucketing scheme to
//! convert continuous time series values into discrete token ids.
//! The training data distribution is divided into `vocab_size`
//! bins; each value is mapped to the bin whose quantile range it
//! falls into.
//!
//! ## Algorithm
//!
//! 1. Compute empirical quantiles from the input series (or use
//!    pre-computed quantiles from training).
//! 2. For each value, find the bin index where
//!    `quantiles[i] <= value < quantiles[i+1]`.
//! 3. Special tokens:
//!    - Token 0: BOS (beginning of sequence)
//!    - Token 1: EOS (end of sequence)
//!    - Token 2: PAD (padding)
//!    - Token 3..vocab_size-1: quantile bins
//!
//! ## Differences from v0 uniform bucketing
//!
//! The v0 implementation used simple uniform bucketing:
//! `floor((value - min) / range * (vocab_size - 1))`. This is
//! incorrect for Chronos2 because:
//!
//! 1. It doesn't match the training distribution — Chronos2
//!    trains on quantile-sampled tokens, not uniform ones.
//! 2. It doesn't handle skewed distributions well — most values
//!    end up in a few bins.
//! 3. It doesn't preserve the special token convention (BOS=0,
//!    EOS=1, PAD=2).

use std::borrow::Cow;
use std::path::Path;

/// Special token ids used by Chronos2.
pub mod special_tokens {
    /// Beginning of sequence.
    pub const BOS: u32 = 0;
    /// End of sequence.
    pub const EOS: u32 = 1;
    /// Padding token.
    pub const PAD: u32 = 2;
    /// First quantile bin token.
    pub const FIRST_BIN: u32 = 3;
}

/// Quantile tokenizer configuration.
#[derive(Debug, Clone)]
pub struct QuantileTokenizerConfig {
    /// Total vocabulary size (including special tokens).
    pub vocab_size: usize,
    /// Number of quantile bins (vocab_size - 3 for BOS/EOS/PAD).
    pub num_bins: usize,
    /// Pre-computed quantile boundaries. If `None`, quantiles
    /// are computed from the input series at tokenization time.
    /// Length should be `num_bins + 1` (inclusive boundaries).
    pub quantile_boundaries: Option<Vec<f64>>,
}

impl QuantileTokenizerConfig {
    /// Create a new config with the given vocabulary size.
    /// Quantiles will be computed from the input series.
    pub fn new(vocab_size: usize) -> Self {
        assert!(vocab_size >= 4, "vocab_size must be >= 4 (3 special + at least 1 bin)");
        Self {
            vocab_size,
            num_bins: vocab_size - special_tokens::FIRST_BIN as usize,
            quantile_boundaries: None,
        }
    }

    /// Create a config with pre-computed quantile boundaries.
    /// Boundaries should be sorted in ascending order with
    /// length `num_bins + 1`.
    pub fn with_boundaries(vocab_size: usize, boundaries: Vec<f64>) -> Self {
        let num_bins = vocab_size - special_tokens::FIRST_BIN as usize;
        assert_eq!(
            boundaries.len(),
            num_bins + 1,
            "boundaries length must be num_bins + 1"
        );
        Self {
            vocab_size,
            num_bins,
            quantile_boundaries: Some(boundaries),
        }
    }

    /// Load pre-computed quantile boundaries from a JSON file.
    ///
    /// The file should contain a JSON array of float values
    /// representing the quantile boundaries.
    pub fn load_from_json(path: impl AsRef<Path>, vocab_size: usize) -> std::io::Result<Self> {
        let path = path.as_ref();
        let data = std::fs::read_to_string(path)?;
        let boundaries: Vec<f64> = serde_json::from_str(&data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Ok(Self::with_boundaries(vocab_size, boundaries))
    }

    /// Load pre-computed quantile boundaries from a safetensors file.
    ///
    /// The file should contain a tensor named "quantile_boundaries"
    /// with shape [num_bins + 1].
    #[cfg(feature = "chronos2")]
    pub fn load_from_safetensors(path: impl AsRef<Path>, vocab_size: usize) -> std::io::Result<Self> {
        let path = path.as_ref();
        let data = std::fs::read(path)?;
        let tensors = safetensors::SafeTensors::deserialize(&data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let tensor = tensors.tensor("quantile_boundaries")
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let boundaries: Vec<f64> = tensor.data()
            .chunks_exact(8)
            .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        Ok(Self::with_boundaries(vocab_size, boundaries))
    }

    /// Get the quantile boundaries if pre-computed.
    pub fn boundaries(&self) -> Option<&[f64]> {
        self.quantile_boundaries.as_deref()
    }
}

/// Quantile tokenizer for Chronos2.
///
/// Converts continuous time series values into discrete token ids
/// using quantile-based bucketing.
#[derive(Debug, Clone)]
pub struct QuantileTokenizer {
    config: QuantileTokenizerConfig,
}

impl QuantileTokenizer {
    /// Create a new tokenizer with the given configuration.
    pub fn new(config: QuantileTokenizerConfig) -> Self {
        Self { config }
    }

    /// Tokenize a time series into token ids.
    ///
    /// Returns a vector of token ids with:
    /// - BOS token at the start
    /// - Quantile bin tokens for each value
    /// - No EOS token (added by the caller if needed)
    ///
    /// If the series is empty, returns just `[BOS]`.
    pub fn tokenize(&self, series: &[f64]) -> Vec<u32> {
        if series.is_empty() {
            return vec![special_tokens::BOS];
        }

        let boundaries = match &self.config.quantile_boundaries {
            Some(b) => Cow::Borrowed(b),
            None => Cow::Owned(self.compute_quantiles(series)),
        };

        let mut tokens = Vec::with_capacity(series.len() + 1);
        tokens.push(special_tokens::BOS);

        for &value in series {
            let bin = self.value_to_bin(value, &boundaries);
            tokens.push(bin + special_tokens::FIRST_BIN);
        }

        tokens
    }

    /// Tokenize a series and add horizon placeholder tokens.
    ///
    /// Returns: `[BOS, series_tokens..., placeholder * horizon]`
    pub fn tokenize_with_horizon(&self, series: &[f64], horizon: usize) -> Vec<u32> {
        let mut tokens = self.tokenize(series);
        // Add placeholder tokens for the forecast horizon.
        // Use PAD token as placeholder — the model learns to
        // predict the next token given the context.
        tokens.extend(std::iter::repeat_n(special_tokens::PAD, horizon));
        tokens
    }

    /// Detokenize token ids back to values using the quantile
    /// boundaries. Each token id is mapped to the midpoint of
    /// its bin.
    pub fn detokenize(&self, tokens: &[u32]) -> Vec<f64> {
        let boundaries = match &self.config.quantile_boundaries {
            Some(b) => Cow::Borrowed(b),
            None => {
                // Without boundaries, we can't detokenize.
                // Return zeros.
                return vec![0.0; tokens.len()];
            }
        };

        tokens
            .iter()
            .filter(|&&t| t >= special_tokens::FIRST_BIN)
            .map(|&t| {
                let bin = (t - special_tokens::FIRST_BIN) as usize;
                if bin + 1 < boundaries.len() {
                    // Midpoint of the bin.
                    (boundaries[bin] + boundaries[bin + 1]) / 2.0
                } else {
                    // Last bin — use the upper boundary.
                    boundaries[boundaries.len() - 1]
                }
            })
            .collect()
    }

    /// Compute empirical quantiles from the input series.
    ///
    /// Returns `num_bins + 1` boundary values that divide the
    /// series into `num_bins` equal-probability bins.
    fn compute_quantiles(&self, series: &[f64]) -> Vec<f64> {
        let n = series.len();
        if n == 0 {
            return vec![0.0; self.config.num_bins + 1];
        }

        // Sort a copy of the series.
        let mut sorted: Vec<f64> = series.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let num_bins = self.config.num_bins;
        let mut boundaries = Vec::with_capacity(num_bins + 1);

        // Compute quantile boundaries.
        // For num_bins bins, we need num_bins + 1 boundaries.
        // The i-th boundary is at quantile i/num_bins.
        for i in 0..=num_bins {
            let quantile = i as f64 / num_bins as f64;
            let idx = quantile * (n - 1) as f64;
            let lower = idx.floor() as usize;
            let upper = idx.ceil() as usize;
            let frac = idx - lower as f64;

            if lower == upper {
                boundaries.push(sorted[lower]);
            } else {
                // Linear interpolation between adjacent values.
                boundaries.push(sorted[lower] * (1.0 - frac) + sorted[upper] * frac);
            }
        }

        boundaries
    }

    /// Map a value to its bin index using the boundaries.
    ///
    /// Returns an index in `[0, num_bins)`.
    fn value_to_bin(&self, value: f64, boundaries: &[f64]) -> u32 {
        if boundaries.is_empty() {
            return 0;
        }

        // Binary search for the bin.
        // We want the largest i such that boundaries[i] <= value.
        let mut lo = 0;
        let mut hi = boundaries.len() - 1;

        // Handle edge cases.
        if value <= boundaries[0] {
            return 0;
        }
        if value >= boundaries[hi] {
            return (self.config.num_bins - 1) as u32;
        }

        // Binary search.
        while lo < hi - 1 {
            let mid = (lo + hi) / 2;
            if boundaries[mid] <= value {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        lo as u32
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    /// Get the number of quantile bins.
    pub fn num_bins(&self) -> usize {
        self.config.num_bins
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_special_tokens_are_correct() {
        assert_eq!(special_tokens::BOS, 0);
        assert_eq!(special_tokens::EOS, 1);
        assert_eq!(special_tokens::PAD, 2);
        assert_eq!(special_tokens::FIRST_BIN, 3);
    }

    #[test]
    fn test_tokenizer_empty_series() {
        let config = QuantileTokenizerConfig::new(8);
        let tokenizer = QuantileTokenizer::new(config);
        let tokens = tokenizer.tokenize(&[]);
        assert_eq!(tokens, vec![special_tokens::BOS]);
    }

    #[test]
    fn test_tokenizer_single_value() {
        let config = QuantileTokenizerConfig::new(8);
        let tokenizer = QuantileTokenizer::new(config);
        let tokens = tokenizer.tokenize(&[5.0]);
        assert_eq!(tokens.len(), 2); // BOS + 1 value
        assert_eq!(tokens[0], special_tokens::BOS);
        // Single value should map to bin 0 (or last bin depending on implementation).
        assert!(tokens[1] >= special_tokens::FIRST_BIN);
    }

    #[test]
    fn test_tokenizer_monotonic_mapping() {
        let config = QuantileTokenizerConfig::new(8);
        let tokenizer = QuantileTokenizer::new(config);

        // Larger values should map to larger (or equal) bin indices.
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let tokens = tokenizer.tokenize(&series);

        // Skip BOS token.
        for i in 2..tokens.len() {
            assert!(
                tokens[i] >= tokens[i - 1],
                "token at {} ({}) should be >= token at {} ({})",
                i,
                tokens[i],
                i - 1,
                tokens[i - 1]
            );
        }
    }

    #[test]
    fn test_tokenizer_with_precomputed_boundaries() {
        let boundaries = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let config = QuantileTokenizerConfig::with_boundaries(8, boundaries);
        let tokenizer = QuantileTokenizer::new(config);

        let series = vec![0.5, 1.5, 2.5, 3.5, 4.5];
        let tokens = tokenizer.tokenize(&series);

        // Each value should map to its corresponding bin.
        assert_eq!(tokens[1], special_tokens::FIRST_BIN + 0); // 0.5 in [0,1)
        assert_eq!(tokens[2], special_tokens::FIRST_BIN + 1); // 1.5 in [1,2)
        assert_eq!(tokens[3], special_tokens::FIRST_BIN + 2); // 2.5 in [2,3)
        assert_eq!(tokens[4], special_tokens::FIRST_BIN + 3); // 3.5 in [3,4)
        assert_eq!(tokens[5], special_tokens::FIRST_BIN + 4); // 4.5 in [4,5)
    }

    #[test]
    fn test_tokenizer_boundary_values() {
        let boundaries = vec![0.0, 1.0, 2.0, 3.0];
        let config = QuantileTokenizerConfig::with_boundaries(6, boundaries);
        let tokenizer = QuantileTokenizer::new(config);

        // Test exact boundary values.
        let series = vec![0.0, 1.0, 2.0, 3.0];
        let tokens = tokenizer.tokenize(&series);

        // 0.0 at lower boundary -> bin 0
        assert_eq!(tokens[1], special_tokens::FIRST_BIN);
        // 1.0 at boundary -> bin 1
        assert_eq!(tokens[2], special_tokens::FIRST_BIN + 1);
        // 2.0 at boundary -> bin 2
        assert_eq!(tokens[3], special_tokens::FIRST_BIN + 2);
        // 3.0 at upper boundary -> bin 2 (last bin)
        assert_eq!(tokens[4], special_tokens::FIRST_BIN + 2);
    }

    #[test]
    fn test_tokenizer_with_horizon() {
        let config = QuantileTokenizerConfig::new(8);
        let tokenizer = QuantileTokenizer::new(config);

        let series = vec![1.0, 2.0, 3.0];
        let tokens = tokenizer.tokenize_with_horizon(&series, 5);

        // BOS + 3 series tokens + 5 horizon tokens = 9.
        assert_eq!(tokens.len(), 9);
        assert_eq!(tokens[0], special_tokens::BOS);
        // Last 5 should be PAD (horizon placeholders).
        for i in 4..9 {
            assert_eq!(tokens[i], special_tokens::PAD);
        }
    }

    #[test]
    fn test_detokenize_with_boundaries() {
        let boundaries = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let config = QuantileTokenizerConfig::with_boundaries(7, boundaries);
        let tokenizer = QuantileTokenizer::new(config);

        let tokens = vec![
            special_tokens::BOS,
            special_tokens::FIRST_BIN,
            special_tokens::FIRST_BIN + 1,
            special_tokens::FIRST_BIN + 2,
            special_tokens::FIRST_BIN + 3,
        ];
        let values = tokenizer.detokenize(&tokens);

        // Should return midpoints of bins.
        assert_eq!(values.len(), 4);
        assert!((values[0] - 0.5).abs() < 1e-10); // [0,1) midpoint
        assert!((values[1] - 1.5).abs() < 1e-10); // [1,2) midpoint
        assert!((values[2] - 2.5).abs() < 1e-10); // [2,3) midpoint
        assert!((values[3] - 3.5).abs() < 1e-10); // [3,4) midpoint
    }

    #[test]
    fn test_tokenizer_vocab_size() {
        let config = QuantileTokenizerConfig::new(16);
        let tokenizer = QuantileTokenizer::new(config);
        assert_eq!(tokenizer.vocab_size(), 16);
        assert_eq!(tokenizer.num_bins(), 13); // 16 - 3 special tokens
    }

    #[test]
    fn test_tokenizer_uniform_distribution() {
        let config = QuantileTokenizerConfig::new(8);
        let tokenizer = QuantileTokenizer::new(config);

        let series: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let tokens = tokenizer.tokenize(&series);

        // All tokens should be valid (in range).
        for &t in &tokens[1..] {
            assert!(t >= special_tokens::FIRST_BIN && t < 8, "token {} out of range", t);
        }
        // Should have BOS at start.
        assert_eq!(tokens[0], special_tokens::BOS);
    }

    #[test]
    fn test_tokenizer_skewed_distribution() {
        let config = QuantileTokenizerConfig::new(8);
        let tokenizer = QuantileTokenizer::new(config);

        let mut series = vec![0.0; 90];
        series.extend((1..=10).map(|x| x as f64));

        let tokens = tokenizer.tokenize(&series);

        // Should still produce valid tokens.
        for &t in &tokens {
            assert!(t < 8, "token {} out of range", t);
        }
    }

    #[test]
    fn test_compute_quantiles_empty() {
        let config = QuantileTokenizerConfig::new(8);
        let tokenizer = QuantileTokenizer::new(config);
        let boundaries = tokenizer.compute_quantiles(&[]);
        assert_eq!(boundaries.len(), 6); // num_bins + 1
        assert!(boundaries.iter().all(|&b| b == 0.0));
    }

    #[test]
    fn test_compute_quantiles_single_value() {
        let config = QuantileTokenizerConfig::new(8);
        let tokenizer = QuantileTokenizer::new(config);
        let boundaries = tokenizer.compute_quantiles(&[5.0]);
        assert_eq!(boundaries.len(), 6);
        // All boundaries should be 5.0 (single value).
        assert!(boundaries.iter().all(|&b| (b - 5.0).abs() < 1e-10));
    }

    #[test]
    fn test_compute_quantiles_sorted() {
        let config = QuantileTokenizerConfig::new(8);
        let tokenizer = QuantileTokenizer::new(config);
        let series = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0];
        let boundaries = tokenizer.compute_quantiles(&series);

        // Boundaries should be sorted.
        for i in 1..boundaries.len() {
            assert!(
                boundaries[i] >= boundaries[i - 1],
                "boundaries not sorted at index {}: {} < {}",
                i,
                boundaries[i],
                boundaries[i - 1]
            );
        }
    }

    #[test]
    fn test_value_to_bin_edge_cases() {
        let boundaries = vec![0.0, 1.0, 2.0, 3.0];
        let config = QuantileTokenizerConfig::with_boundaries(6, boundaries);
        let tokenizer = QuantileTokenizer::new(config);

        // Below lower boundary.
        assert_eq!(tokenizer.value_to_bin(-1.0, &[0.0, 1.0, 2.0, 3.0]), 0);
        // Above upper boundary.
        assert_eq!(tokenizer.value_to_bin(4.0, &[0.0, 1.0, 2.0, 3.0]), 2);
        // Exactly at lower boundary.
        assert_eq!(tokenizer.value_to_bin(0.0, &[0.0, 1.0, 2.0, 3.0]), 0);
    }

    #[test]
    fn test_roundtrip_tokenize_detokenize() {
        let boundaries = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let config = QuantileTokenizerConfig::with_boundaries(8, boundaries.clone());
        let tokenizer = QuantileTokenizer::new(config);

        let original = vec![0.5, 1.5, 2.5, 3.5, 4.5];
        let tokens = tokenizer.tokenize(&original);
        let recovered = tokenizer.detokenize(&tokens);

        // Recovered values should be midpoints of bins.
        assert_eq!(recovered.len(), 5);
        for (i, (_orig, rec)) in original.iter().zip(recovered.iter()).enumerate() {
            // Each original value should be within its bin's range.
            let bin = i as u32;
            let lower = boundaries[bin as usize];
            let upper = boundaries[bin as usize + 1];
            assert!(
                *rec >= lower && *rec <= upper,
                "recovered value {} at index {} not in bin range [{}, {}]",
                rec,
                i,
                lower,
                upper
            );
        }
    }

    #[test]
    fn test_load_from_json() {
        let boundaries = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let json = serde_json::to_string(&boundaries).unwrap();
        let dir = std::env::temp_dir().join("chronos2-test-json");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("quantiles.json");
        std::fs::write(&path, json).unwrap();

        let config = QuantileTokenizerConfig::load_from_json(&path, 8).unwrap();
        assert_eq!(config.vocab_size, 8);
        assert_eq!(config.num_bins, 5);
        assert!(config.boundaries().is_some());
        assert_eq!(config.boundaries().unwrap().len(), 6);

        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn test_config_boundaries_accessor() {
        let config = QuantileTokenizerConfig::new(8);
        assert!(config.boundaries().is_none());

        let boundaries = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let config = QuantileTokenizerConfig::with_boundaries(8, boundaries);
        assert!(config.boundaries().is_some());
        assert_eq!(config.boundaries().unwrap().len(), 6);
    }
}
