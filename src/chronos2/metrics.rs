//! Forecast evaluation metrics.
//!
//! Common metrics for evaluating time series forecast accuracy:
//! - MAE (Mean Absolute Error)
//! - RMSE (Root Mean Squared Error)
//! - MAPE (Mean Absolute Percentage Error)
//! - SMAPE (Symmetric Mean Absolute Percentage Error)
//! - MASE (Mean Absolute Scaled Error)

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastMetrics {
    pub mae: f64,
    pub rmse: f64,
    pub mape: f64,
    pub smape: f64,
    pub mase: Option<f64>,
}

/// Calculate MAE (Mean Absolute Error).
///
/// MAE = mean(|actual - predicted|)
pub fn mae(actual: &[f64], predicted: &[f64]) -> f64 {
    assert_eq!(actual.len(), predicted.len(), "arrays must have same length");
    if actual.is_empty() {
        return 0.0;
    }
    let sum: f64 = actual.iter()
        .zip(predicted.iter())
        .map(|(a, p)| (a - p).abs())
        .sum();
    sum / actual.len() as f64
}

/// Calculate RMSE (Root Mean Squared Error).
///
/// RMSE = sqrt(mean((actual - predicted)^2))
pub fn rmse(actual: &[f64], predicted: &[f64]) -> f64 {
    assert_eq!(actual.len(), predicted.len(), "arrays must have same length");
    if actual.is_empty() {
        return 0.0;
    }
    let sum: f64 = actual.iter()
        .zip(predicted.iter())
        .map(|(a, p)| (a - p).powi(2))
        .sum();
    (sum / actual.len() as f64).sqrt()
}

/// Calculate MAPE (Mean Absolute Percentage Error).
///
/// MAPE = mean(|actual - predicted| / |actual|) * 100
///
/// Returns None if any actual value is zero.
pub fn mape(actual: &[f64], predicted: &[f64]) -> Option<f64> {
    assert_eq!(actual.len(), predicted.len(), "arrays must have same length");
    if actual.is_empty() {
        return Some(0.0);
    }
    let mut sum = 0.0;
    for (a, p) in actual.iter().zip(predicted.iter()) {
        if a.abs() < f64::EPSILON {
            return None;
        }
        sum += ((a - p) / a).abs();
    }
    Some(sum / actual.len() as f64 * 100.0)
}

/// Calculate SMAPE (Symmetric Mean Absolute Percentage Error).
///
/// SMAPE = mean(2 * |actual - predicted| / (|actual| + |predicted|)) * 100
pub fn smape(actual: &[f64], predicted: &[f64]) -> f64 {
    assert_eq!(actual.len(), predicted.len(), "arrays must have same length");
    if actual.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0;
    let mut count = 0;
    for (a, p) in actual.iter().zip(predicted.iter()) {
        let denom = a.abs() + p.abs();
        if denom > f64::EPSILON {
            sum += 2.0 * (a - p).abs() / denom;
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        sum / count as f64 * 100.0
    }
}

/// Calculate MASE (Mean Absolute Scaled Error).
///
/// MASE = MAE / MAE_naive
///
/// where MAE_naive is the MAE of the naive forecast (using the
/// training data's one-step-ahead errors). Returns None if
/// training_data is empty.
pub fn mase(actual: &[f64], predicted: &[f64], training_data: &[f64]) -> Option<f64> {
    assert_eq!(actual.len(), predicted.len(), "arrays must have same length");
    if actual.is_empty() {
        return Some(0.0);
    }
    if training_data.len() < 2 {
        return None;
    }

    let mae_actual = mae(actual, predicted);
    let mae_naive: f64 = training_data.windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .sum::<f64>() / (training_data.len() - 1) as f64;

    if mae_naive < f64::EPSILON {
        return None;
    }

    Some(mae_actual / mae_naive)
}

/// Calculate all metrics at once.
pub fn calculate_metrics(
    actual: &[f64],
    predicted: &[f64],
    training_data: Option<&[f64]>,
) -> ForecastMetrics {
    ForecastMetrics {
        mae: mae(actual, predicted),
        rmse: rmse(actual, predicted),
        mape: mape(actual, predicted).unwrap_or(f64::NAN),
        smape: smape(actual, predicted),
        mase: training_data.and_then(|t| mase(actual, predicted, t)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    #[test]
    fn test_mae_perfect() {
        let actual = vec![1.0, 2.0, 3.0];
        let predicted = vec![1.0, 2.0, 3.0];
        assert!((mae(&actual, &predicted) - 0.0).abs() < EPS);
    }

    #[test]
    fn test_mae_simple() {
        let actual = vec![1.0, 2.0, 3.0];
        let predicted = vec![1.5, 2.5, 3.5];
        assert!((mae(&actual, &predicted) - 0.5).abs() < EPS);
    }

    #[test]
    fn test_mae_empty() {
        let actual: Vec<f64> = vec![];
        let predicted: Vec<f64> = vec![];
        assert!((mae(&actual, &predicted) - 0.0).abs() < EPS);
    }

    #[test]
    fn test_rmse_perfect() {
        let actual = vec![1.0, 2.0, 3.0];
        let predicted = vec![1.0, 2.0, 3.0];
        assert!((rmse(&actual, &predicted) - 0.0).abs() < EPS);
    }

    #[test]
    fn test_rmse_simple() {
        let actual = vec![0.0, 0.0];
        let predicted = vec![3.0, 4.0];
        // MSE = (9 + 16) / 2 = 12.5, RMSE = sqrt(12.5) ≈ 3.5355
        let expected = 3.5355339059327378;
        assert!((rmse(&actual, &predicted) - expected).abs() < EPS);
    }

    #[test]
    fn test_mape_perfect() {
        let actual = vec![1.0, 2.0, 3.0];
        let predicted = vec![1.0, 2.0, 3.0];
        assert!((mape(&actual, &predicted).unwrap() - 0.0).abs() < EPS);
    }

    #[test]
    fn test_mape_simple() {
        let actual = vec![100.0, 200.0];
        let predicted = vec![110.0, 180.0];
        // |10/100| + |20/200| = 0.1 + 0.1 = 0.2, mean = 0.1, * 100 = 10%
        assert!((mape(&actual, &predicted).unwrap() - 10.0).abs() < EPS);
    }

    #[test]
    fn test_mape_zero_actual() {
        let actual = vec![0.0, 1.0];
        let predicted = vec![1.0, 1.0];
        assert!(mape(&actual, &predicted).is_none());
    }

    #[test]
    fn test_smape_perfect() {
        let actual = vec![1.0, 2.0, 3.0];
        let predicted = vec![1.0, 2.0, 3.0];
        assert!((smape(&actual, &predicted) - 0.0).abs() < EPS);
    }

    #[test]
    fn test_smape_simple() {
        let actual = vec![100.0, 200.0];
        let predicted = vec![110.0, 180.0];
        // 2*|10|/(100+110) = 20/210 ≈ 0.0952
        // 2*|20|/(200+180) = 40/380 ≈ 0.1053
        // mean ≈ 0.1002, * 100 ≈ 10.02%
        let result = smape(&actual, &predicted);
        assert!((result - 10.026).abs() < 0.01);
    }

    #[test]
    fn test_mase_perfect() {
        let actual = vec![1.0, 2.0, 3.0];
        let predicted = vec![1.0, 2.0, 3.0];
        let training = vec![1.0, 2.0, 3.0, 4.0];
        assert!((mase(&actual, &predicted, &training).unwrap() - 0.0).abs() < EPS);
    }

    #[test]
    fn test_mase_simple() {
        let actual = vec![2.0, 3.0];
        let predicted = vec![1.0, 2.0];
        let training = vec![1.0, 2.0, 3.0, 4.0];
        // MAE = 1.0
        // MAE_naive = mean(|2-1|, |3-2|, |4-3|) = 1.0
        // MASE = 1.0 / 1.0 = 1.0
        assert!((mase(&actual, &predicted, &training).unwrap() - 1.0).abs() < EPS);
    }

    #[test]
    fn test_mase_short_training() {
        let actual = vec![1.0];
        let predicted = vec![1.0];
        let training = vec![1.0];
        assert!(mase(&actual, &predicted, &training).is_none());
    }

    #[test]
    fn test_calculate_metrics() {
        let actual = vec![1.0, 2.0, 3.0];
        let predicted = vec![1.1, 2.2, 2.8];
        let metrics = calculate_metrics(&actual, &predicted, None);
        assert!(metrics.mae > 0.0);
        assert!(metrics.rmse > 0.0);
        assert!(metrics.mape.is_finite());
        assert!(metrics.smape > 0.0);
        assert!(metrics.mase.is_none());
    }

    #[test]
    fn test_calculate_metrics_with_training() {
        let actual = vec![1.0, 2.0, 3.0];
        let predicted = vec![1.1, 2.2, 2.8];
        let training = vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
        let metrics = calculate_metrics(&actual, &predicted, Some(&training));
        assert!(metrics.mase.is_some());
    }
}
