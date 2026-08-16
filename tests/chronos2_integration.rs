//! Integration tests for Chronos2 with real model weights.
//!
//! These tests require the `chronos-bolt-small` model to be
//! downloaded. Run the download script first:
//!
//! ```bash
//! ./scripts/download-chronos2.sh
//! ```

#[cfg(test)]
mod integration {
    use std::path::PathBuf;

    fn model_dir() -> Option<PathBuf> {
        let dir = dirs::home_dir()
            .unwrap_or_default()
            .join(".neure/models/chronos2/chronos-bolt-small");
        if dir.exists() {
            Some(dir)
        } else {
            None
        }
    }

    #[tokio::test]
    async fn test_chronos2_load_and_forecast() {
        let Some(dir) = model_dir() else {
            eprintln!("Skipping test: chronos-bolt-small not found at ~/.neure/models/chronos2/chronos-bolt-small");
            return;
        };

        let device = candle_core::Device::Cpu;
        let rt = neure_lib::chronos2::CandleChronos2Runtime::load(&dir, device)
            .expect("failed to load model");

        let request = neure_lib::chronos2::ForecastRequest {
            model: "chronos-bolt-small".into(),
            series: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            horizon: 5,
            method: neure_lib::chronos2::ForecastMethod::Mean,
        };

        let response = rt.forecast(request).await.expect("forecast failed");
        assert_eq!(response.forecast.len(), 5);
        assert_eq!(response.model, "model");
        assert!(response.took_ms > 0);

        println!("Forecast: {:?}", response.forecast);
        println!("Took: {} ms", response.took_ms);
    }

    #[tokio::test]
    async fn test_chronos2_batch_forecast() {
        let Some(dir) = model_dir() else {
            eprintln!("Skipping test: chronos-bolt-small not found");
            return;
        };

        let device = candle_core::Device::Cpu;
        let rt = neure_lib::chronos2::CandleChronos2Runtime::load(&dir, device)
            .expect("failed to load model");

        let series_list = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![10.0, 20.0, 30.0, 40.0, 50.0],
            vec![100.0, 200.0, 300.0, 400.0, 500.0],
        ];

        let results = rt.forecast_batch(&series_list, 3, neure_lib::chronos2::ForecastMethod::Median)
            .await
            .expect("batch forecast failed");

        assert_eq!(results.len(), 3);
        for (i, forecast) in results.iter().enumerate() {
            assert_eq!(forecast.len(), 3);
            println!("Series {} forecast: {:?}", i, forecast);
        }
    }

    #[tokio::test]
    async fn test_chronos2_confidence_intervals() {
        let Some(dir) = model_dir() else {
            eprintln!("Skipping test: chronos-bolt-small not found");
            return;
        };

        let device = candle_core::Device::Cpu;
        let rt = neure_lib::chronos2::CandleChronos2Runtime::load(&dir, device)
            .expect("failed to load model");

        let request = neure_lib::chronos2::ForecastRequest {
            model: "chronos-bolt-small".into(),
            series: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            horizon: 3,
            method: neure_lib::chronos2::ForecastMethod::Mean,
        };

        let response = rt.forecast(request).await.expect("forecast failed");
        assert_eq!(response.forecast.len(), 3);

        println!("Point forecast: {:?}", response.forecast);
    }
}
