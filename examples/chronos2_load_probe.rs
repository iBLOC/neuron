//! One-off probe: load the chronos2 checkpoint directly (bypassing the
//! registry's stub fallback) and print the REAL load error.
//!
//! ```bash
//! cargo run --release --example chronos2_load_probe --features chronos2
//! ```

use neuron_lib::chronos2::Chronos2Runtime as _;

fn main() {
    let home = std::env::var("USERPROFILE").unwrap_or_default();
    let dir = std::path::PathBuf::from(home)
        .join(".neuron/models/chronos2/chronos-t5-small");
    unsafe {
        std::env::set_var("NEURON_CHRONOS2_MODEL_PATH", &dir);
        std::env::set_var("NEURON_CHRONOS2_DEFAULT_MODEL", "chronos-t5-small");
    }
    println!("env_discovery: {:?}", neuron_lib::chronos2::env_discovery());
    println!("resolve: {:?}", neuron_lib::chronos2::resolve_model_path("chronos-t5-small"));
    let dev = neuron_lib::config::NeuronConfig::default_device();
    match neuron_lib::chronos2::candle_runtime::CandleChronos2Runtime::load(&dir, dev) {
        Ok(rt) => {
            println!("LOAD OK: model_id={}", rt.model_id());
            let series: Vec<f64> = (0..64).map(|i| (i as f64 * 0.5).sin()).collect();
            let req = neuron_lib::chronos2::ForecastRequest {
                model: "chronos-t5-small".into(),
                series,
                horizon: 12,
                method: neuron_lib::chronos2::ForecastMethod::Median,
            };
            let rt = std::sync::Arc::new(rt);
            let rt2 = rt.clone();
            let resp = tokio::runtime::Runtime::new()
                .unwrap()
                .block_on(async move { rt2.forecast(req).await });
            match resp {
                Ok(r) => println!("FORECAST OK: {} points in {} ms", r.forecast.len(), r.took_ms),
                Err(e) => println!("FORECAST ERR: {e}"),
            }
        }
        Err(e) => println!("LOAD ERR: {e}"),
    }
}
