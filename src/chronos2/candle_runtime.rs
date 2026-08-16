//! Chronos2 candle runtime: safetensors loader + T5 encoder-decoder
//! + multi-quantile output head.
//!
//! This runtime loads a Chronos2 checkpoint from disk and runs
//! inference to produce time series forecasts. The pipeline:
//!
//! 1. Tokenize the input series using `QuantileTokenizer`
//! 2. Build encoder/decoder token sequences
//! 3. Run the T5 encoder-decoder forward pass
//! 4. Extract forecasts from the multi-quantile output head
//!
//! ## Checkpoint layout
//!
//! ```text
//! <model_dir>/
//!   config.json              # model hyperparameters
//!   *.safetensors            # architecture weights
//! ```

use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use thiserror::Error;

use super::mask;
use super::output::{OutputHeadConfig, Chronos2OutputHead, take_horizon};
use super::tokenizer::{QuantileTokenizer, QuantileTokenizerConfig};
use super::vendor::T5BlockConfig;
use super::vendor::T5EncoderDecoder;
use super::{Chronos2Error, Chronos2Runtime, ForecastMethod, ForecastRequest, ForecastResponse};

const DEFAULT_NUM_QUANTILES: usize = 64;

/// Select the best available device for inference.
///
/// Priority: CUDA > Metal > CPU
pub fn best_device() -> Device {
    #[cfg(feature = "cuda")]
    {
        if let Ok(device) = Device::cuda_if_available(0) {
            return device;
        }
    }
    #[cfg(feature = "metal")]
    {
        if let Ok(device) = Device::new_metal(0) {
            return device;
        }
    }
    Device::Cpu
}

/// Model hyperparameters loaded from config.json.
#[derive(Debug, Clone)]
pub struct Chronos2Config {
    pub d_model: usize,
    pub num_heads: usize,
    pub d_ff: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub n_enc: usize,
    pub n_dec: usize,
    pub layer_norm_eps: f64,
    pub num_quantiles: usize,
}

impl Chronos2Config {
    pub fn default_bolt_small() -> Self {
        Self {
            d_model: 512,
            num_heads: 8,
            d_ff: 2048,
            vocab_size: 4096,
            max_seq_len: 2048,
            n_enc: 6,
            n_dec: 6,
            layer_norm_eps: 1e-6,
            num_quantiles: DEFAULT_NUM_QUANTILES,
        }
    }

    pub fn t5_block_config(&self) -> T5BlockConfig {
        T5BlockConfig {
            d_model: self.d_model,
            d_ff: self.d_ff,
            num_heads: self.num_heads,
            dropout: 0.0,
            layer_norm_eps: self.layer_norm_eps,
        }
    }

    pub fn output_head_config(&self) -> OutputHeadConfig {
        OutputHeadConfig {
            d_model: self.d_model,
            num_quantiles: self.num_quantiles,
            layer_norm_eps: self.layer_norm_eps,
        }
    }
}

/// Errors raised by the loader.
#[derive(Debug, Error)]
pub enum CandleChronos2Error {
    #[error("config not found at {0}: expected a JSON file")]
    ConfigMissing(std::path::PathBuf),
    #[error("config parse error: {0}")]
    ConfigParse(#[from] serde_json::Error),
    #[error("config missing field: {0}")]
    ConfigField(&'static str),
    #[error("safetensors not found at {0}: expected *.safetensors")]
    WeightsMissing(std::path::PathBuf),
    #[error("safetensors load: {0}")]
    SafeTensors(String),
    #[error("runtime error: {0}")]
    Runtime(#[from] Chronos2Error),
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

/// Chronos2 candle runtime.
///
/// Holds the T5 encoder-decoder, multi-quantile output head,
/// and quantile tokenizer after loading from disk.
pub struct CandleChronos2Runtime {
    model_id: String,
    config: Chronos2Config,
    architecture: T5EncoderDecoder,
    head: Chronos2OutputHead,
    tokenizer: QuantileTokenizer,
    device: Device,
}

impl CandleChronos2Runtime {
    /// Load a model from a directory.
    pub fn load(model_dir: impl AsRef<Path>, device: Device) -> Result<Self, CandleChronos2Error> {
        let dir = model_dir.as_ref();
        let config_path = dir.join("config.json");
        if !config_path.exists() {
            return Err(CandleChronos2Error::ConfigMissing(config_path));
        }
        let config_text = std::fs::read_to_string(&config_path)?;
        let raw: serde_json::Value = serde_json::from_str(&config_text)?;
        let config = parse_config(&raw)?;
        let weights_path = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .find(|p| p.extension().map(|x| x == "safetensors").unwrap_or(false))
            .ok_or(CandleChronos2Error::WeightsMissing(dir.to_path_buf()))?;
        Self::load_with_paths(&config, weights_path, device)
    }

    /// Load from explicit config + weights path.
    pub fn load_with_paths(
        config: &Chronos2Config,
        weights_path: std::path::PathBuf,
        device: Device,
    ) -> Result<Self, CandleChronos2Error> {
        let safetensors_paths = vec![weights_path.clone()];
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&safetensors_paths, DType::F32, &device)
        }
        .map_err(|e| {
            CandleChronos2Error::SafeTensors(format!(
                "from_mmaped_safetensors {}: {e}",
                weights_path.display()
            ))
        })?;

        let block = config.t5_block_config();
        let embeddings = super::vendor::T5Embeddings {
            config: super::vendor::T5EmbeddingsConfig {
                vocab_size: config.vocab_size,
                d_model: config.d_model,
                max_seq_len: config.max_seq_len,
                learned: true,
            },
            token: vb.pp("token_emb").get_with_hints(
                (config.vocab_size, config.d_model),
                "weight",
                candle_nn::init::ZERO,
            ).map_err(CandleChronos2Error::Candle)?,
            position: vb.pp("pos_emb").get_with_hints(
                (config.max_seq_len, config.d_model),
                "weight",
                candle_nn::init::ZERO,
            ).map_err(CandleChronos2Error::Candle)?,
        };

        let arch = T5EncoderDecoder {
            encoder: super::vendor::T5Encoder::new(
                block.clone(),
                config.n_enc,
                embeddings.clone(),
                vb.pp("encoder"),
            )?,
            decoder: super::vendor::T5Decoder::new(
                block,
                config.n_dec,
                embeddings,
                vb.pp("decoder"),
            )?,
        };

        let head_cfg = config.output_head_config();
        let head = Chronos2OutputHead::new(head_cfg, vb.pp("head"))?;

        let tokenizer_config = QuantileTokenizerConfig::new(config.vocab_size);
        let tokenizer = QuantileTokenizer::new(tokenizer_config);

        Ok(Self {
            model_id: weights_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("chronos2")
                .to_string(),
            config: config.clone(),
            architecture: arch,
            head,
            tokenizer,
            device,
        })
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn config(&self) -> &Chronos2Config {
        &self.config
    }

    /// Build encoder token ids from the input series.
    fn build_encoder_ids(&self, series: &[f64]) -> Vec<u32> {
        self.tokenizer.tokenize(series)
    }

    /// Build decoder token ids: encoder tokens + horizon placeholders.
    fn build_decoder_ids(&self, series: &[f64], horizon: usize) -> Vec<u32> {
        self.tokenizer.tokenize_with_horizon(series, horizon)
    }

    /// Run batch inference on multiple time series.
    ///
    /// All series must use the same horizon and method. Returns
    /// a vector of forecasts, one per input series.
    pub async fn forecast_batch(
        &self,
        series_list: &[Vec<f64>],
        horizon: u32,
        method: ForecastMethod,
    ) -> Result<Vec<Vec<f64>>, super::Chronos2Error> {
        if series_list.is_empty() {
            return Ok(Vec::new());
        }

        let horizon = horizon.max(1) as usize;
        let batch_size = series_list.len();

        let mut all_encoder_ids = Vec::with_capacity(batch_size);
        let mut all_decoder_ids = Vec::with_capacity(batch_size);
        let mut max_enc_len = 0;
        let mut max_dec_len = 0;

        for series in series_list {
            let enc_ids = self.build_encoder_ids(series);
            let dec_ids = self.build_decoder_ids(series, horizon);
            max_enc_len = max_enc_len.max(enc_ids.len());
            max_dec_len = max_dec_len.max(dec_ids.len());
            all_encoder_ids.push(enc_ids);
            all_decoder_ids.push(dec_ids);
        }

        let pad_token = crate::chronos2::tokenizer::special_tokens::PAD;
        for ids in &mut all_encoder_ids {
            while ids.len() < max_enc_len {
                ids.push(pad_token);
            }
        }
        for ids in &mut all_decoder_ids {
            while ids.len() < max_dec_len {
                ids.push(pad_token);
            }
        }

        let enc_flat: Vec<u32> = all_encoder_ids.into_iter().flatten().collect();
        let dec_flat: Vec<u32> = all_decoder_ids.into_iter().flatten().collect();

        let encoder_tensor = Tensor::from_vec(
            enc_flat,
            (batch_size, max_enc_len),
            &self.device,
        ).map_err(|e| super::Chronos2Error::Runtime(e.to_string()))?;

        let decoder_tensor = Tensor::from_vec(
            dec_flat,
            (batch_size, max_dec_len),
            &self.device,
        ).map_err(|e| super::Chronos2Error::Runtime(e.to_string()))?;

        let enc_out = self.architecture.encoder.forward(&encoder_tensor)
            .map_err(|e| super::Chronos2Error::Runtime(e.to_string()))?;
        let dec_out = self.architecture.decoder.forward(&decoder_tensor, &enc_out.hidden_states)
            .map_err(|e| super::Chronos2Error::Runtime(e.to_string()))?;
        let cdf = self.head.forward(&dec_out.hidden_states)
            .map_err(|e| super::Chronos2Error::Runtime(e.to_string()))?;
        let horizon_cdf = take_horizon(&cdf, horizon as u32)
            .map_err(|e| super::Chronos2Error::Runtime(e.to_string()))?;

        let forecast_tensor = match method {
            ForecastMethod::Mean => self.head.extract_mean(&horizon_cdf)
                .map_err(|e| super::Chronos2Error::Runtime(e.to_string()))?,
            ForecastMethod::Median => self.head.extract_median(&horizon_cdf)
                .map_err(|e| super::Chronos2Error::Runtime(e.to_string()))?,
            ForecastMethod::Quantile => self.head.extract_median(&horizon_cdf)
                .map_err(|e| super::Chronos2Error::Runtime(e.to_string()))?,
        };

        let flat: Vec<f32> = forecast_tensor.flatten_all()
            .map_err(|e| super::Chronos2Error::Runtime(e.to_string()))?
            .to_vec1()
            .map_err(|e| super::Chronos2Error::Runtime(e.to_string()))?;

        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let start = i * horizon;
            let end = start + horizon;
            let forecast: Vec<f64> = flat[start..end].iter().map(|&f| f as f64).collect();
            results.push(forecast);
        }

        Ok(results)
    }
}

#[async_trait::async_trait]
impl Chronos2Runtime for CandleChronos2Runtime {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn forecast(
        &self,
        request: ForecastRequest,
    ) -> Result<ForecastResponse, super::Chronos2Error> {
        let started = std::time::Instant::now();
        let horizon = request.horizon.max(1) as usize;

        let encoder_ids = self.build_encoder_ids(&request.series);
        let decoder_ids = self.build_decoder_ids(&request.series, horizon);

        let encoder_tensor = Tensor::from_vec(
            encoder_ids.clone(),
            (1, encoder_ids.len()),
            &self.device,
        ).map_err(|e| super::Chronos2Error::Runtime(e.to_string()))?;

        let decoder_tensor = Tensor::from_vec(
            decoder_ids.clone(),
            (1, decoder_ids.len()),
            &self.device,
        ).map_err(|e| super::Chronos2Error::Runtime(e.to_string()))?;

        let method = request.method;
        let model_id = self.model_id.clone();

        let enc_out = self.architecture.encoder.forward(&encoder_tensor)
            .map_err(|e| super::Chronos2Error::Runtime(e.to_string()))?;
        let dec_out = self.architecture.decoder.forward(&decoder_tensor, &enc_out.hidden_states)
            .map_err(|e| super::Chronos2Error::Runtime(e.to_string()))?;
        let cdf = self.head.forward(&dec_out.hidden_states)
            .map_err(|e| super::Chronos2Error::Runtime(e.to_string()))?;
        let horizon_cdf = take_horizon(&cdf, horizon as u32)
            .map_err(|e| super::Chronos2Error::Runtime(e.to_string()))?;

        let forecast = match method {
            ForecastMethod::Mean => self.head.extract_mean(&horizon_cdf)
                .map_err(|e| super::Chronos2Error::Runtime(e.to_string()))?,
            ForecastMethod::Median => self.head.extract_median(&horizon_cdf)
                .map_err(|e| super::Chronos2Error::Runtime(e.to_string()))?,
            ForecastMethod::Quantile => self.head.extract_median(&horizon_cdf)
                .map_err(|e| super::Chronos2Error::Runtime(e.to_string()))?,
        };

        let v: Vec<f32> = forecast.flatten_all()
            .map_err(|e| super::Chronos2Error::Runtime(e.to_string()))?
            .to_vec1()
            .map_err(|e| super::Chronos2Error::Runtime(e.to_string()))?;
        let forecast: Vec<f64> = v.into_iter().map(|f| f as f64).collect();
        let took_ms = started.elapsed().as_millis() as u64;

        Ok(ForecastResponse {
            object: "list".to_string(),
            model: model_id,
            method: request.method,
            forecast,
            took_ms,
        })
    }
}

fn parse_config(raw: &serde_json::Value) -> Result<Chronos2Config, CandleChronos2Error> {
    let require_usize = |key: &'static str| -> Result<usize, CandleChronos2Error> {
        raw.get(key)
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .ok_or(CandleChronos2Error::ConfigField(key))
    };

    Ok(Chronos2Config {
        d_model: require_usize("d_model")?,
        num_heads: require_usize("num_heads")?,
        d_ff: require_usize("d_ff")?,
        vocab_size: require_usize("vocab_size")?,
        max_seq_len: require_usize("max_seq_len")?,
        n_enc: require_usize("n_enc")?,
        n_dec: require_usize("n_dec")?,
        layer_norm_eps: raw
            .get("layer_norm_eps")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-6),
        num_quantiles: raw
            .get("num_quantiles")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(DEFAULT_NUM_QUANTILES),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::TensorView;

    fn tempdir() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "chronos2-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_config(dir: &Path, cfg: &Chronos2Config) {
        let body = serde_json::json!({
            "d_model": cfg.d_model,
            "num_heads": cfg.num_heads,
            "d_ff": cfg.d_ff,
            "vocab_size": cfg.vocab_size,
            "max_seq_len": cfg.max_seq_len,
            "n_enc": cfg.n_enc,
            "n_dec": cfg.n_dec,
            "layer_norm_eps": cfg.layer_norm_eps,
            "num_quantiles": cfg.num_quantiles,
        });
        std::fs::write(dir.join("config.json"), body.to_string()).unwrap();
    }

    fn write_synthetic_safetensors(dir: &Path, cfg: &Chronos2Config) {
        use safetensors::Dtype as SD;
        use safetensors::tensor::Dtype;
        use std::collections::BTreeMap;
        let mut tensors: BTreeMap<String, (Dtype, Vec<usize>, Vec<f32>)> = BTreeMap::new();
        let mut insert = |name: &str, shape: &[usize], fill: f32| {
            let data = vec![fill; shape.iter().product()];
            tensors.insert(name.to_string(), (Dtype::F32, shape.to_vec(), data));
        };

        insert("token_emb.weight", &[cfg.vocab_size, cfg.d_model], 0.01);
        insert("pos_emb.weight", &[cfg.max_seq_len, cfg.d_model], 0.0);

        for enc in 0..cfg.n_enc {
            let p = format!("encoder.block_{enc}");
            insert(&format!("{p}.q.weight"), &[cfg.d_model, cfg.d_model], 0.0);
            insert(&format!("{p}.q.bias"), &[cfg.d_model], 0.0);
            insert(&format!("{p}.k.weight"), &[cfg.d_model, cfg.d_model], 0.0);
            insert(&format!("{p}.k.bias"), &[cfg.d_model], 0.0);
            insert(&format!("{p}.v.weight"), &[cfg.d_model, cfg.d_model], 0.0);
            insert(&format!("{p}.v.bias"), &[cfg.d_model], 0.0);
            insert(&format!("{p}.o.weight"), &[cfg.d_model, cfg.d_model], 0.0);
            insert(&format!("{p}.o.bias"), &[cfg.d_model], 0.0);
            insert(&format!("{p}.attn_norm.weight"), &[cfg.d_model], 1.0);
            insert(&format!("{p}.ffn_norm.weight"), &[cfg.d_model], 1.0);
            insert(&format!("{p}.ffn_up.weight"), &[cfg.d_ff, cfg.d_model], 0.0);
            insert(&format!("{p}.ffn_up.bias"), &[cfg.d_ff], 0.0);
            insert(&format!("{p}.ffn_down.weight"), &[cfg.d_model, cfg.d_ff], 0.0);
            insert(&format!("{p}.ffn_down.bias"), &[cfg.d_model], 0.0);
        }

        for dec in 0..cfg.n_dec {
            let p = format!("decoder.block_{dec}");
            insert(&format!("{p}.self_q.weight"), &[cfg.d_model, cfg.d_model], 0.0);
            insert(&format!("{p}.self_q.bias"), &[cfg.d_model], 0.0);
            insert(&format!("{p}.self_k.weight"), &[cfg.d_model, cfg.d_model], 0.0);
            insert(&format!("{p}.self_k.bias"), &[cfg.d_model], 0.0);
            insert(&format!("{p}.self_v.weight"), &[cfg.d_model, cfg.d_model], 0.0);
            insert(&format!("{p}.self_v.bias"), &[cfg.d_model], 0.0);
            insert(&format!("{p}.self_o.weight"), &[cfg.d_model, cfg.d_model], 0.0);
            insert(&format!("{p}.self_o.bias"), &[cfg.d_model], 0.0);
            insert(&format!("{p}.self_attn_norm.weight"), &[cfg.d_model], 1.0);
            insert(&format!("{p}.ffn_norm.weight"), &[cfg.d_model], 1.0);
            insert(&format!("{p}.ffn_up.weight"), &[cfg.d_ff, cfg.d_model], 0.0);
            insert(&format!("{p}.ffn_up.bias"), &[cfg.d_ff], 0.0);
            insert(&format!("{p}.ffn_down.weight"), &[cfg.d_model, cfg.d_ff], 0.0);
            insert(&format!("{p}.ffn_down.bias"), &[cfg.d_model], 0.0);
            insert(&format!("{p}.k_enc.weight"), &[cfg.d_model, cfg.d_model], 0.0);
            insert(&format!("{p}.k_enc.bias"), &[cfg.d_model], 0.0);
            insert(&format!("{p}.v_enc.weight"), &[cfg.d_model, cfg.d_model], 0.0);
            insert(&format!("{p}.v_enc.bias"), &[cfg.d_model], 0.0);
            insert(&format!("{p}.cross_attn_norm.weight"), &[cfg.d_model], 1.0);
        }

        let nq = cfg.num_quantiles;
        insert("head.final_norm.weight", &[cfg.d_model], 1.0);
        insert("head.proj.weight", &[nq, cfg.d_model], 0.0);
        insert("head.proj.bias", &[nq], 0.0);
        insert("head.bias", &[nq], 0.0);

        let views: Vec<(&str, TensorView<'_>)> = tensors
            .iter()
            .map(|(name, (_dtype, shape, data))| {
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        data.as_ptr() as *const u8,
                        data.len() * std::mem::size_of::<f32>(),
                    )
                };
                (name.as_str(), TensorView::new(SD::F32, shape.clone(), bytes).unwrap())
            })
            .collect();
        let serialized = safetensors::serialize(views, &None).unwrap();
        std::fs::write(dir.join("model.safetensors"), &serialized).unwrap();
    }

    fn build_synthetic_checkpoint() -> std::path::PathBuf {
        let dir = tempdir();
        let cfg = Chronos2Config {
            d_model: 8,
            num_heads: 2,
            d_ff: 16,
            vocab_size: 8,
            max_seq_len: 32,
            n_enc: 1,
            n_dec: 1,
            layer_norm_eps: 1e-6,
            num_quantiles: 4,
        };
        write_config(&dir, &cfg);
        write_synthetic_safetensors(&dir, &cfg);
        dir
    }

    #[test]
    fn parse_config_extracts_all_fields() {
        let raw = serde_json::json!({
            "d_model": 64, "num_heads": 4, "d_ff": 128,
            "vocab_size": 32, "max_seq_len": 16, "n_enc": 2, "n_dec": 2,
            "num_quantiles": 32,
        });
        let cfg = parse_config(&raw).unwrap();
        assert_eq!(cfg.d_model, 64);
        assert_eq!(cfg.num_heads, 4);
        assert_eq!(cfg.d_ff, 128);
        assert_eq!(cfg.vocab_size, 32);
        assert_eq!(cfg.max_seq_len, 16);
        assert_eq!(cfg.n_enc, 2);
        assert_eq!(cfg.n_dec, 2);
        assert_eq!(cfg.num_quantiles, 32);
    }

    #[test]
    fn parse_config_defaults_num_quantiles() {
        let raw = serde_json::json!({
            "d_model": 64, "num_heads": 4, "d_ff": 128,
            "vocab_size": 32, "max_seq_len": 16, "n_enc": 2, "n_dec": 2,
        });
        let cfg = parse_config(&raw).unwrap();
        assert_eq!(cfg.num_quantiles, DEFAULT_NUM_QUANTILES);
    }

    #[test]
    fn parse_config_missing_field_returns_err() {
        let raw = serde_json::json!({ "d_model": 64 });
        assert!(matches!(
            parse_config(&raw).unwrap_err(),
            CandleChronos2Error::ConfigField("num_heads")
        ));
    }

    #[test]
    fn default_bolt_small_config_is_valid() {
        let cfg = Chronos2Config::default_bolt_small();
        let t5 = cfg.t5_block_config();
        assert_eq!(t5.d_model, 512);
        assert!(t5.validate().is_ok());
        assert_eq!(cfg.num_quantiles, DEFAULT_NUM_QUANTILES);
    }

    #[test]
    fn t5_block_config_from_model_config() {
        let cfg = Chronos2Config {
            d_model: 256,
            num_heads: 8,
            d_ff: 1024,
            vocab_size: 1024,
            max_seq_len: 512,
            n_enc: 4,
            n_dec: 4,
            layer_norm_eps: 1e-5,
            num_quantiles: 64,
        };
        let t5 = cfg.t5_block_config();
        assert_eq!(t5.d_model, 256);
        assert_eq!(t5.d_ff, 1024);
        assert_eq!(t5.num_heads, 8);
    }

    #[tokio::test]
    async fn runtime_loads_and_forecasts() {
        let dir = build_synthetic_checkpoint();
        let rt = tokio::task::spawn_blocking({
            let dir = dir.clone();
            move || CandleChronos2Runtime::load(&dir, Device::Cpu)
        })
        .await
        .unwrap()
        .expect("load");

        assert_eq!(rt.model_id(), "model");

        let req = ForecastRequest {
            model: "model".into(),
            series: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            horizon: 3,
            method: ForecastMethod::Mean,
        };
        let resp = rt.forecast(req).await.expect("forecast");
        assert_eq!(resp.method, ForecastMethod::Mean);
        assert_eq!(resp.forecast.len(), 3);
        assert!(resp.took_ms < 60_000);
    }

    #[tokio::test]
    async fn runtime_forecast_median() {
        let dir = build_synthetic_checkpoint();
        let rt = tokio::task::spawn_blocking({
            let dir = dir.clone();
            move || CandleChronos2Runtime::load(&dir, Device::Cpu)
        })
        .await
        .unwrap()
        .expect("load");

        let req = ForecastRequest {
            model: "model".into(),
            series: vec![1.0, 2.0, 3.0],
            horizon: 2,
            method: ForecastMethod::Median,
        };
        let resp = rt.forecast(req).await.expect("forecast");
        assert_eq!(resp.forecast.len(), 2);
    }

    #[tokio::test]
    async fn runtime_empty_series() {
        let dir = build_synthetic_checkpoint();
        let rt = tokio::task::spawn_blocking({
            let dir = dir.clone();
            move || CandleChronos2Runtime::load(&dir, Device::Cpu)
        })
        .await
        .unwrap()
        .expect("load");

        let req = ForecastRequest {
            model: "model".into(),
            series: vec![],
            horizon: 1,
            method: ForecastMethod::Mean,
        };
        let resp = rt.forecast(req).await.expect("forecast");
        assert_eq!(resp.forecast.len(), 1);
    }

    #[tokio::test]
    async fn runtime_single_value_series() {
        let dir = build_synthetic_checkpoint();
        let rt = tokio::task::spawn_blocking({
            let dir = dir.clone();
            move || CandleChronos2Runtime::load(&dir, Device::Cpu)
        })
        .await
        .unwrap()
        .expect("load");

        let req = ForecastRequest {
            model: "model".into(),
            series: vec![42.0],
            horizon: 1,
            method: ForecastMethod::Mean,
        };
        let resp = rt.forecast(req).await.expect("forecast");
        assert_eq!(resp.forecast.len(), 1);
    }

    #[tokio::test]
    async fn runtime_large_horizon() {
        let dir = build_synthetic_checkpoint();
        let rt = tokio::task::spawn_blocking({
            let dir = dir.clone();
            move || CandleChronos2Runtime::load(&dir, Device::Cpu)
        })
        .await
        .unwrap()
        .expect("load");

        let req = ForecastRequest {
            model: "model".into(),
            series: vec![1.0, 2.0, 3.0],
            horizon: 10,
            method: ForecastMethod::Mean,
        };
        let resp = rt.forecast(req).await.expect("forecast");
        assert_eq!(resp.forecast.len(), 10);
    }

    #[test]
    fn build_encoder_ids_uses_quantile_tokenizer() {
        let dir = build_synthetic_checkpoint();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let rt = rt.block_on(async {
            tokio::task::spawn_blocking({
                let dir = dir.clone();
                move || CandleChronos2Runtime::load(&dir, Device::Cpu)
            })
            .await
            .unwrap()
            .expect("load")
        });

        let ids = rt.build_encoder_ids(&[1.0, 2.0, 3.0]);
        // BOS + 3 tokens.
        assert_eq!(ids.len(), 4);
        assert_eq!(ids[0], 0); // BOS
    }

    #[test]
    fn build_decoder_ids_includes_horizon() {
        let dir = build_synthetic_checkpoint();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let rt = rt.block_on(async {
            tokio::task::spawn_blocking({
                let dir = dir.clone();
                move || CandleChronos2Runtime::load(&dir, Device::Cpu)
            })
            .await
            .unwrap()
            .expect("load")
        });

        let ids = rt.build_decoder_ids(&[1.0, 2.0, 3.0], 5);
        // BOS + 3 tokens + 5 placeholders = 9.
        assert_eq!(ids.len(), 9);
    }

    #[test]
    fn config_output_head_config() {
        let cfg = Chronos2Config::default_bolt_small();
        let head_cfg = cfg.output_head_config();
        assert_eq!(head_cfg.d_model, 512);
        assert_eq!(head_cfg.num_quantiles, DEFAULT_NUM_QUANTILES);
    }

    #[tokio::test]
    async fn runtime_forecast_batch() {
        let dir = build_synthetic_checkpoint();
        let rt = tokio::task::spawn_blocking({
            let dir = dir.clone();
            move || CandleChronos2Runtime::load(&dir, Device::Cpu)
        })
        .await
        .unwrap()
        .expect("load");

        let series_list = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let results = rt.forecast_batch(&series_list, 2, ForecastMethod::Mean).await.expect("batch");
        assert_eq!(results.len(), 3);
        for forecast in &results {
            assert_eq!(forecast.len(), 2);
        }
    }

    #[tokio::test]
    async fn runtime_forecast_batch_empty() {
        let dir = build_synthetic_checkpoint();
        let rt = tokio::task::spawn_blocking({
            let dir = dir.clone();
            move || CandleChronos2Runtime::load(&dir, Device::Cpu)
        })
        .await
        .unwrap()
        .expect("load");

        let series_list: Vec<Vec<f64>> = vec![];
        let results = rt.forecast_batch(&series_list, 2, ForecastMethod::Mean).await.expect("batch");
        assert_eq!(results.len(), 0);
    }

    #[tokio::test]
    async fn runtime_forecast_batch_different_lengths() {
        let dir = build_synthetic_checkpoint();
        let rt = tokio::task::spawn_blocking({
            let dir = dir.clone();
            move || CandleChronos2Runtime::load(&dir, Device::Cpu)
        })
        .await
        .unwrap()
        .expect("load");

        let series_list = vec![
            vec![1.0],
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
        ];
        let results = rt.forecast_batch(&series_list, 3, ForecastMethod::Median).await.expect("batch");
        assert_eq!(results.len(), 2);
        for forecast in &results {
            assert_eq!(forecast.len(), 3);
        }
    }
}
