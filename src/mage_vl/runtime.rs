//! Engine-level LLM runtime that wraps the Microsoft Mage-VL
//! multimodal model. Hosts the same checkpoint as
//! `MageVLModel::load_safetensors` but speaks the new
//! `engine::LlmRuntime` trait (`execute` / `execute_stream` /
//! `capabilities` / `name`) so cortex can dispatch chat requests
//! to it via `state.engines.llm`.
//!
//! Vision tower: mage_vl::vision::MageViT (our candle port) for
//! standalone vision tasks and the StreamMind gate forward. Text
//! tower: candle's `qwen3_vl::Qwen3VLModel` because the model is
//! already bundled there and `Qwen3VLTextModel` is private. The
//! trade-off is that text forward goes through candle's vision
//! encoder; MageVLModel is kept for vision-only / gate work.

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use candle_core::{DType, Device, Result, Tensor};
use futures_util::stream::{BoxStream, StreamExt};
use tokenizers::Tokenizer;
use tokio::sync::OnceCell;

use crate::canonical::{
    CanonicalLlmRequest, CanonicalLlmResponse, CanonicalLlmStreamEvent, ContentBlock,
    StopReason, TextBlock, UsageInfo,
};
use crate::capabilities::{Modality, ModelCapabilities};
use crate::config::DeviceSelection;
use crate::engine::LlmRuntime;
use crate::llm::NeuronError;

use super::config::MageVLConfig;

pub struct MageVlRuntime {
    config: MageVLConfig,
    shard_dir: PathBuf,
    /// Cached model. Loaded lazily on first `execute` so the host
    /// can register the runtime without paying the 3.5GB text
    /// checkpoint cost until a chat request actually arrives.
    model: OnceCell<Arc<MageVlCachedModel>>,
}

struct MageVlCachedModel {
    /// The full multimodal model (vision + text + scatter all
    /// inside). `MageVLModel::forward_multimodal` is a public
    /// wrapper around the underlying candle `Qwen3VLModel::forward`
    /// so we never reach into private candle state.
    model: Arc<super::MageVLModel>,
    device: Device,
    _device_sel: DeviceSelection,
    /// Qwen3 BPE tokenizer loaded from `{shard_dir}/tokenizer.json`.
    /// `None` only when the checkpoint dir lacks the file — encode
    /// falls back to byte-level ids and decode emits a sentinel.
    tokenizer: Option<Tokenizer>,
}

impl MageVlRuntime {
    pub fn new(config: MageVLConfig, shard_dir: impl Into<PathBuf>) -> Self {
        Self {
            config,
            shard_dir: shard_dir.into(),
            model: OnceCell::new(),
        }
    }

    async fn ensure_loaded(&self) -> std::result::Result<Arc<MageVlCachedModel>, NeuronError> {
        self.model
            .get_or_try_init(|| async {
                let (device, dtype) = resolve_device();
                let shard1 = self.shard_dir.join("model-00001-of-00002.safetensors");
                let shard2 = self.shard_dir.join("model-00002-of-00002.safetensors");
                if !shard1.exists() || !shard2.exists() {
                    return Err(NeuronError::new(format!(
                        "MageVlRuntime: missing shards at {} (need both \
                         model-00001-of-00002.safetensors and \
                         model-00002-of-00002.safetensors)",
                        self.shard_dir.display()
                    )));
                }
                let model = super::MageVLModel::load_safetensors(
                    &[shard1.as_path(), shard2.as_path()],
                    self.config.clone(),
                    dtype,
                    &device,
                )
                .map_err(|e| {
                    NeuronError::new(format!("MageVLModel::load_safetensors failed: {e}"))
                })?;
                let tokenizer = load_tokenizer_from(&self.shard_dir);
                if tokenizer.is_none() {
                    tracing::warn!(
                        "mage-vl: no tokenizer.json under {}; text encode \
                         degrades to byte-level ids",
                        self.shard_dir.display()
                    );
                }
                Ok(Arc::new(MageVlCachedModel {
                    model: Arc::new(model),
                    device,
                    _device_sel: DeviceSelection::Cpu,
                    tokenizer,
                }))
            })
            .await
            .cloned()
    }

    fn capabilities() -> &'static ModelCapabilities {
        use std::sync::OnceLock;
        static CAPS: OnceLock<ModelCapabilities> = OnceLock::new();
        CAPS.get_or_init(|| ModelCapabilities {
            engine_impl: "mage_vl".into(),
            model_id: "microsoft/Mage-VL".into(),
            input_modalities: [Modality::TextInput, Modality::ImageInput]
                .into_iter()
                .collect(),
            output_modalities: [Modality::TextOutput].into_iter().collect(),
            api_styles: [crate::capabilities::ApiStyle::openai_chat()]
                .into_iter()
                .collect(),
            supports_streaming: true,
            supports_tools: false,
            supports_prompt_caching: false,
            supports_extended_thinking: false,
            context_window: Some(262_144),
            max_output_tokens: Some(8_192),
        })
    }
}

fn resolve_device() -> (Device, DType) {
    // Every gated Mage-VL test on this machine has run on CPU.
    // Candle's CPU backend has no BF16 matmul kernel ("unsupported
    // dtype BF16 for op matmul"), so the BF16 shards are upcast to
    // F32 at load time (~18 GB resident for the 4B text tower —
    // sized for a 32 GB host; CUDA / Metal / quantization are
    // follow-ups).
    (Device::Cpu, DType::F32)
}

#[async_trait]
impl LlmRuntime for MageVlRuntime {
    async fn execute(
        &self,
        req: CanonicalLlmRequest,
    ) -> crate::llm::ChatResult<CanonicalLlmResponse> {
        let cached = self.ensure_loaded().await?;
        // Drop KV entries the previous request left behind — candle's
        // text tower appends forever and never resets on its own.
        // Safe to call before the first request (no-op on an empty
        // cache). Serializes concurrent requests on the same model.
        cached.model.reset_cache();
        let (input_ids, pixel_values, image_grid_thw) =
            build_qwen3_inputs(&req, &cached.device, &cached.tokenizer)
                .await
                .map_err(|e| NeuronError::new(format!("build mage-vl inputs: {e}")))?;

        // `<|im_end|>` is Qwen3's EOS. Fall back to the known id if
        // the tokenizer is absent.
        let eos_token_id = cached
            .tokenizer
            .as_ref()
            .and_then(|t| t.token_to_id("<|im_end|>"))
            .unwrap_or(151_645);
        let max_new_tokens = req
            .sampling
            .max_tokens
            .map(|n| (n as usize).min(512))
            .unwrap_or(256);

        let generated = cached
            .model
            .generate(
                &input_ids,
                pixel_values.as_ref(),
                image_grid_thw.as_ref(),
                max_new_tokens,
                eos_token_id,
            )
            .map_err(|e| NeuronError::new(format!("mage-vl generate: {e}")))?;

        let text = match &cached.tokenizer {
            Some(t) if !generated.is_empty() => t
                .decode(&generated, true)
                .map_err(|e| NeuronError::new(format!("mage-vl decode: {e}")))?,
            _ => String::new(),
        };

        Ok(CanonicalLlmResponse {
            id: format!("mage-vl-{}", uuid::Uuid::new_v4()),
            model: req.model,
            stop_reason: StopReason::EndTurn,
            content: vec![ContentBlock::Text(TextBlock { text })],
            usage: UsageInfo::default(),
            extensions: Default::default(),
        })
    }

    async fn execute_stream(
        &self,
        req: CanonicalLlmRequest,
    ) -> crate::llm::ChatResult<BoxStream<'static, CanonicalLlmStreamEvent>> {
        let resp = self.execute(req.clone()).await?;
        let model = req.model.clone();
        let stream = futures_util::stream::once(async move {
            let mut events = Vec::new();
            events.push(CanonicalLlmStreamEvent::MessageStart {
                message: {
                    let mut r = resp;
                    r.model = model;
                    r
                },
            });
            events.push(CanonicalLlmStreamEvent::MessageStop);
            futures_util::stream::iter(events)
        })
        .flatten();
        Ok(Box::pin(stream))
    }

    fn capabilities(&self) -> &ModelCapabilities {
        Self::capabilities()
    }

    fn name(&self) -> &str {
        "mage_vl"
    }
}

/// Translate a `CanonicalLlmRequest` into the inputs candle's
/// `Qwen3VLModel::forward` expects: `(input_ids, pixel_values?,
/// image_grid_thw?)`. The prompt is rendered with the Qwen3 chat
/// template (`<|im_start|>` turns, `<|vision_start|><|image_pad|>×N
/// <|vision_end|>` for images) and encoded with the real BPE
/// tokenizer when available.
async fn build_qwen3_inputs(
    req: &CanonicalLlmRequest,
    device: &Device,
    tokenizer: &Option<Tokenizer>,
) -> Result<(Tensor, Option<Tensor>, Option<Tensor>)> {
    use crate::canonical::{ContentBlock, MessageRole};

    let mut system_text = String::new();
    let mut user_texts: Vec<&str> = Vec::new();
    let mut image: Option<crate::canonical::ImageBlock> = None;

    for msg in &req.messages {
        if matches!(msg.role, MessageRole::System) && system_text.is_empty() {
            if let Some(ContentBlock::Text(t)) = msg.content.first() {
                system_text = t.text.clone();
            }
            continue;
        }
        if matches!(msg.role, MessageRole::User) {
            for block in &msg.content {
                match block {
                    ContentBlock::Text(t) => user_texts.push(&t.text),
                    ContentBlock::Image(img) => {
                        // Mage-VL v1 path: one image per request.
                        if image.is_none() {
                            image = Some(img.clone());
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    // Decode + preprocess the image FIRST so the prompt can carry
    // exactly N = (h/32)*(w/32) `<|image_pad|>` placeholders.
    let preprocessed = match &image {
        Some(img) => decode_image_block(img, device).await?,
        None => None,
    };
    let n_pad_tokens = preprocessed
        .as_ref()
        .map(|(_, grid)| {
            let g = grid
                .flatten_all()
                .and_then(|t| t.to_vec1::<u32>())
                .unwrap_or_default();
            let h = g.get(1).copied().unwrap_or(0) as usize;
            let w = g.get(2).copied().unwrap_or(0) as usize;
            (h / 2) * (w / 2)
        })
        .unwrap_or(0);

    let prompt = render_chat_template(
        if system_text.is_empty() { "You are a helpful assistant." } else { &system_text },
        &user_texts,
        n_pad_tokens,
    );
    let token_ids = encode_prompt(&prompt, tokenizer);
    let input_ids = Tensor::from_vec(token_ids.clone(), (1usize, token_ids.len()), device)?;

    let (pixel_values, image_grid_thw) = match preprocessed {
        Some((p, g)) => (Some(p), Some(g)),
        None => (None, None),
    };
    Ok((input_ids, pixel_values, image_grid_thw))
}

/// Render the Qwen3 chat template for a single-turn exchange:
/// system turn, user turn (with optional image placeholders),
/// then an open assistant turn for generation.
fn render_chat_template(system: &str, user_texts: &[&str], n_pad_tokens: usize) -> String {
    const IMAGE_PAD: &str = "<|image_pad|>";
    let mut s = String::with_capacity(256 + user_texts.iter().map(|t| t.len()).sum::<usize>());
    s.push_str("<|im_start|>system\n");
    s.push_str(system);
    s.push_str("<|im_end|>\n");
    s.push_str("<|im_start|>user\n");
    if n_pad_tokens > 0 {
        s.push_str("<|vision_start|>");
        for _ in 0..n_pad_tokens {
            s.push_str(IMAGE_PAD);
        }
        s.push_str("<|vision_end|>");
    }
    for t in user_texts {
        s.push_str(t);
        s.push('\n');
    }
    s.push_str("<|im_end|>\n");
    s.push_str("<|im_start|>assistant\n");
    s
}

/// Decode one `ImageSource::Base64` into the (pixel_values,
/// image_grid_thw) pair candle expects, using the real Qwen
/// image preprocessing pipeline:
/// RGB → resize to multiples of 32 (patch 16 × merge 2) →
/// scale to [0,1] → CLIP-stats normalize → channel-first
/// `[1, 3, 1, H, W]` with grid `[1, h/16, w/16]` (patchify is the
/// strided conv inside MageViT's PatchEmbed).
/// Returns `None` for URLs/FileIds (no fetcher wired in).
async fn decode_image_block(
    block: &crate::canonical::ImageBlock,
    device: &Device,
) -> Result<Option<(Tensor, Tensor)>> {
    use crate::canonical::ImageSource;
    match &block.source {
        ImageSource::Base64 { media_type, data } => {
            let _ = media_type;
            let b64 = data
                .rsplit_once(',')
                .map(|(_, b)| b)
                .unwrap_or(data.as_str());
            let bytes = base64_decode(b64)
                .map_err(|e| candle_core::Error::msg(format!("base64 decode: {e}")))?;
            let (pixels, grid) = preprocess_image_qwen(&bytes, device)?;
            Ok(Some((pixels, grid)))
        }
        ImageSource::Url(_) | ImageSource::FileId(_) => {
            tracing::warn!(
                "mage-vl runtime: URL/FileId image sources are not fetched yet; \
                 running text-only forward"
            );
            Ok(None)
        }
    }
}

const PATCH_SIZE: usize = 16;
const SPATIAL_MERGE: usize = 2;
// CLIP stats used by every Qwen-VL-family image processor.
const PIXEL_MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
const PIXEL_STD: [f32; 3] = [0.268_629_54, 0.261_302_58, 0.275_777_11];

fn preprocess_image_qwen(bytes: &[u8], device: &Device) -> Result<(Tensor, Tensor)> {
    use image::ImageReader;
    use std::io::Cursor;

    let img = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| candle_core::Error::msg(format!("image sniff: {e}")))?
        .decode()
        .map_err(|e| candle_core::Error::msg(format!("image decode: {e}")))?;
    let rgb = img.to_rgb8();

    let (raw_w, raw_h) = (rgb.width() as usize, rgb.height() as usize);
    // Resize so both dims are multiples of patch*merge (=28).
    let unit = PATCH_SIZE * SPATIAL_MERGE;
    let round_to_unit = |v: usize| ((v.max(unit) as f32 / unit as f32).round() as usize).max(1) * unit;
    let target_w = round_to_unit(raw_w);
    let target_h = round_to_unit(raw_h);
    let resized = image::imageops::resize(&rgb, target_w as u32, target_h as u32, image::imageops::FilterType::Lanczos3);

    let gh = target_h / PATCH_SIZE;
    let gw = target_w / PATCH_SIZE;

    // Channel-first normalize: [3, H, W], then prepend the
    // singleton temporal axis — exactly what MageViT's
    // `PatchEmbed::forward` expects (`[n, c, t, h, w]`, patchify is
    // the internal strided conv).
    let mut norm = vec![0f32; 3 * target_h * target_w];
    let plane = target_h * target_w;
    for y in 0..target_h {
        for x in 0..target_w {
            let px = resized.get_pixel(x as u32, y as u32);
            norm[px_index(0, y, x, plane, target_w)] =
                (px[0] as f32 / 255.0 - PIXEL_MEAN[0]) / PIXEL_STD[0];
            norm[px_index(1, y, x, plane, target_w)] =
                (px[1] as f32 / 255.0 - PIXEL_MEAN[1]) / PIXEL_STD[1];
            norm[px_index(2, y, x, plane, target_w)] =
                (px[2] as f32 / 255.0 - PIXEL_MEAN[2]) / PIXEL_STD[2];
        }
    }

    let pixels = Tensor::from_vec(norm, (1usize, 3usize, 1usize, target_h, target_w), device)?;
    let grid = Tensor::from_slice(&[1u32, gh as u32, gw as u32], (1usize, 3usize), device)?;
    Ok((pixels, grid))
}

fn px_index(c: usize, y: usize, x: usize, plane: usize, width: usize) -> usize {
    c * plane + y * width + x
}

fn base64_decode(s: &str) -> std::result::Result<Vec<u8>, String> {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD
        .decode(s)
        .map_err(|e| e.to_string())
}

/// Encode a prompt with the real Qwen3 BPE tokenizer when the
/// checkpoint ships one; otherwise fall back to a byte-level
/// mapping (byte → id = byte+1, 0 reserved) that keeps the wire
/// alive but produces meaningless generations.
fn encode_prompt(prompt: &str, tokenizer: &Option<Tokenizer>) -> Vec<u32> {
    if let Some(tok) = tokenizer {
        match tok.encode(prompt, false) {
            Ok(enc) => return enc.get_ids().to_vec(),
            Err(e) => tracing::warn!("mage-vl tokenize failed, byte fallback: {e}"),
        }
    }
    prompt
        .as_bytes()
        .iter()
        .map(|b| (*b as u32).saturating_add(1))
        .collect()
}

/// Load the Qwen3 BPE tokenizer shipped alongside the shards.
/// `tokenizer.json` is the single authoritative file — vocab.json
/// and merges.txt are redundant derivatives for slow-tokenizer
/// consumers and are not read here.
fn load_tokenizer_from(shard_dir: &std::path::Path) -> Option<Tokenizer> {
    let path = shard_dir.join("tokenizer.json");
    if !path.exists() {
        return None;
    }
    Tokenizer::from_file(&path).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canonical::{CanonicalMessage, MessageRole, TextBlock};

    #[test]
    fn encode_prompt_byte_level_is_deterministic() {
        let ids = encode_prompt("hi", &None);
        assert_eq!(ids, vec![b'h' as u32 + 1, b'i' as u32 + 1]);
    }

    #[test]
    fn render_chat_template_has_vision_tokens_and_assistant_turn() {
        let p = render_chat_template("You are helpful.", &["describe"], 4);
        assert!(p.starts_with("<|im_start|>system\nYou are helpful.<|im_end|>\n"));
        assert!(p.contains("<|vision_start|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|vision_end|>"));
        assert!(p.contains("describe\n<|im_end|>\n"));
        assert!(p.ends_with("<|im_start|>assistant\n"));

        // Text-only: no vision block at all.
        let p2 = render_chat_template("sys", &["hello"], 0);
        assert!(!p2.contains("<|vision_start|>"));
    }

    /// Real BPE round-trip when the checkpoint's tokenizer.json is
    /// reachable from this crate (dev machines with the OpenSource
    /// checkout); skipped silently otherwise so CI stays green.
    #[test]
    fn tokenizer_roundtrip_when_checkpoint_present() {
        let dir = std::path::PathBuf::from(
            crate::config::env_var_result("MAGE_VL_SHARD_DIR")
                .unwrap_or_else(|_| "../OpenSourceProjects/mage/mage-vl-checkpoint".into()),
        );
        let Some(tok) = load_tokenizer_from(&dir) else {
            return;
        };
        let ids = encode_prompt("<|im_start|>user\nhi<|im_end|>\n", &Some(tok.clone()));
        // Special tokens must map to their real Qwen3 ids.
        assert!(ids.contains(&151_644), "<|im_start|> id missing");
        assert!(ids.contains(&151_645), "<|im_end|> id missing");
        // Round-trip: keep specials for the exact string; stripping
        // them leaves only the plain text.
        let text = tok.decode(&ids, false).unwrap();
        assert_eq!(text, "<|im_start|>user\nhi<|im_end|>\n");
        let bare = tok.decode(&ids, true).unwrap();
        assert_eq!(bare, "user\nhi\n");
    }

    #[test]
    fn preprocess_image_qwen_shapes_and_normalization() {
        // 64x64 red image → resized stays 64x64 (already a multiple of 28? no:
        // 64 → rounds to 56 or 84; both divisible by 28).
        let mut png = Vec::new();
        image::RgbaImage::from_pixel(64, 64, image::Rgba([255u8, 0, 0, 255]))
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        let (pixels, grid) = preprocess_image_qwen(&png, &Device::Cpu).unwrap();
        let g = grid
            .flatten_all()
            .and_then(|t| t.to_vec1::<u32>())
            .unwrap();
        let (_t, gh, gw) = (g[0], g[1], g[2]);
        assert_eq!(gh * PATCH_SIZE as u32 % 32, 0);
        assert_eq!(gw * PATCH_SIZE as u32 % 32, 0);
        // 5D image tensor: [n=1, c=3, t=1, H, W] — MageViT's
        // PatchEmbed consumes this and does the patchify itself.
        assert_eq!(pixels.dims(), &[1usize, 3usize, 1usize, gh as usize * PATCH_SIZE, gw as usize * PATCH_SIZE]);
        let v = pixels
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .unwrap();
        // Red channel ≈ +2.07; green plane ≈ -1.75.
        assert!(v[0] > 0.5);
        let plane = pixels.dim(4).unwrap() * pixels.dim(3).unwrap();
        assert!(v[plane] < -0.8);
    }

    #[test]
    fn name_reports_mage_vl() {
        let rt = MageVlRuntime::new(MageVLConfig::default(), "/tmp");
        assert_eq!(rt.name(), "mage_vl");
    }

    #[test]
    fn capabilities_declares_image_input_and_streaming() {
        let caps = MageVlRuntime::capabilities();
        assert!(caps.input_modalities.contains(&Modality::ImageInput));
        assert!(caps.input_modalities.contains(&Modality::TextInput));
        assert!(caps.output_modalities.contains(&Modality::TextOutput));
        assert!(caps.supports_streaming);
        assert!(!caps.supports_tools);
    }

    #[tokio::test]
    async fn build_qwen3_inputs_handles_text_only_request() {
        let req = CanonicalLlmRequest {
            model: "microsoft/Mage-VL".into(),
            messages: vec![CanonicalMessage {
                role: MessageRole::User,
                content: vec![ContentBlock::Text(TextBlock {
                    text: "describe the image".into(),
                })],
                reasoning_content: None,
                tool_call_id: None,
                tool_calls: vec![],
            }],
            system: vec![],
            tools: vec![],
            sampling: Default::default(),
            stop_sequences: vec![],
            stream: false,
            cache_control: None,
            metadata: Default::default(),
            extensions: Default::default(),
        };
        let device = Device::Cpu;
        let (ids, pixels, grid) = build_qwen3_inputs(&req, &device, &None)
            .await
            .unwrap();
        // No image attached → vision tensors stay None and the
        // template still opens the assistant turn for generation.
        assert!(pixels.is_none());
        assert!(grid.is_none());
        let (seq_len, ) = (ids.dims()[1],);
        assert!(seq_len > 0);
    }
}
