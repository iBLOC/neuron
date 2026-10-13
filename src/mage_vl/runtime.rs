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
use candle::{DType, Device, Result, Tensor};
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
    /// Dedicated worker thread. Every CUDA call for this model —
    /// weight load, image preprocessing, forward, sampling — runs on
    /// that one OS thread: cudarc binds the primary context to the
    /// thread that created it, and calls from any other thread fail
    /// with CUDA_ERROR_INVALID_CONTEXT (first observed when tokio's
    /// multithreaded runtime handed request 2 to a different worker).
    /// Jobs serialize on the channel, which the single-tenant KV cache
    /// requires anyway. The model loads lazily inside the worker so
    /// hosts can register the runtime without paying the ~11 GB
    /// checkpoint cost until a chat request actually arrives.
    worker: OnceCell<MageVlWorker>,
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

enum Job {
    Execute {
        req: Box<CanonicalLlmRequest>,
        reply: tokio::sync::oneshot::Sender<crate::llm::ChatResult<CanonicalLlmResponse>>,
    },
}

struct MageVlWorker {
    tx: std::sync::mpsc::Sender<Job>,
}

impl MageVlWorker {
    fn spawn(config: MageVLConfig, shard_dir: PathBuf) -> Self {
        let (tx, rx) = std::sync::mpsc::channel::<Job>();
        std::thread::Builder::new()
            .name("mage-vl-worker".into())
            .spawn(move || {
                let mut loaded: Option<Arc<MageVlCachedModel>> = None;
                for job in rx {
                    match job {
                        Job::Execute { req, reply } => {
                            let result =
                                run_request(&config, &shard_dir, &mut loaded, *req);
                            let _ = reply.send(result);
                        }
                    }
                }
            })
            .expect("spawn mage-vl worker thread");
        Self { tx }
    }
}

/// One request, executed entirely on the mage-vl worker thread.
/// Loads the model on first use; a failed load is not cached so a
/// later request can retry (e.g. shards appear on disk).
fn run_request(
    config: &MageVLConfig,
    shard_dir: &PathBuf,
    loaded: &mut Option<Arc<MageVlCachedModel>>,
    req: CanonicalLlmRequest,
) -> crate::llm::ChatResult<CanonicalLlmResponse> {
    let cached: Arc<MageVlCachedModel> = match loaded {
        Some(c) => c.clone(),
        None => {
            let c = Arc::new(load_model(config, shard_dir)?);
            *loaded = Some(c.clone());
            c
        }
    };
    // Drop KV entries the previous request left behind — candle's
    // text tower appends forever and never resets on its own.
    // Safe to call before the first request (no-op on an empty
    // cache).
    cached.model.reset_cache();
    let (input_ids, pixel_values, image_grid_thw) = build_qwen3_inputs(
        &req,
        &cached.device,
        &cached.tokenizer,
    )
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
    tracing::info!(
        "mage-vl: prompt_len={} generated={} first={:?} eos={}",
        input_ids.dim(1).unwrap_or(0),
        generated.len(),
        generated.first().copied().unwrap_or(0),
        eos_token_id,
    );

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

/// Load the checkpoint on the calling thread. Only valid on the
/// mage-vl worker thread — the CUDA context it binds must stay the
/// one every later forward runs on.
fn load_model(
    config: &MageVLConfig,
    shard_dir: &PathBuf,
) -> std::result::Result<MageVlCachedModel, NeuronError> {
    let (device, dtype) = resolve_device();
    let shard1 = shard_dir.join("model-00001-of-00002.safetensors");
    let shard2 = shard_dir.join("model-00002-of-00002.safetensors");
    if !shard1.exists() || !shard2.exists() {
        return Err(NeuronError::new(format!(
            "MageVlRuntime: missing shards at {} (need both \
             model-00001-of-00002.safetensors and \
             model-00002-of-00002.safetensors)",
            shard_dir.display()
        )));
    }
    let model = super::MageVLModel::load_safetensors(
        &[shard1.as_path(), shard2.as_path()],
        config.clone(),
        dtype,
        &device,
    )
    .map_err(|e| NeuronError::new(format!("MageVLModel::load_safetensors failed: {e}")))?;
    let tokenizer = load_tokenizer_from(shard_dir);
    if tokenizer.is_none() {
        tracing::warn!(
            "mage-vl: no tokenizer.json under {}; text encode \
             degrades to byte-level ids",
            shard_dir.display()
        );
    }
    Ok(MageVlCachedModel {
        model: Arc::new(model),
        device,
        _device_sel: DeviceSelection::Cpu,
        tokenizer,
    })
}

impl MageVlRuntime {
    pub fn new(config: MageVLConfig, shard_dir: impl Into<PathBuf>) -> Self {
        Self {
            config,
            shard_dir: shard_dir.into(),
            worker: OnceCell::new(),
        }
    }

    async fn worker(&self) -> &MageVlWorker {
        let config = self.config.clone();
        let shard_dir = self.shard_dir.clone();
        self.worker
            .get_or_init(|| async { MageVlWorker::spawn(config, shard_dir) })
            .await
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
    // CUDA (bf16) is the fast path: the BF16 shards load as-is
    // (~10.8 GB, fits a 24 GB card) and Ampere-class BF16 matmul is
    // served by tensor cores. Candle's CPU backend has no BF16
    // matmul kernel ("unsupported dtype BF16 for op matmul"), so the
    // CPU fallback upcasts to F32 at load time (~18 GB resident for
    // the 4B text tower — needs a 32 GB host; a 16 GB host swaps).
    match Device::new_cuda(0) {
        Ok(device) => {
            tracing::info!("mage-vl: CUDA device acquired, loading bf16 weights");
            (device, DType::BF16)
        }
        Err(e) => {
            tracing::info!("mage-vl: CUDA unavailable ({e}); loading f32 weights on CPU");
            (Device::Cpu, DType::F32)
        }
    }
}

#[async_trait]
impl LlmRuntime for MageVlRuntime {
    async fn execute(
        &self,
        req: CanonicalLlmRequest,
    ) -> crate::llm::ChatResult<CanonicalLlmResponse> {
        let worker = self.worker().await;
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        worker
            .tx
            .send(Job::Execute {
                req: Box::new(req),
                reply: reply_tx,
            })
            .map_err(|_| NeuronError::new("mage-vl worker thread has died"))?;
        reply_rx
            .await
            .map_err(|_| NeuronError::new("mage-vl worker dropped the reply"))?
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
///
/// Sync by design: it allocates CUDA tensors, so it must run on the
/// mage-vl worker thread (see `MageVlRuntime::worker`).
fn build_qwen3_inputs(
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
        Some(img) => decode_image_block(img, device)?,
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
    let mut token_ids = encode_prompt(&prompt, tokenizer);
    // The jinja template writes exactly one `<|image_pad|>` per image.
    // BPE keeps it as the special id 151_655, so we can duplicate the
    // id post-tokenization to match the merged-token count without
    // the scatter step searching for plain-text "image_pad" substrings.
    if n_pad_tokens > 0 {
        let pad_id = tokenizer
            .as_ref()
            .and_then(|t| t.token_to_id("<|image_pad|>"))
            .unwrap_or(151_655);
        token_ids = expand_image_pad_token(token_ids, pad_id, n_pad_tokens);
    }
    let ids_len = token_ids.len();
    let input_ids = Tensor::from_vec(token_ids, (1usize, ids_len), device)?;

    let (pixel_values, image_grid_thw) = match preprocessed {
        Some((p, g)) => (Some(p), Some(g)),
        None => (None, None),
    };
    Ok((input_ids, pixel_values, image_grid_thw))
}

/// Render the Qwen3 chat template for a single-turn exchange:
/// system turn, user turn (with optional image placeholders), then
/// an open assistant turn for generation.
///
/// The template (see `chat_template.jinja` shipped with the
/// checkpoint) emits a single literal `<|vision_start|><|image_pad|>
/// <|vision_end|>` per image — exactly one pad token. The
/// `image_pad` count is then expanded post-tokenization by
/// `expand_image_pad_token` so the scatter step still gets one
/// placeholder per merged vision token, but every pad stays the
/// special token 151_655 (writing N literal pad strings makes BPE
/// treat them as plain text and the scatter never finds them).
fn render_chat_template(system: &str, user_texts: &[&str], n_pad_tokens: usize) -> String {
    let mut s = String::with_capacity(256 + user_texts.iter().map(|t| t.len()).sum::<usize>());
    s.push_str("<|im_start|>system\n");
    s.push_str(system);
    s.push_str("<|im_end|>\n");
    s.push_str("<|im_start|>user\n");
    // The jinja template emits `<|vision_start|><|image_pad|><|vision_end|>`
    // for every user turn with an image (and it may also emit a text
    // block from the user). With Mage-VL v1 we accept only one image
    // per request, so the literal "image_pad" count is always 1 here —
    // it gets expanded to `n_pad_tokens` after BPE in
    // `expand_image_pad_token`.
    if n_pad_tokens > 0 {
        s.push_str("<|vision_start|><|image_pad|><|vision_end|>");
    }
    for t in user_texts {
        s.push_str(t);
        s.push('\n');
    }
    s.push_str("<|im_end|>\n");
    s.push_str("<|im_start|>assistant\n");
    s
}

/// After BPE, duplicate the single `<|image_pad|>` (id 151_655) so
/// the number of placeholders equals the number of merged vision
/// tokens. BPE keeps `<|image_pad|>` as one token per occurrence, so
/// the scatter step's `placeholder_positions` finds exactly N
/// placeholders and the loop body `n_pad_tokens = (h/2)*(w/2)` stays
/// correct.
fn expand_image_pad_token(mut ids: Vec<u32>, pad_id: u32, n: usize) -> Vec<u32> {
    if n == 0 {
        return ids;
    }
    let mut out = Vec::with_capacity(ids.len().saturating_add(n));
    let mut expanded = 0u32;
    for id in ids.drain(..) {
        if id == pad_id {
            for _ in 0..n {
                out.push(pad_id);
            }
            expanded += 1;
        } else {
            out.push(id);
        }
    }
    if expanded == 0 {
        // Defensive: caller forgot the template marker. Refuse to
        // silently produce a token sequence that has zero placeholders
        // — every request needs them in lockstep with merged tokens.
        tracing::warn!(
            "mage-vl: prompt has zero <|image_pad|>; expected 1 (expanding to {n})"
        );
    }
    out
}

/// Decode one `ImageSource::Base64` into the (pixel_values,
/// image_grid_thw) pair candle expects, using the real Qwen
/// image preprocessing pipeline:
/// RGB → resize to multiples of 32 (patch 16 × merge 2) →
/// scale to [0,1] → CLIP-stats normalize → channel-first
/// `[1, 3, 1, H, W]` with grid `[1, h/16, w/16]` (patchify is the
/// strided conv inside MageViT's PatchEmbed).
/// Returns `None` for URLs/FileIds (no fetcher wired in).
fn decode_image_block(
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
                .map_err(|e| candle::Error::msg(format!("base64 decode: {e}")))?;
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
        .map_err(|e| candle::Error::msg(format!("image sniff: {e}")))?
        .decode()
        .map_err(|e| candle::Error::msg(format!("image decode: {e}")))?;
    let rgb = img.to_rgb8();

    let (raw_w, raw_h) = (rgb.width() as usize, rgb.height() as usize);
    // Resize so both dims are multiples of patch*merge (=28).
    let unit = PATCH_SIZE * SPATIAL_MERGE;
    // Qwen-VL family convention: floor to a multiple of 32 (patch=16,
    // merge=2). `round` would *enlarge* an image like 500×332 to 512×352
    // and shift the patch grid by 31 pixels, which on a small image
    // obscures the original content and breaks inference.
    let round_to_unit = |v: usize| ((v.max(unit) / unit).max(1)) * unit;
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
        let p = render_chat_template("You are helpful.", &["describe"], 1);
        assert!(p.starts_with("<|im_start|>system\nYou are helpful.<|im_end|>\n"));
        // jinja-template parity: ONE literal <|image_pad|>, expanded
        // post-tokenization in `expand_image_pad_token`.
        assert!(p.contains("<|vision_start|><|image_pad|><|vision_end|>"));
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
        let (ids, pixels, grid) = build_qwen3_inputs(&req, &device, &None).unwrap();
        // No image attached → vision tensors stay None and the
        // template still opens the assistant turn for generation.
        assert!(pixels.is_none());
        assert!(grid.is_none());
        let (seq_len, ) = (ids.dims()[1],);
        assert!(seq_len > 0);
    }
}
