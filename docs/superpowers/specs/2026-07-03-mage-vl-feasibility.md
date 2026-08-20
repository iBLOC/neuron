# Mage-VL Integration Feasibility Assessment

**Date**: 2026-07-03
**Status**: Feasibility analysis (not yet a design spec)
**Model**: Microsoft Mage-VL (https://huggingface.co/microsoft/Mage-VL)
**Target integration**: neure LLM multimodal path

## Executive Summary

**Verdict**: ✅ **Feasible via mistral.rs backend** (most practical path)

Mage-VL is a 4B VLM (Vision-Language Model) with a custom codec-native visual encoder. It uses Qwen3-4B-Instruct as its language decoder, making it loadable by mistral.rs (which already supports Qwen3 models). The visual encoder is a custom module shipped with the weights via `auto_map`. Since mistral.rs uses HuggingFace Transformers under the hood with `trust_remote_code=True` support, Mage-VL should be loadable as a multimodal model.

The key constraint: Mage-VL's codec-native preprocessing (anchor/predicted frame selection) requires its custom `processing_mage_vl.py` processor. If mistral.rs doesn't support the custom image processor, we fall back to ORT (ONNX Runtime) for the visual encoder and route the text tokens through the standard Qwen3 decoder in mistral.rs.

## Model Architecture

### Mage-VL structure

| Component | Architecture | Parameters |
|---|---|---|
| **Visual encoder** (Mage-ViT) | Codec-ViT from scratch, 24 layers, hidden 1024, 16 heads | ~316M |
| **Language decoder** | Qwen3-4B-Instruct-2507 (standard, unmodified) | ~4B |
| **Streaming gate** | StreamMind gate model | ~0.5B |

Total: ~5B parameters (BF16, ~10.8 GB safetensors)

### Input/Output format

| Aspect | Format |
|---|---|
| **Input** | Image or video (codec-native processing) |
| **Output** | Text tokens (streaming generation) |
| **Image processing** | Codec-native: anchor (I) frames fully processed, predicted (P) frames only motion-salient patches |
| **Token reduction** | 75% fewer visual tokens vs uniform frame sampling |

### Key innovation: Codec-native sparsity

Instead of decoding video into uniformly-sampled frames, Mage-VL:
1. Splits video into anchor (I) frames and predicted (P) frames
2. Keeps every anchor patch in full
3. Retains only predicted-frame patches where the codec spends bits
4. This reduces visual tokens by 75% while preserving spatio-temporal context

## Integration Paths

### Path A: Via mistral.rs (Recommended) ✅

**How it works:**
- mistral.rs supports `MultimodalPlain` with custom HuggingFace models
- Mage-VL's language decoder is standard Qwen3-4B-Instruct-2507
- The visual encoder ships as custom Python code in the model repo
- mistral.rs loads models via HuggingFace Transformers with `trust_remote_code=True`

**Feasibility**: High
- mistral.rs already handles 40+ model families including Qwen2-VL, Llama 3.2 Vision
- Mage-VL uses standard Qwen3 architecture for text generation
- Custom visual encoder should be loadable via `auto_map` in config.json

**Limitations:**
- Codec-native preprocessing may not be natively supported by mistral.rs's image processor
- For video input, the codec pipeline (H.264/HEVC parsing) would need custom integration
- For image input, standard image processing should work fine

**neure integration:**
```rust
// Load Mage-VL as a multimodal model via mistralrs
let model = MultimodalModelBuilder::new("microsoft/Mage-VL")
    .with_auto_isq(IsqBits::Four)
    .build()
    .await?;

// Send image + text prompt
let messages = MultimodalMessages::new()
    .add_image_message(TextMessageRole::User, "Describe this image", vec![image]);

let response = model.send_chat_request(messages).await?;
```

### Path B: ORT (ONNX Runtime) for visual encoder + mistral.rs for decoder

**How it works:**
- Export Mage-ViT visual encoder to ONNX format
- Run visual encoder via `ort` crate
- Pass visual tokens to Qwen3 decoder via mistral.rs

**Feasibility**: Medium
- Requires ONNX export of Mage-ViT (not trivial)
- Two-stage pipeline: visual encoder → language decoder
- More complex but allows fine-grained control

### Path C: Custom candle implementation

**How it works:**
- Implement Mage-ViT architecture in pure candle
- Use standard Qwen3 decoder from candle-transformers

**Feasibility**: Low
- Mage-ViT is a custom architecture (not in candle-transformers)
- Would require implementing 24-layer custom ViT from scratch
- Significantly more effort than using existing backends

## Recommended Approach

**Primary**: Path A (mistral.rs backend) — load Mage-VL as a single multimodal model via the existing `mistralrs` LLM runtime.

**Fallback**: Path B (ORT visual encoder) — if mistral.rs can't handle the custom image processor.

**Integration point**: `/v1/chat/completions` endpoint with `image_url` content blocks, exactly like Qwen2-VL.

## Key Technical Questions to Resolve

1. **Does mistral.rs support Mage-VL's custom image processor?**
   - Check if `MultimodalPlain` accepts custom processors
   - If not, standard image processing may still work for images (not video)

2. **Can mage-VL handle standard image inputs without codec processing?**
   - Yes — Mage-VL handles static images as "single-frame video"
   - Standard image input should work without codec processing

3. **What about the StreamMind gate?**
   - The streaming gate is separate from the core inference
   - Can be ignored for non-streaming use cases

4. **ONNX export feasibility?**
   - Mage-ViT uses custom ops (codec-driven patch selection)
   - ONNX export may require workarounds for custom ops

## Expected Effort

| Component | Effort |
|---|---|
| Load Mage-VL via mistral.rs | 1-2 days |
| Image input preprocessing | 1 day |
| Integration testing with real images | 1 day |
| Documentation + examples | 0.5 day |
| **Total** | **~3-4 days** |

## Next Steps

1. Test loading Mage-VL via mistral.rs (`MultimodalPlain`)
2. Verify image input works without codec processing
3. If codec processing needed, implement ORT-based Mage-ViT encoder
4. Add Mage-VL to default capabilities catalog
5. Write integration tests + smoke example

## References

- Mage-VL paper: https://arxiv.org/abs/2607.24904
- Mage-VL weights: https://huggingface.co/microsoft/Mage-VL
- Mage-ViT encoder: https://huggingface.co/microsoft/Mage-ViT
- mistral.rs multimodal docs: https://github.com/ericlbuehler/mistral.rs
- License: Apache-2.0 (Mage-VL), MIT (Mage-ViT)