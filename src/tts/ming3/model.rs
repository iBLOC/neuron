//! Burn port of the Ming-3.0-tts dense generation model
//! (`modeling_bailingmm.py::BailingMMNativeForConditionalGeneration`).
//!
//! The pipeline: a Qwen2.5-0.5B backbone consumes text/speaker/latent-patch
//! embeddings; at every decode step the backbone's last hidden state
//! (`z_diff`) conditions a DiT flow head (`flowloss`) that predicts one
//! latent patch, a `stop_head` decides whether to stop, and the sampled
//! latent is projected back into the LLM input by `linear_proj_audio` (the
//! Aggregator). The AudioVAE that turns the concatenated latents into a
//! 44.1 kHz waveform lives in [`super::audiovae`] (separate knife).
//!
//! Differences from the VoxCPM2 stack that motivated this port: a single
//! LM (no residual LM / FSQ), patch_size 1, a static 512-position KV cache
//! bounding one utterance, and speaker conditioning through a 192-dim
//! CAM++ embedding projected by `spk_head`.

use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    prelude::Backend,
    tensor::{ElementConversion, Tensor, s},
};

use super::config::Ming3Config;
use super::contract::BuiltInput;
use super::dit::{MingAggregator, MingDiT};
use super::super::minicpm4::{MiniCPMConfig, MiniCPMModel, RopeScalingConfig};

/// The reference decodes with a 512-position static KV cache, so an
/// utterance can run for at most `(512 - prompt length)` steps.
pub const MAX_CACHE_LEN: usize = 512;

/// CAM++ speaker-embedding dimension (`SPEAKER_EMBEDDING_DIM` upstream,
/// `spk_head`'s input width).
pub const SPEAKER_EMBEDDING_DIM: usize = 192;

/// Qwen2 uses plain RoPE at the model's own theta with no length scaling:
/// factors of 1.0 and `original_max == max` make the LongRoPE wrapper a
/// no-op (its `scaling_factor` collapses to 1). The factor tables carry
/// `head_dim / 2` entries — the wrapper multiplies them against the
/// `inv_freq` vector of the same length.
fn qwen2_rope_scaling(max_position_embeddings: usize, head_dim: usize) -> RopeScalingConfig {
    let half = head_dim / 2;
    RopeScalingConfig::new(
        "default".to_string(),
        vec![1.0; half],
        vec![1.0; half],
        max_position_embeddings,
    )
}

/// The dense Ming-3.0-tts model.
#[derive(Module, Debug)]
pub struct Ming3Model<B: Backend> {
    pub patch_size: usize,
    pub history_patch_size: usize,
    pub latent_dim: usize,
    pub llm_hidden_size: usize,
    pub backbone: MiniCPMModel<B>,
    pub linear_proj_audio: MingAggregator<B>,
    pub flowloss: MingDiT<B>,
    pub stop_head: Linear<B>,
    pub spk_head: Linear<B>,
}

impl<B: Backend> Ming3Model<B> {
    pub fn init(config: &Ming3Config, device: &B::Device) -> Self {
        let llm = &config.llm_config;
        let dit = &config.ditar_config;
        let agg = &config.aggregator_config;
        let latent_dim = config.audio_tokenizer_config.enc_kwargs.latent_dim;

        let head_dim = llm.hidden_size / llm.num_attention_heads;
        let mut backbone_cfg = MiniCPMConfig::new(
            151643,
            151645,
            llm.hidden_size,
            llm.intermediate_size,
            llm.max_position_embeddings,
            llm.num_attention_heads,
            llm.num_hidden_layers,
            llm.num_key_value_heads,
            llm.rms_norm_eps,
            qwen2_rope_scaling(llm.max_position_embeddings, head_dim),
            llm.vocab_size,
            // Qwen2: no muP scaling, plain embedding scale.
            false,
            1.0,
            1,
            0.0,
            llm.rope_theta,
        );
        backbone_cfg.kv_channels = Some(llm.hidden_size / llm.num_attention_heads);
        backbone_cfg.qkv_bias = llm.qkv_bias;

        Self {
            patch_size: dit.patch_size,
            history_patch_size: config.history_patch_size(),
            latent_dim,
            llm_hidden_size: llm.hidden_size,
            backbone: backbone_cfg.init(Some((1, MAX_CACHE_LEN)), device),
            linear_proj_audio: MingAggregator::init(
                latent_dim,
                llm.hidden_size,
                agg.depth,
                agg.hidden_size,
                agg.num_heads,
                agg.mlp_ratio,
                dit.patch_size,
                device,
            ),
            flowloss: MingDiT::init(
                latent_dim,
                dit.patch_size,
                config.history_patch_size(),
                llm.hidden_size,
                dit.depth,
                dit.hidden_size,
                dit.num_heads,
                dit.mlp_ratio,
                device,
            ),
            stop_head: LinearConfig::new(llm.hidden_size, 2)
                .with_bias(true)
                .init(device),
            spk_head: LinearConfig::new(SPEAKER_EMBEDDING_DIM, llm.hidden_size)
                .with_bias(true)
                .init(device),
        }
    }

    /// `prepare_input_embed` upstream: embed the contract's token ids, then
    /// overwrite the rows the contract reserved — `speaker_positions[i]`
    /// receives the i-th (projected) speaker embedding and
    /// `latent_position..` receives the projected prompt latents.
    ///
    /// `spk_embs` are raw CAM++ vectors `(1, 192)`; `use_zero_spk_emb`
    /// projects them first and then zeroes the row (upstream semantics —
    /// the projection runs even for the zero-voice path). The device comes
    /// from the caller (the runtime holds it, like the VoxCPM2 runtime).
    pub fn prepare_input_embed(
        &self,
        device: &B::Device,
        built: &BuiltInput,
        spk_embs: Option<&[Tensor<B, 2>]>,
        use_zero_spk_emb: bool,
        prompt_latent: Option<&Tensor<B, 3>>,
    ) -> Result<Tensor<B, 3>, String> {
        let embed_tokens = self
            .backbone
            .embed_tokens
            .as_ref()
            .ok_or("Ming-3.0-tts backbone must carry an embedding table")?;

        let ids = Tensor::<B, 1, burn::tensor::Int>::from_data(built.input_ids.as_slice(), device)
            .reshape([1, built.input_ids.len()]);
        let mut embeds = embed_tokens.forward(ids);

        if let Some(list) = spk_embs {
            if list.len() != built.speaker_positions.len() {
                return Err(format!(
                    "got {} speaker embeddings for {} contract slots",
                    list.len(),
                    built.speaker_positions.len()
                ));
            }
            for (position, se) in built.speaker_positions.iter().zip(list) {
                let projected = self.spk_head.forward(se.clone());
                let projected = if use_zero_spk_emb {
                    projected.zeros_like()
                } else {
                    projected
                };
                embeds = embeds.slice_assign(
                    [s![0..1], s![*position..position + 1], s![..]],
                    projected.reshape([1, 1, self.llm_hidden_size]),
                );
            }
        }

        if let (Some(position), Some(latent)) = (built.latent_position, prompt_latent) {
            let [_, steps, dim] = latent.dims();
            // Each patch is its own Aggregator batch row: (steps, patch, latent).
            let flat = latent
                .clone()
                .reshape([steps, self.patch_size, dim]);
            let projected = self.linear_proj_audio.forward(flat);
            let projected = projected.reshape([1, steps, self.llm_hidden_size]);
            embeds = embeds.slice_assign(
                [s![0..1], s![position..position + steps], s![..]],
                projected,
            );
        }

        Ok(embeds)
    }

    /// The AR decode loop (`BailingMMNativeForConditionalGeneration.sample`
    /// dense path). Returns the concatenated latent patches
    /// `(1, n_steps * patch, latent)` and whether the stop head fired.
    ///
    /// `prompt_latent` is the VAE latents of the acoustic prompt
    /// `(1, T_patch, latent)` — its tail seeds `latent_history`.
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &mut self,
        inputs_embeds: Tensor<B, 3>,
        prompt_latent: Option<&Tensor<B, 3>>,
        spk_embs: Option<&[Tensor<B, 2>]>,
        use_zero_spk_emb: bool,
        built: &BuiltInput,
        max_decode_steps: usize,
        cfm: &super::cfm::CfmOptions,
    ) -> Result<(Tensor<B, 3>, bool), String> {
        let device = inputs_embeds.device();
        let prompt_len = inputs_embeds.dims()[1];

        // Budget check BEFORE the prompt pass: the static KV cache cannot
        // hold a prompt at or beyond MAX_CACHE_LEN.
        let decode_budget = (MAX_CACHE_LEN.saturating_sub(prompt_len)) / self.patch_size;
        let budget = max_decode_steps.min(decode_budget);
        if budget == 0 {
            return Err(format!(
                "prompt of {prompt_len} embeds leaves no decode budget within \
                 MAX_CACHE_LEN={MAX_CACHE_LEN}"
            ));
        }

        // First pass over the whole prompt, filling the static KV cache.
        let (hidden, kv_caches) = self.backbone.forward(inputs_embeds, true);
        if let Some(kv_cache) = self.backbone.kv_cache.as_mut() {
            kv_cache.fill_cache(kv_caches);
        }
        let seq = hidden.dims()[1];
        let mut z = hidden.slice([s![0..1], s![seq - 1..seq], s![..]]);

        // latent_history seeds from the prompt latents' tail.
        let history = self.history_patch_size;
        let mut latent_history = Tensor::<B, 3>::zeros(
            [1, history, self.latent_dim],
            &device,
        );
        if let Some(prompt) = prompt_latent {
            let pl_len = prompt.dims()[1];
            if pl_len > history {
                // Keep only the last `history` patches.
                let skip = pl_len - history;
                latent_history =
                    prompt.clone().slice([s![0..1], s![skip..pl_len], s![..]]);
            } else {
                let pad = history - pl_len;
                latent_history = latent_history.slice_assign(
                    [s![0..1], s![pad..pad + pl_len], s![..]],
                    prompt.clone(),
                );
            }
        }

        let mut latents: Vec<Tensor<B, 3>> = Vec::new();
        let mut stopped = false;
        for step in 0..budget {
            let noise = Tensor::random(
                [1, self.patch_size, self.latent_dim],
                Default::default(),
                &device,
            );
            let sampled = super::cfm::sample(
                &self.flowloss,
                z.clone(),
                latent_history.clone(),
                noise,
                None,
                cfm,
            );
            latents.push(sampled.clone());

            // stop_head on z_diff: softmax over 2, stop when the positive
            // class crosses 0.5 after a few warmup steps.
            let stop_logits = self
                .stop_head
                .forward(z.clone().reshape([1, self.llm_hidden_size]));
            let stop_prob = burn::tensor::activation::softmax(stop_logits, 1)
                .slice([s![0..1], s![1..2]])
                .into_scalar()
                .elem::<f32>();
            if stop_prob > 0.5 && step > 3 {
                stopped = true;
                break;
            }

            // Project the sampled latent back into the LLM input.
            let next_embed = self.linear_proj_audio.forward(sampled.clone());
            let next_embed = next_embed.reshape([1, self.llm_hidden_size]);
            let position_id = self
                .backbone
                .kv_cache
                .as_mut()
                .expect("kv cache initialised")
                .step();
            z = self.backbone.forward_step(next_embed, position_id).reshape([1, 1, self.llm_hidden_size]);

            // Slide the latent history forward by one patch.
            if self.patch_size < history {
                latent_history = latent_history.slice([
                    s![0..1],
                    s![self.patch_size..history],
                    s![..],
                ]);
                // width changes; re-pad below
                let width = latent_history.dims()[1];
                latent_history = Tensor::cat(
                    vec![
                        latent_history,
                        Tensor::zeros(
                            [1, history - width, self.latent_dim],
                            &device,
                        ),
                    ],
                    1,
                );
            }
            latent_history = latent_history.slice_assign(
                [s![0..1], s![history - self.patch_size..history], s![..]],
                sampled.reshape([1, self.patch_size, self.latent_dim]),
            );
        }

        let _ = (spk_embs, use_zero_spk_emb); // consumed by prepare_input_embed
        let count = latents.len();
        if count == 0 {
            return Err("decode produced no latents".to_string());
        }
        let latents = Tensor::cat(
            latents.into_iter().map(|t| t.reshape([1, self.patch_size, self.latent_dim])).collect(),
            1,
        );
        Ok((latents, stopped))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::cfm::CfmOptions;
    use burn::backend::ndarray::{NdArray, NdArrayDevice};

    type TestBackend = NdArray;

    fn tiny_config() -> Ming3Config {
        // A minimal dense config with the same shape family as the
        // checkpoint: 2-layer backbone, 1-layer DiT/Aggregator, small
        // hidden sizes, latent 8.
        serde_json::from_value(serde_json::json!({
            "model_type": "dense",
            "llm_config": {
                "hidden_size": 32, "intermediate_size": 64,
                "max_position_embeddings": 512, "num_attention_heads": 4,
                "num_hidden_layers": 2, "num_key_value_heads": 2,
                "rms_norm_eps": 1e-6, "rope_theta": 1000000.0,
                "vocab_size": 64, "qkv_bias": true
            },
            "audio_tokenizer_config": {
                "enc_kwargs": { "latent_dim": 8, "input_dim": 16, "hop_size": 16,
                    "backbone": { "hidden_size": 32, "intermediate_size": 64,
                        "max_position_embeddings": 512, "num_attention_heads": 4,
                        "num_hidden_layers": 2, "num_key_value_heads": 2,
                        "rms_norm_eps": 1e-6, "rope_theta": 1000000.0,
                        "vocab_size": 1, "use_sliding_window": false } },
                "dec_kwargs": { "latent_dim": 8, "output_dim": 16,
                    "backbone": { "hidden_size": 32, "intermediate_size": 64,
                        "max_position_embeddings": 512, "num_attention_heads": 4,
                        "num_hidden_layers": 2, "num_key_value_heads": 2,
                        "rms_norm_eps": 1e-6, "rope_theta": 1000000.0,
                        "vocab_size": 1, "use_sliding_window": false } },
                "sample_rate": 44100, "patch_size": 4
            },
            "ditar_config": { "depth": 1, "hidden_size": 32, "num_heads": 4,
                "mlp_ratio": 4, "patch_size": 1, "history_patch_size": 1 },
            "aggregator_config": { "depth": 1, "hidden_size": 32, "num_heads": 4,
                "mlp_ratio": 4 }
        }))
        .expect("tiny config parses")
    }

    #[test]
    fn generate_loop_runs_and_emits_one_latent_per_step() {
        let device = NdArrayDevice::default();
        let config = tiny_config();
        let mut model = Ming3Model::<TestBackend>::init(&config, &device);

        // Zero the stop head: randomly-initialized weights make the stop
        // probability cross 0.5 nondeterministically, which would make this
        // test flaky (an intermittent failure cannot be proven by rerun
        // counts — pin it by constructing the input instead).
        model.stop_head.weight = burn::module::Param::from_tensor(
            Tensor::<TestBackend, 2>::zeros(model.stop_head.weight.dims(), &device),
        );
        model.stop_head.bias = Some(burn::module::Param::from_tensor(
            Tensor::<TestBackend, 1>::zeros([2], &device),
        ));

        // A prompt of 10 random embeddings (as prepare_input_embed would
        // produce), no acoustic prompt, no speaker embedding.
        let prompt = Tensor::<TestBackend, 3>::random([1, 10, 32], Default::default(), &device);
        let built = BuiltInput {
            input_ids: vec![],
            speaker_positions: vec![],
            latent_position: None,
        };
        let (latents, stopped) = model
            .generate(
                prompt,
                None,
                None,
                false,
                &built,
                6,
                &CfmOptions::default(),
            )
            .expect("generate");
        // Zeroed stop head: logits 0 → softmax 0.5 → never stops (0.5 is
        // not > 0.5) → exactly `max_decode_steps` patches.
        assert_eq!(latents.dims(), [1, 6, 8]);
        assert!(!stopped);
    }

    #[test]
    fn decode_budget_is_bounded_by_the_static_cache() {
        let device = NdArrayDevice::default();
        let config = tiny_config();
        let mut model = Ming3Model::<TestBackend>::init(&config, &device);
        // A prompt longer than MAX_CACHE_LEN leaves no budget.
        let prompt = Tensor::<TestBackend, 3>::random(
            [1, MAX_CACHE_LEN + 10, 32],
            Default::default(),
            &device,
        );
        let built = BuiltInput {
            input_ids: vec![],
            speaker_positions: vec![],
            latent_position: None,
        };
        let err = model
            .generate(prompt, None, None, false, &built, 100, &CfmOptions::default())
            .unwrap_err();
        assert!(err.contains("no decode budget"), "got: {err}");
    }

    #[test]
    fn prepare_input_embed_splices_speaker_and_latent_rows() {
        let device = NdArrayDevice::default();
        let config = tiny_config();
        let model = Ming3Model::<TestBackend>::init(&config, &device);

        // The contract's ids include the latent block: 6 text tokens plus
        // 5 <audioPatch> ids (11 rows); one speaker slot at position 3;
        // the projected latents overwrite rows 6..11.
        let built = BuiltInput {
            input_ids: vec![1, 2, 3, 4, 5, 6, 7, 7, 7, 7, 7],
            speaker_positions: vec![3],
            latent_position: Some(6),
        };
        let spk = vec![Tensor::<TestBackend, 2>::random(
            [1, SPEAKER_EMBEDDING_DIM],
            Default::default(),
            &device,
        )];
        let prompt_latent = Tensor::<TestBackend, 3>::random([1, 5, 8], Default::default(), &device);
        let embeds = model
            .prepare_input_embed(&device, &built, Some(&spk), false, Some(&prompt_latent))
            .expect("embeds");
        assert_eq!(embeds.dims(), [1, 11, 32]);

        // The spliced rows are deterministic functions of their inputs —
        // re-embed the ids alone and compare every non-spliced row.
        let ids_only = {
            let ids =
                Tensor::<TestBackend, 1, burn::tensor::Int>::from_data(&[1u64, 2, 3, 4, 5, 6][..], &device)
                    .reshape([1, 6]);
            model
                .backbone
                .embed_tokens
                .as_ref()
                .unwrap()
                .forward(ids)
        };
        let plain = ids_only.to_data().to_vec::<f32>().expect("read plain");
        let spliced = embeds.to_data().to_vec::<f32>().expect("read spliced");
        for position in 0..11 {
            let row = &spliced[position * 32..(position + 1) * 32];
            let is_spk = position == 3;
            let is_latent = (6..11).contains(&position);
            if !is_spk && !is_latent {
                let expected_start = if position < 6 { position * 32 } else { usize::MAX };
                if expected_start < plain.len() {
                    assert_eq!(row, &plain[expected_start..expected_start + 32]);
                }
            }
        }
    }

    /// The zero-voice path projects then zeroes the row (upstream runs the
    /// projection even when zeroing).
    #[test]
    fn zero_spk_embedding_produces_zero_rows() {
        let device = NdArrayDevice::default();
        let config = tiny_config();
        let model = Ming3Model::<TestBackend>::init(&config, &device);
        let built = BuiltInput {
            input_ids: vec![1, 2, 3],
            speaker_positions: vec![2],
            latent_position: None,
        };
        let spk = vec![Tensor::<TestBackend, 2>::random(
            [1, SPEAKER_EMBEDDING_DIM],
            Default::default(),
            &device,
        )];
        let embeds = model
            .prepare_input_embed(&device, &built, Some(&spk), true, None)
            .expect("embeds");
        let value = embeds.to_data().to_vec::<f32>().expect("read embeds");
        let row = &value[2 * 32..3 * 32];
        assert!(row.iter().all(|v| *v == 0.0));
    }
}
