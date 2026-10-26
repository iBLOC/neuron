use std::{marker::PhantomData, path::Path};

use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig},
    prelude::Backend,
    tensor::{
        activation::{silu, tanh},
        ops::PadMode,
        DType, Int,
    },
    Tensor,
};

use burn::prelude::*;
use tokenizers::Tokenizer;

use super::audiovae::AudioVae;
use super::minicpm4::{MiniCPMConfig, MiniCPMModel};

pub fn display_tensor<const D: usize, B: Backend>(t: &Tensor<B, D>) -> String {
    std::format!("({:?}, {:?})", t.dims(), t.dtype())
}

pub fn display_tensor_int<const D: usize, B: Backend>(t: &Tensor<B, D>) -> String {
    std::format!("({:?}, {:?})", t.dims(), t.dtype())
}

#[derive(Debug, Config)]
pub struct VoxCPMConfig {
    pub lm_config: MiniCPMConfig,
    #[config(default = 2)]
    pub patch_size: usize,
    #[config(default = 64)]
    pub feat_dim: usize,
    #[config(default = 6)]
    pub residual_lm_num_layers: usize,
    #[config(default = 256)]
    pub scalar_quantization_latent_dim: usize,
    #[config(default = 9)]
    pub scalar_quantization_scale: usize,
    pub encoder_config: VoxCPMLocEncConfig,
    pub dit_config: VoxCPMDitConfig,
    #[config(default = 4096)]
    pub max_length: usize,
}

impl VoxCPMConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> VoxCPM<B> {
        let mut residual_lm_config = self.lm_config.clone();
        residual_lm_config.num_hidden_layers = self.residual_lm_num_layers;
        residual_lm_config.vocab_size = 0;

        let mut feat_encoder_lm_config = self.lm_config.clone();
        feat_encoder_lm_config.hidden_size = self.encoder_config.hidden_dim;
        feat_encoder_lm_config.intermediate_size = self.encoder_config.ffn_dim;
        feat_encoder_lm_config.num_attention_heads = self.encoder_config.num_heads;
        feat_encoder_lm_config.num_hidden_layers = self.encoder_config.num_layers;
        feat_encoder_lm_config.kv_channels = self.encoder_config.kv_channels;
        feat_encoder_lm_config.vocab_size = 0;

        let mut feat_decoder_lm_config = self.lm_config.clone();
        feat_decoder_lm_config.hidden_size = self.dit_config.hidden_dim;
        feat_decoder_lm_config.intermediate_size = self.dit_config.ffn_dim;
        feat_decoder_lm_config.num_attention_heads = self.dit_config.num_heads;
        feat_decoder_lm_config.num_hidden_layers = self.dit_config.num_layers;
        feat_decoder_lm_config.kv_channels = self.dit_config.kv_channels;
        feat_decoder_lm_config.vocab_size = 0;

        VoxCPM {
            audio_start_token: 101,
            audio_end_token: 102,
            use_mup: self.lm_config.use_mup,
            scale_emb: self.lm_config.scale_emb,
            patch_size: self.patch_size,
            base_lm: self.lm_config.init(Some((1, self.max_length)), device),
            residual_lm: residual_lm_config.init(Some((1, self.max_length)), device),
            feat_encoder: self
                .encoder_config
                .init(feat_encoder_lm_config, self.feat_dim, device),
            feat_decoder: self.dit_config.cfm_config.init(
                VoxCPMLocDiTConfig::new(self.feat_dim),
                feat_decoder_lm_config,
                self.feat_dim,
                device,
            ),
            fsq_layer: ScalarQuantizationLayerConfig::new(
                self.lm_config.hidden_size,
                self.lm_config.hidden_size,
                self.scalar_quantization_latent_dim,
                self.scalar_quantization_scale,
            )
            .init(device),
            enc_to_lm_proj: LinearConfig::new(
                self.encoder_config.hidden_dim,
                self.lm_config.hidden_size,
            )
            .init(device),
            lm_to_dit_proj: LinearConfig::new(
                self.lm_config.hidden_size,
                self.dit_config.hidden_dim,
            )
            .init(device),
            res_to_dit_proj: LinearConfig::new(
                self.lm_config.hidden_size,
                self.dit_config.hidden_dim,
            )
            .init(device),
            stop_proj: LinearConfig::new(self.lm_config.hidden_size, self.lm_config.hidden_size)
                .init(device),
            stop_head: LinearConfig::new(self.lm_config.hidden_size, 2)
                .with_bias(false)
                .init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct VoxCPM<B: Backend> {
    use_mup: bool,
    scale_emb: f32,
    pub patch_size: usize,
    pub audio_start_token: usize,
    pub audio_end_token: usize,
    pub base_lm: MiniCPMModel<B>,
    pub residual_lm: MiniCPMModel<B>,
    pub feat_encoder: VoxCPMLocEnc<B>,
    pub feat_decoder: UnifiedCFM<B>,
    pub fsq_layer: ScalarQuantizationLayer<B>,
    pub enc_to_lm_proj: Linear<B>,
    pub lm_to_dit_proj: Linear<B>,
    pub res_to_dit_proj: Linear<B>,
    pub stop_proj: Linear<B>,
    pub stop_head: Linear<B>,
}

#[allow(clippy::too_many_arguments)]
impl<B: Backend> VoxCPM<B> {
    pub fn generate<AB: Backend>(
        &mut self,
        target_text: &str,
        prompt: Option<(String, Tensor<AB, 2>)>,
        tokenizer_path: &Path,
        min_len: Option<usize>,
        max_len: Option<usize>,
        inference_timesteps: Option<usize>,
        cfg_value: Option<f32>,
        _retry_badcase: bool,
        _retry_badcase_max_times: usize,
        _retry_badcase_ratio_threshold: f32,
        audio_vae: &AudioVae<AB>,
        device: &B::Device,
        adevice: &AB::Device,
    ) -> Tensor<AB, 1> {
        let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();
        let text_token_in;
        let text_mask_in;
        let audio_feat_in;
        let audio_mask_in;
        let target_text_length;
        if let Some((prompt_text, mut prompt_audio)) = prompt {
            let text = prompt_text.to_string() + target_text;
            let text_token = tokenizer.encode(text, false).unwrap();
            target_text_length = text_token.get_ids().len();
            let text_token: Tensor<B, 1, Int> = Tensor::from_data(text_token.get_ids(), device);
            let text_token = Tensor::cat(
                vec![
                    text_token,
                    Tensor::from_data([self.audio_start_token], device),
                ],
                0,
            );
            let text_length = text_token.dims()[0];
            let patch_len = self.patch_size * audio_vae.chunk_size;
            let pad_size = prompt_audio.dims()[1] % patch_len;
            if pad_size != 0 {
                prompt_audio = Tensor::pad(
                    prompt_audio,
                    (0, patch_len - pad_size, 0, 0),
                    PadMode::Constant(0.0),
                );
            }

            let audio_feat =
                audio_vae.encode(prompt_audio.unsqueeze(), Some(audio_vae.sample_rate));

            let audio_feat = audio_feat
                .reshape([audio_vae.latent_dim as i64, -1, self.patch_size as i64])
                .permute([1, 2, 0]);

            let audio_feat = audio_feat.slice([s![..-1], s![..], s![..]]);
            let audio_feat = Tensor::from(audio_feat.to_data());
            let audio_length = audio_feat.dims()[0];
            let text_pad_token = Tensor::zeros([audio_length], device);

            let text_token = Tensor::cat(vec![text_token, text_pad_token], 0);

            let audio_pad_feat =
                Tensor::zeros([text_length, self.patch_size, audio_vae.latent_dim], device);

            let audio_feat = Tensor::cat(vec![audio_pad_feat, audio_feat], 0);

            let text_mask = Tensor::cat(
                vec![
                    Tensor::ones([text_length], device),
                    Tensor::zeros([audio_length], device),
                ],
                0,
            );

            let audio_mask = Tensor::cat(
                vec![
                    Tensor::zeros([text_length], device),
                    Tensor::ones([audio_length], device),
                ],
                0,
            );

            text_token_in = text_token;
            text_mask_in = text_mask;
            audio_feat_in = audio_feat;
            audio_mask_in = audio_mask;
        } else {
            let text = target_text;
            let text_token = tokenizer.encode(text, false).unwrap();
            target_text_length = text_token.get_ids().len();
            let text_token: Tensor<B, 1, Int> = Tensor::from_data(text_token.get_ids(), device);
            let text_token = Tensor::cat(
                vec![
                    text_token,
                    Tensor::from_data([self.audio_start_token], device),
                ],
                0,
            );

            let text_length = text_token.dims()[0];

            let audio_feat: Tensor<B, 3> =
                Tensor::zeros([text_length, self.patch_size, audio_vae.latent_dim], device);
            let text_mask: Tensor<B, 1> = Tensor::ones([text_length], device);
            let audio_mask: Tensor<B, 1> = Tensor::zeros([text_length], device);

            text_token_in = text_token;
            text_mask_in = text_mask;
            audio_feat_in = audio_feat;
            audio_mask_in = audio_mask;
        }

        let text_token = text_token_in.unsqueeze_dim(0);
        let text_mask = text_mask_in.unsqueeze();
        let audio_feat = audio_feat_in.unsqueeze_dim(0);
        let audio_mask = audio_mask_in.unsqueeze();

        let (latent_pred, pred_audio_feat) = self.forward(
            text_token,
            text_mask,
            audio_feat,
            audio_mask,
            min_len,
            max_len,
            inference_timesteps,
            cfg_value,
        );
        let pred_audio_feat_len = pred_audio_feat.dims()[0];

        let latent_pred: Tensor<AB, 3> =
            Tensor::from_data(latent_pred.cast(DType::F32).to_data(), adevice);

        let decode_audio = audio_vae.decode(latent_pred).squeeze_dim::<2>(0);
        decode_audio.slice([s![..], s![640..-640]]).squeeze()
    }

    pub fn forward(
        &mut self,
        text: Tensor<B, 2, Int>,
        text_mask: Tensor<B, 2>,
        feat: Tensor<B, 4>,
        feat_mask: Tensor<B, 2>,
        min_len: Option<usize>,
        max_len: Option<usize>,
        inference_timesteps: Option<usize>,
        cfg_value: Option<f32>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let min_len = min_len.unwrap_or(2);
        let max_len = max_len.unwrap_or(2000);
        let inference_timesteps = inference_timesteps.unwrap_or(10);
        let cfg_value = cfg_value.unwrap_or(2.0);

        let [B, T, P, D] = feat.dims();

        let feat_embed = self.feat_encoder.forward(feat.clone());
        let feat_embed = self.enc_to_lm_proj.forward(feat_embed.clone());

        let scale_emb = if self.use_mup { self.scale_emb } else { 1.0 };

        let text_embed = match &self.base_lm.embed_tokens {
            Some(val) => val.forward(text),
            None => text.unsqueeze().float(),
        };

        let text_embed = text_embed * scale_emb;
        let combined_embed = text_mask.clone().unsqueeze_dims(&[-1]) * text_embed
            + feat_mask.clone().unsqueeze_dims(&[-1]) * feat_embed.clone();

        let mut prefix_feat_cond = feat
            .slice([s![..], s![-1], s![..], s![..]])
            .squeeze_dim::<3>(0);
        let mut pred_feat_seq = vec![];

        let (enc_outputs, kv_cache_tuple) = self.base_lm.forward(combined_embed, true);
        if let Some(kv_cache) = self.base_lm.kv_cache.as_mut() {
            kv_cache.fill_cache(kv_cache_tuple)
        }

        let enc_outputs = self.fsq_layer.forward(enc_outputs.clone())
            * feat_mask.clone().unsqueeze_dims(&[-1])
            + enc_outputs * text_mask.unsqueeze_dims(&[-1]);
        let mut lm_hidden: Tensor<B, 2> = enc_outputs
            .clone()
            .slice([s![..], s![-1], s![..]])
            .squeeze_dim::<2>(0);

        let (residual_enc_outputs, residual_kv_cache_tuple) = self.residual_lm.forward(
            enc_outputs.unsqueeze() + feat_mask.unsqueeze_dims(&[-1]) * feat_embed,
            true,
        );

        if let Some(kv_cache) = self.residual_lm.kv_cache.as_mut() {
            kv_cache.fill_cache(residual_kv_cache_tuple)
        }
        let mut residual_hidden = residual_enc_outputs
            .slice([s![..], s![-1], s![..]])
            .squeeze_dim::<2>(0);

        let mut curr_embed: Tensor<B, 3>;
        for i in 0..max_len {
            let dit_hidden_1 = self.lm_to_dit_proj.forward(lm_hidden.clone());
            let dit_hidden_2 = self.res_to_dit_proj.forward(residual_hidden);
            let dit_hidden = dit_hidden_1 + dit_hidden_2;

            let pred_feat = self
                .feat_decoder
                .forward(
                    dit_hidden,
                    inference_timesteps,
                    self.patch_size,
                    prefix_feat_cond.clone().swap_dims(1, 2),
                    None,
                    Some(cfg_value),
                    None,
                    None,
                )
                .swap_dims(1, 2);

            curr_embed = self
                .feat_encoder
                .forward(pred_feat.clone().unsqueeze_dim(1));
            curr_embed = self.enc_to_lm_proj.forward(curr_embed);

            pred_feat_seq.push(pred_feat.clone().unsqueeze_dim(1));
            prefix_feat_cond = pred_feat.clone();

            let stop_data = self
                .stop_head
                .forward(silu(self.stop_proj.forward(lm_hidden)));

            let stop_flag: i64 = stop_data
                .clone()
                .argmax(stop_data.rank() - 1)
                .slice_dim(0, 0..1)
                // The Int dtype is backend-dependent (i64 on ndarray, i32 on
                // the CUDA backend) and the data reader does not convert, so
                // cast to i64 before reading it back.
                .cast(burn::tensor::IntDType::I64)
                .to_data()
                .to_vec::<i64>()
                .ok()
                .and_then(|v| v.first().copied())
                .unwrap_or(0);

            if i > min_len && stop_flag == 1 {
                break;
            }

            let step = self.base_lm.kv_cache.as_mut().unwrap().step();
            lm_hidden = self
                .base_lm
                .forward_step(
                    curr_embed
                        .clone()
                        .slice([s![..], s![0], s![..]])
                        .squeeze_dim::<2>(0),
                    step,
                )
                .unsqueeze_dim(1);

            lm_hidden = self.fsq_layer.forward(lm_hidden);

            let step = self.residual_lm.kv_cache.as_mut().unwrap().step();
            residual_hidden = self
                .residual_lm
                .forward_step(
                    lm_hidden.clone()
                        + curr_embed
                            .slice([s![..], s![0], s![..]])
                            .squeeze_dim::<2>(0),
                    step,
                )
                .unsqueeze_dim(1);
        }

        let pred_feat_seq: Tensor<B, 4> = Tensor::cat(pred_feat_seq, 1);
        let [_, t, _, d] = pred_feat_seq.dims();
        let pred_feat_seq = pred_feat_seq.permute([0, 3, 1, 2]);
        let feat_pred = pred_feat_seq.clone().reshape([B, d, t * self.patch_size]);

        (feat_pred, pred_feat_seq.squeeze_dim::<3>(0))
    }
}

#[derive(Debug, Config)]
pub struct VoxCPMLocEncConfig {
    #[config(default = 1024)]
    hidden_dim: usize,
    #[config(default = 4096)]
    ffn_dim: usize,
    #[config(default = 16)]
    num_heads: usize,
    #[config(default = 4)]
    num_layers: usize,
    kv_channels: Option<usize>,
}

impl VoxCPMLocEncConfig {
    pub fn init<B: Backend>(
        &self,
        config: MiniCPMConfig,
        input_dim: usize,
        device: &B::Device,
    ) -> VoxCPMLocEnc<B> {
        assert!(config.vocab_size == 0);
        VoxCPMLocEnc {
            special_token: burn::module::Param::from_tensor(Tensor::random(
                [1, 1, 1, config.hidden_size],
                Default::default(),
                device,
            )),
            in_proj: LinearConfig::new(input_dim, config.hidden_size)
                .with_bias(true)
                .init(device),
            encoder: config.init(None, device),
        }
    }
}

#[derive(Module, Debug)]
pub struct VoxCPMLocEnc<B: Backend> {
    special_token: burn::module::Param<Tensor<B, 4>>,
    in_proj: Linear<B>,
    encoder: MiniCPMModel<B>,
}

impl<B: Backend> VoxCPMLocEnc<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        let [B, T, P, D] = x.dims();

        let x = self.in_proj.forward(x);
        let special_tokens =
            self.special_token
                .val()
                .expand([B, T, 1, self.special_token.val().dims()[3]]);
        let x = Tensor::cat(vec![special_tokens, x], 2);
        let [b, t, p, c] = x.dims();
        let x = x.reshape([b * t, p, c]);

        let (outputs, _) = self.encoder.forward(x.clone(), false);

        let cls_output = outputs.slice([s![..], s![0], s![..]]).squeeze_dim::<2>(1);
        let [bt, c] = cls_output.dims();

        cls_output.reshape([b, bt / b, c])
    }
}

#[derive(Debug, Config)]
pub struct VoxCPMDitConfig {
    #[config(default = 1024)]
    hidden_dim: usize,
    #[config(default = 4096)]
    ffn_dim: usize,
    #[config(default = 16)]
    num_heads: usize,
    #[config(default = 4)]
    num_layers: usize,
    kv_channels: Option<usize>,
    pub cfm_config: UnifiedCFMConfig,
}

#[derive(Debug, Config)]
pub struct ScalarQuantizationLayerConfig {
    in_dim: usize,
    out_dim: usize,
    latent_dim: usize,
    scale: usize,
}

impl ScalarQuantizationLayerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ScalarQuantizationLayer<B> {
        ScalarQuantizationLayer {
            in_proj: LinearConfig::new(self.in_dim, self.latent_dim).init(device),
            out_proj: LinearConfig::new(self.latent_dim, self.out_dim).init(device),
            scale: self.scale,
        }
    }
}

#[derive(Module, Debug)]
pub struct ScalarQuantizationLayer<B: Backend> {
    in_proj: Linear<B>,
    out_proj: Linear<B>,
    scale: usize,
}

impl<B: Backend> ScalarQuantizationLayer<B> {
    pub fn forward<const D: usize>(&self, hidden: Tensor<B, D>) -> Tensor<B, D> {
        let hidden = self.in_proj.forward(hidden);
        let hidden = tanh(hidden);

        // The scale multiplies the activations, so it must be a float
        // scalar: an integer scalar would make the multiply a mixed
        // int/float op, which the f16 GPU backend rejects ("DTypeMismatch").
        let scale = self.scale as f32;

        let hidden = if B::ad_enabled(&hidden.device()) {
            let quantized = (hidden.clone() * scale).round() / scale;
            hidden.clone() + (quantized - hidden).detach()
        } else {
            (hidden * scale).round() / scale
        };
        let out = self.out_proj.forward(hidden);
        out
    }
}

#[derive(Debug, Config)]
pub struct UnifiedCFMConfig {
    #[config(default = 1e-06)]
    sigma_min: f32,
    #[config(default = "\"euler\".into()")]
    solver: String,
    #[config(default = "\"log-norm\".into()")]
    t_scheduler: String,
}

impl UnifiedCFMConfig {
    pub fn init<B: Backend>(
        &self,
        dit_config: VoxCPMLocDiTConfig,
        config: MiniCPMConfig,
        in_channels: usize,
        device: &B::Device,
    ) -> UnifiedCFM<B> {
        UnifiedCFM {
            in_channels,
            mean_mode: false,
            estimator: VoxCPMLocDiTConfig::new(in_channels).init(config, device),
        }
    }
}

#[derive(Module, Debug)]
pub struct UnifiedCFM<B: Backend> {
    in_channels: usize,
    mean_mode: bool,
    pub estimator: VoxCPMLocDiT<B>,
}

#[allow(clippy::too_many_arguments)]
impl<B: Backend> UnifiedCFM<B> {
    pub fn forward(
        &self,
        mu: Tensor<B, 2>,
        n_timesteps: usize,
        patch_size: usize,
        cond: Tensor<B, 3>,
        temperature: Option<f32>,
        cfg_value: Option<f32>,
        sway_sampling_coef: Option<f32>,
        use_cfg_zero_star: Option<bool>,
    ) -> Tensor<B, 3> {
        let temperature = temperature.unwrap_or(1.0);
        let cfg_value = cfg_value.unwrap_or(1.0);
        let sway_sampling_coef = sway_sampling_coef.unwrap_or(1.0);
        let use_cfg_zero_star = use_cfg_zero_star.unwrap_or(true);

        let [b, c] = mu.dims();
        let t = patch_size;
        let z: Tensor<B, 3> =
            Tensor::random([b, self.in_channels, t], Default::default(), &mu.device())
                * temperature;

        let t_span = Self::linespace(1.0, 0.0, (n_timesteps + 1) as u32, &mu.device());

        let t_span = t_span.clone()
            + sway_sampling_coef
                * ((std::f32::consts::PI / 2.0 * t_span.clone()).cos() - 1.0 + t_span);

        self.solve_euler(z, t_span, mu, cond, cfg_value, use_cfg_zero_star)
    }

    pub fn solve_euler(
        &self,
        mut x: Tensor<B, 3>,
        t_span: Tensor<B, 1>,
        mu: Tensor<B, 2>,
        cond: Tensor<B, 3>,
        cfg_value: f32,
        use_cfg_zero_star: bool,
    ) -> Tensor<B, 3> {
        let mut t = t_span.clone().slice([s![0]]);
        let mut dt = t_span.clone().slice([s![0]]) - t_span.clone().slice([s![1]]);

        let mut sol = vec![];
        let t_span_len = t_span.dims()[0];
        let zero_init_steps = *[1, (t_span_len as f32 * 0.04) as usize]
            .iter()
            .max()
            .unwrap();

        for step in 1..t_span_len {
            let dphi_dt = if use_cfg_zero_star && step <= zero_init_steps {
                None
            } else {
                let b = x.dims()[0];
                let x_in = Tensor::zeros([2 * b, self.in_channels, x.dims()[2]], &mu.device());
                let mu_in = Tensor::zeros([2 * b, mu.dims()[1]], &mu.device());
                let t_in = Tensor::zeros([2 * b], &mu.device());
                let dt_in = Tensor::zeros([2 * b], &mu.device());
                let cond_in = Tensor::zeros([2 * b, self.in_channels, x.dims()[2]], &mu.device());
                let x_in = x_in
                    .clone()
                    .slice_assign([s![..b], s![..], s![..]], x.clone());
                let x_in = x_in
                    .clone()
                    .slice_assign([s![b..], s![..], s![..]], x.clone());
                let mu_in = mu_in.clone().slice_assign([s![..b], s![..]], mu.clone());
                let t_in = t_in.clone().slice_assign(s![..b], t.clone());
                let t_in = t_in.clone().slice_assign(s![b..], t.clone());
                let dt_in = dt_in.clone().slice_assign(s![..b], dt.clone());
                let mut dt_in = dt_in.clone().slice_assign(s![b..], dt.clone());
                if !self.mean_mode {
                    dt_in = Tensor::zeros_like(&dt_in);
                }
                let cond_in = cond_in
                    .clone()
                    .slice_assign([s![..b], s![..], s![..]], cond.clone());
                let cond_in = cond_in
                    .clone()
                    .slice_assign([s![b..], s![..], s![..]], cond.clone());

                let dphi_dt_data = self.estimator.forward(x_in, mu_in, t_in, cond_in, dt_in);
                let data = dphi_dt_data.split(x.dims()[0], 0);
                let dphi_dt_data = data[0].clone();
                let cfg_dphi_dt_data = data[1].clone();

                let st_star = if use_cfg_zero_star {
                    let positive_flat = dphi_dt_data.clone().reshape([b as i64, -1]);
                    let negative_flat = cfg_dphi_dt_data.clone().reshape([b as i64, -1]);
                    let st_star = Self::optimized_scale(positive_flat, negative_flat);

                    let mut shape = vec![b];
                    shape.extend(std::iter::repeat_n(1, dphi_dt_data.dims().len() - 1));
                    Some(st_star.reshape(Shape::from(shape)))
                } else {
                    None
                };

                let dphi_dt = match st_star {
                    Some(val) => {
                        cfg_dphi_dt_data.clone() * val.clone()
                            + cfg_value
                                * (dphi_dt_data.clone() - cfg_dphi_dt_data.clone() * val.clone())
                    }
                    None => {
                        cfg_dphi_dt_data.clone() * 1.0
                            + cfg_value * (dphi_dt_data.clone() - cfg_dphi_dt_data.clone() * 1.0)
                    }
                };
                Some(dphi_dt)
            };
            x = match dphi_dt {
                Some(val) => x - dt.clone().unsqueeze() * val,
                None => x,
            };
            t = t.clone() - dt.clone();
            sol.push(x.clone());
            if step < t_span_len - 1 {
                dt = t.clone()
                    - t_span
                        .clone()
                        .select(0, Tensor::from_data([step + 1], &mu.device()));
            }
        }
        sol.last().unwrap().clone()
    }

    fn optimized_scale<const D: usize>(
        positive_flat: Tensor<B, D>,
        negative_flat: Tensor<B, D>,
    ) -> Tensor<B, D> {
        let dot_product = (positive_flat * negative_flat.clone()).sum_dim(1);
        let squared_norm = (negative_flat.square()).sum_dim(1) + 1e-8;
        dot_product / squared_norm
    }

    fn linespace(start: f32, end: f32, steps: u32, device: &B::Device) -> Tensor<B, 1> {
        let arrange = Tensor::<B, 1, Int>::arange(0..steps as i64, device);
        arrange.float() * (end - start) / (steps - 1) + start
    }
}

#[derive(Debug, Config)]
pub struct VoxCPMLocDiTConfig {
    in_channels: usize,
}

impl VoxCPMLocDiTConfig {
    pub fn init<B: Backend>(&self, config: MiniCPMConfig, device: &B::Device) -> VoxCPMLocDiT<B> {
        let out_channels = self.in_channels;
        VoxCPMLocDiT {
            in_proj: LinearConfig::new(self.in_channels, config.hidden_size)
                .with_bias(true)
                .init(device),
            cond_proj: LinearConfig::new(self.in_channels, config.hidden_size)
                .with_bias(true)
                .init(device),
            out_proj: LinearConfig::new(config.hidden_size, out_channels)
                .with_bias(true)
                .init(device),
            time_embeddings: SinusoidalPosEmbConfig::new(config.hidden_size).init(device),
            time_mlp: TimestepEmbeddingConfig::new(config.hidden_size, config.hidden_size)
                .init(device),
            delta_time_mlp: TimestepEmbeddingConfig::new(config.hidden_size, config.hidden_size)
                .init(device),
            decoder: config.init(None, device),
        }
    }
}

#[derive(Module, Debug)]
pub struct VoxCPMLocDiT<B: Backend> {
    pub in_proj: Linear<B>,
    cond_proj: Linear<B>,
    out_proj: Linear<B>,
    pub time_embeddings: SinusoidalPosEmb<B>,
    pub time_mlp: TimestepEmbedding<B>,
    delta_time_mlp: TimestepEmbedding<B>,
    decoder: MiniCPMModel<B>,
}

impl<B: Backend> VoxCPMLocDiT<B> {
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        mu: Tensor<B, 2>,
        t: Tensor<B, 1>,
        cond: Tensor<B, 3>,
        dt: Tensor<B, 1>,
    ) -> Tensor<B, 3> {
        let x = self.in_proj.forward(x.swap_dims(1, 2));
        let cond = self.cond_proj.forward(cond.swap_dims(1, 2));
        let prefix = cond.dims()[1];

        let t = self.time_embeddings.forward(t, None).cast(x.dtype());
        let t = self.time_mlp.forward(t);
        let dt = self.time_embeddings.forward(dt, None).cast(x.dtype());
        let dt = self.delta_time_mlp.forward(dt);
        let t = t + dt;

        let x = Tensor::cat(vec![(mu + t.unsqueeze()).unsqueeze_dim(1), cond, x], 1);

        let (hidden, _) = self.decoder.forward(x, false);
        let hidden = hidden.slice([s![..], s![prefix + 1..], s![..]]);
        let hidden = self.out_proj.forward(hidden);

        hidden.swap_dims(1, 2)
    }
}

#[derive(Debug, Config)]
pub struct SinusoidalPosEmbConfig {
    dim: usize,
}

impl SinusoidalPosEmbConfig {
    pub fn init<B: Backend>(&self, _device: &B::Device) -> SinusoidalPosEmb<B> {
        assert!(self.dim.is_multiple_of(2));
        SinusoidalPosEmb {
            dim: self.dim,
            _p: Default::default(),
        }
    }
}

#[derive(Module, Debug)]
pub struct SinusoidalPosEmb<B: Backend> {
    dim: usize,
    _p: PhantomData<B>,
}

impl<B: Backend> SinusoidalPosEmb<B> {
    pub fn forward(&self, x: Tensor<B, 1>, scale: Option<usize>) -> Tensor<B, 2> {
        let scale = scale.unwrap_or(1000);

        let x = if x.dims().len() < 1 {
            x.unsqueeze_dim(0)
        } else {
            x
        };
        let device = x.device();
        let half_dim = self.dim / 2;

        let emb = (10000.0_f64).ln() / (half_dim - 1) as f64;
        let emb = (Tensor::arange(0..half_dim as i64, &device).float() * -emb)
            .exp()
            .cast(x.dtype());
        // The position scale multiplies the activations, so it must be a
        // float scalar — an integer scalar makes this a mixed int/float op,
        // which the f16 GPU backend rejects ("DTypeMismatch").
        let emb = scale as f32 * x.unsqueeze_dim::<2>(1) * emb.unsqueeze_dim::<2>(0);

        let e_len = emb.dims().len();

        Tensor::cat(vec![emb.clone().sin(), emb.cos()], e_len - 1)
    }
}

#[derive(Debug, Config)]
pub struct TimestepEmbeddingConfig {
    in_channels: usize,
    time_embed_dim: usize,
    out_dim: Option<usize>,
}

impl TimestepEmbeddingConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TimestepEmbedding<B> {
        let time_embed_dim_out = match self.out_dim {
            Some(val) => val,
            None => self.time_embed_dim,
        };

        TimestepEmbedding {
            linear_1: LinearConfig::new(self.in_channels, self.time_embed_dim)
                .with_bias(true)
                .init(device),
            linear_2: LinearConfig::new(self.time_embed_dim, time_embed_dim_out)
                .with_bias(true)
                .init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct TimestepEmbedding<B: Backend> {
    pub linear_1: Linear<B>,
    linear_2: Linear<B>,
}

impl<B: Backend> TimestepEmbedding<B> {
    pub fn forward(&self, sample: Tensor<B, 2>) -> Tensor<B, 2> {
        let sample = self.linear_1.forward(sample);
        let sample = silu(sample);
        let sample = self.linear_2.forward(sample);
        sample
    }
}
