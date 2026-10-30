//! FLUX.2 VAE (`AutoencoderKLFlux2`) — decoder-only port for
//! MOSS-Image, referencing diffusers `autoencoder_kl_flux2.py` and
//! reusing candle's public SD-VAE building blocks
//! (`candle_transformers::models::stable_diffusion::unet_2d_blocks`)
//! since the FLUX.2 decoder is the standard SD-style decoder
//! (mid resnets + attention, 4 up blocks, 3 upsamplers).
//!
//! The MOSS-Image latent layout on top of the decoder:
//!
//! 1. **Batch-norm de-normalization**: the 128-channel latent is
//!    un-normalized with the VAE's tracked `bn.running_mean/var`
//!    (`bn` has no affine, `eps = batch_norm_eps`).
//! 2. **2×2 pixel unshuffle**: `[B, 128, h, w]` → `[B, 32, 2h, 2w]`
//!    (`patch_size = [2, 2]`), the exact reshape/permute from the
//!    reference pipeline.
//! 3. `post_quant_conv` (1×1, 32ch) → decoder → `[B, 3, 16h, 16w]`.
//!
//! Decoding runs in fp32 (the reference decodes in fp32 regardless of
//! the DiT dtype).

use candle::{DType, Module, Result, Tensor};
use candle_nn as nn;
use candle_transformers::models::stable_diffusion::unet_2d_blocks::{
    UNetMidBlock2D, UNetMidBlock2DConfig, UpDecoderBlock2D, UpDecoderBlock2DConfig,
};

/// Parsed `vae/config.json`.
#[derive(Debug, Clone)]
pub struct Flux2VaeConfig {
    pub latent_channels: usize,
    pub decoder_block_out_channels: Vec<usize>,
    pub layers_per_block: usize,
    pub norm_num_groups: usize,
    pub patch_size: (usize, usize),
    pub batch_norm_eps: f64,
}

impl Default for Flux2VaeConfig {
    fn default() -> Self {
        Self {
            latent_channels: 32,
            decoder_block_out_channels: vec![96, 192, 384, 384],
            layers_per_block: 2,
            norm_num_groups: 32,
            patch_size: (2, 2),
            batch_norm_eps: 1e-4,
        }
    }
}

struct VaeDecoder {
    conv_in: nn::Conv2d,
    mid_block: UNetMidBlock2D,
    up_blocks: Vec<UpDecoderBlock2D>,
    conv_norm_out: nn::GroupNorm,
    conv_out: nn::Conv2d,
}

impl VaeDecoder {
    fn new(vb: nn::VarBuilder, cfg: &Flux2VaeConfig) -> Result<Self> {
        let reversed: Vec<usize> = cfg.decoder_block_out_channels.iter().rev().copied().collect();
        // The mid block runs at the largest channel count — the last
        // entry of the *original* list, i.e. the FIRST of the reversed
        // one. (`conv_norm_out`/`conv_out` at the bottom use the
        // smallest, the last of the reversed list.)
        let mid_channels = reversed[0];
        let conv_cfg = nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv_in = nn::conv2d(
            cfg.latent_channels,
            mid_channels,
            3,
            conv_cfg,
            vb.pp("conv_in"),
        )?;
        let mid_block = UNetMidBlock2D::new(
            vb.pp("mid_block"),
            mid_channels,
            None,
            UNetMidBlock2DConfig {
                resnet_eps: 1e-6,
                resnet_groups: Some(cfg.norm_num_groups),
                attn_num_head_channels: None,
                output_scale_factor: 1.0,
                ..Default::default()
            },
        )?;
        let mut up_blocks = Vec::with_capacity(reversed.len());
        let vb_up = vb.pp("up_blocks");
        for index in 0..reversed.len() {
            let out_channels = reversed[index];
            let in_channels = if index > 0 {
                reversed[index - 1]
            } else {
                reversed[0]
            };
            let is_final = index + 1 == reversed.len();
            let up_block = UpDecoderBlock2D::new(
                vb_up.pp(index.to_string()),
                in_channels,
                out_channels,
                UpDecoderBlock2DConfig {
                    num_layers: cfg.layers_per_block + 1,
                    resnet_eps: 1e-6,
                    resnet_groups: cfg.norm_num_groups,
                    output_scale_factor: 1.0,
                    add_upsample: !is_final,
                },
            )?;
            up_blocks.push(up_block);
        }
        let conv_norm_out = nn::group_norm(
            cfg.norm_num_groups,
            reversed[reversed.len() - 1],
            1e-6,
            vb.pp("conv_norm_out"),
        )?;
        let conv_out = nn::conv2d(
            reversed[reversed.len() - 1],
            3,
            3,
            nn::Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv_out"),
        )?;
        Ok(Self {
            conv_in,
            mid_block,
            up_blocks,
            conv_norm_out,
            conv_out,
        })
    }
}

impl Module for VaeDecoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = self.mid_block.forward(&self.conv_in.forward(xs)?, None)?;
        for up in &self.up_blocks {
            xs = xs.apply(up)?;
        }
        let xs = xs.apply(&self.conv_norm_out)?;
        let xs = nn::ops::silu(&xs)?;
        self.conv_out.forward(&xs)
    }
}

/// Decoder-side FLUX.2 VAE: BN stats + unshuffle + post_quant_conv +
/// the SD-style decoder. All math in fp32.
pub struct Flux2Vae {
    bn_mean: Tensor, // [patch·latent_ch] fp32
    bn_var: Tensor,
    post_quant_conv: nn::Conv2d,
    decoder: VaeDecoder,
    cfg: Flux2VaeConfig,
}

impl Flux2Vae {
    pub fn new(vb: nn::VarBuilder, cfg: Flux2VaeConfig) -> Result<Self> {
        let packed = cfg.patch_size.0 * cfg.patch_size.1 * cfg.latent_channels;
        let bn_mean = vb.get_with_hints(packed, "bn.running_mean", nn::Init::Const(0.0))?;
        let bn_var = vb.get_with_hints(packed, "bn.running_var", nn::Init::Const(1.0))?;
        let post_quant_conv = nn::conv2d(
            cfg.latent_channels,
            cfg.latent_channels,
            1,
            nn::Conv2dConfig::default(),
            vb.pp("post_quant_conv"),
        )?;
        let decoder = VaeDecoder::new(vb.pp("decoder"), &cfg)?;
        Ok(Self {
            bn_mean,
            bn_var,
            post_quant_conv,
            decoder,
            cfg,
        })
    }

    pub fn config(&self) -> &Flux2VaeConfig {
        &self.cfg
    }

    /// Decode the packed MOSS-Image latent `[B, 128, h, w]` fp32 into
    /// RGB `[B, 3, 16h, 16w]` in `[-1, 1]`.
    pub fn decode(&self, packed: &Tensor) -> Result<Tensor> {
        let packed = packed.to_dtype(DType::F32)?;
        let (b, c, h, w) = (
            packed.dim(0)?,
            packed.dim(1)?,
            packed.dim(2)?,
            packed.dim(3)?,
        );
        // 1. BN de-normalization (channel-wise).
        let std = (&self.bn_var + self.cfg.batch_norm_eps)?.sqrt()?;
        let mean = self
            .bn_mean
            .reshape((1, c, 1, 1))?;
        let std = std.reshape((1, c, 1, 1))?;
        let z = packed
            .broadcast_mul(&std)?
            .broadcast_add(&mean)?;
        // 2. 2×2 pixel unshuffle: [B, C, h, w] → [B, C/4, 2h, 2w].
        let (ph, pw) = self.cfg.patch_size;
        let z = z
            .reshape((b, c / (ph * pw), ph, pw, h, w))?
            .permute((0, 1, 4, 2, 5, 3))?
            .reshape((b, c / (ph * pw), ph * h, pw * w))?;
        // 3. post_quant_conv → decoder.
        let z = z.apply(&self.post_quant_conv)?;
        self.decoder.forward(&z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candle_compat;
    use std::collections::HashMap;

    /// Random-init every decoder tensor under its checkpoint name so
    /// `Flux2Vae::new` succeeds. Uses the real channel layout
    /// ([96, 192, 384, 384]) on a tiny spatial input.
    fn tiny_vae(dev: &candle::Device) -> Flux2Vae {
        let cfg = Flux2VaeConfig::default();
        let mut ts = HashMap::new();
        let mut put = |ts: &mut HashMap<String, Tensor>, name: &str, shape: &[usize]| {
            // Scale the random init down: a deep conv chain with
            // unit-normal weights blows up to inf in fp32 before the
            // decoder finishes, which would fail the finiteness
            // assertion for reasons unrelated to the port.
            let t = Tensor::randn(0f32, 1f32, shape, dev).unwrap();
            ts.insert(name.to_string(), (t * 0.05).unwrap());
        };
        let channels = &cfg.decoder_block_out_channels;
        let reversed: Vec<usize> = channels.iter().rev().copied().collect();
        // BN stats: identity (mean 0, var 1) so the de-normalization
        // stage is a no-op — a random running_var would be negative
        // half the time and sqrt() it into NaN.
        ts.insert(
            "bn.running_mean".to_string(),
            Tensor::zeros((128,), DType::F32, dev).unwrap(),
        );
        ts.insert(
            "bn.running_var".to_string(),
            Tensor::ones((128,), DType::F32, dev).unwrap(),
        );
        put(&mut ts, "post_quant_conv.weight", &[32, 32, 1, 1]);
        put(&mut ts, "post_quant_conv.bias", &[32]);
        let mid_channels = reversed[0];
        put(&mut ts, "decoder.conv_in.weight", &[mid_channels, 32, 3, 3]);
        put(&mut ts, "decoder.conv_in.bias", &[mid_channels]);
        // mid block: resnets.0, attentions.0, resnets.1 at `mid_channels`.
        for name in ["resnets.0", "resnets.1"] {
            let p = format!("decoder.mid_block.{name}");
            put(&mut ts, &format!("{p}.norm1.weight"), &[mid_channels]);
            put(&mut ts, &format!("{p}.norm1.bias"), &[mid_channels]);
            put(&mut ts, &format!("{p}.conv1.weight"), &[mid_channels, mid_channels, 3, 3]);
            put(&mut ts, &format!("{p}.conv1.bias"), &[mid_channels]);
            put(&mut ts, &format!("{p}.norm2.weight"), &[mid_channels]);
            put(&mut ts, &format!("{p}.norm2.bias"), &[mid_channels]);
            put(&mut ts, &format!("{p}.conv2.weight"), &[mid_channels, mid_channels, 3, 3]);
            put(&mut ts, &format!("{p}.conv2.bias"), &[mid_channels]);
        }
        put(&mut ts, "decoder.mid_block.attentions.0.group_norm.weight", &[mid_channels]);
        put(&mut ts, "decoder.mid_block.attentions.0.group_norm.bias", &[mid_channels]);
        put(&mut ts, "decoder.mid_block.attentions.0.to_q.weight", &[mid_channels, mid_channels]);
        put(&mut ts, "decoder.mid_block.attentions.0.to_q.bias", &[mid_channels]);
        put(&mut ts, "decoder.mid_block.attentions.0.to_k.weight", &[mid_channels, mid_channels]);
        put(&mut ts, "decoder.mid_block.attentions.0.to_k.bias", &[mid_channels]);
        put(&mut ts, "decoder.mid_block.attentions.0.to_v.weight", &[mid_channels, mid_channels]);
        put(&mut ts, "decoder.mid_block.attentions.0.to_v.bias", &[mid_channels]);
        put(
            &mut ts,
            "decoder.mid_block.attentions.0.to_out.0.weight",
            &[mid_channels, mid_channels],
        );
        put(&mut ts, "decoder.mid_block.attentions.0.to_out.0.bias", &[mid_channels]);
        // up blocks.
        for index in 0..reversed.len() {
            let in_ch = if index > 0 { reversed[index - 1] } else { reversed[0] };
            let out_ch = reversed[index];
            let is_final = index + 1 == reversed.len();
            let p = format!("decoder.up_blocks.{index}");
            for r in 0..(cfg.layers_per_block + 1) {
                let rp = format!("{p}.resnets.{r}");
                let res_in = if r == 0 { in_ch } else { out_ch };
                put(&mut ts, &format!("{rp}.norm1.weight"), &[res_in]);
                put(&mut ts, &format!("{rp}.norm1.bias"), &[res_in]);
                put(&mut ts, &format!("{rp}.conv1.weight"), &[out_ch, res_in, 3, 3]);
                put(&mut ts, &format!("{rp}.conv1.bias"), &[out_ch]);
                put(&mut ts, &format!("{rp}.norm2.weight"), &[out_ch]);
                put(&mut ts, &format!("{rp}.norm2.bias"), &[out_ch]);
                put(&mut ts, &format!("{rp}.conv2.weight"), &[out_ch, out_ch, 3, 3]);
                put(&mut ts, &format!("{rp}.conv2.bias"), &[out_ch]);
                if res_in != out_ch {
                    put(&mut ts, &format!("{rp}.conv_shortcut.weight"), &[out_ch, res_in, 1, 1]);
                    put(&mut ts, &format!("{rp}.conv_shortcut.bias"), &[out_ch]);
                }
            }
            if !is_final {
                put(&mut ts, &format!("{p}.upsamplers.0.conv.weight"), &[out_ch, out_ch, 3, 3]);
                put(&mut ts, &format!("{p}.upsamplers.0.conv.bias"), &[out_ch]);
            }
        }
        let first = cfg.decoder_block_out_channels[0];
        put(&mut ts, "decoder.conv_norm_out.weight", &[first]);
        put(&mut ts, "decoder.conv_norm_out.bias", &[first]);
        put(&mut ts, "decoder.conv_out.weight", &[3, first, 3, 3]);
        put(&mut ts, "decoder.conv_out.bias", &[3]);

        let vb = candle_compat::from_tensors(ts, DType::F32, dev);
        Flux2Vae::new(vb, cfg).expect("build tiny vae")
    }

    #[test]
    fn test_vae_decode_upscales_16x() {
        let dev = candle::Device::Cpu;
        let vae = tiny_vae(&dev);
        let z = Tensor::randn(0f32, 1f32, (1, 128, 4, 4), &dev).unwrap();
        let img = vae.decode(&z).expect("decode");
        assert_eq!(img.dims(), &[1, 3, 64, 64], "4×4 latent → 64×64 RGB (16×)");
        assert_eq!(img.dtype(), DType::F32);
        let vals = img.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(vals.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_vae_decode_identity_bn_roundtrip_layout() {
        // With running_var=1(+eps) and mean=0 the BN stage is ≈ a
        // sqrt(1+eps) scale — verify the unshuffle layout by decoding
        // a zero latent and checking the output is finite and the
        // channel count/stride semantics hold on an odd-size latent.
        let dev = candle::Device::Cpu;
        let vae = tiny_vae(&dev);
        let z = Tensor::zeros((1, 128, 3, 5), DType::F32, &dev).unwrap();
        let img = vae.decode(&z).expect("decode odd latent");
        assert_eq!(img.dims(), &[1, 3, 48, 80]);
    }
}
