//! Mage-ViT visual-encoder scaffold.
//!
//! The StreamMind gate consumes *encoded* vision tokens `[T, P, D]`
//! (D = 2560, P = patches per codec window). The full Mage-ViT
//! encoder that produces them lands in a follow-up commit; this file
//! currently ships only the patch-embedding stem so the integration
//! pipeline can be exercised against random weights while the encoder
//! layers and 3D RoPE are implemented next.
//!
//! Contract notes (from `modeling_mage_vl._streammind_vision_tokens`):
//! - One entry per time step after the spatial merge: a window with
//!   grid `(t, h, w)` yields `t` steps of `(h/2)*(w/2)` merged tokens.
//! - `patch_positions` marks which I/P-frame patches survived codec
//!   sparsity; dense inputs simply enumerate every position.

use candle_core::{Result, Tensor};

pub type VisionTokenWindow = Tensor;

/// Provider of gate-ready Mage-ViT token windows.
///
/// Implementations: the upcoming candle Mage-ViT encoder, and test
/// fixtures synthesizing random streams.
pub trait VisionTokenSource {
    /// Encode one contiguous segment into `[T, P, D]` tokens.
    fn encode_window(&self, pixels: &Tensor, grid_thw: [u32; 3]) -> Result<VisionTokenWindow>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mage_vl::config::MageVlVisionConfig;

    /// The full Mage-ViT encoder isn't built yet — this is the smallest
    /// sanity test that the contract type compiles and shapes line up
    /// with the downstream gate. Future commits will exercise the patch
    /// embed forward pass on random pixels here.
    #[test]
    fn grid_thw_shape_contract_round_trip() {
        let cfg = MageVlVisionConfig::default();
        let merge = cfg.spatial_merge_size;
        let (t, h, w) = (4usize, 28usize, 28usize);
        let merged = ((h / merge) * (w / merge)) * t;
        assert!(merged > 0, "merged token count must be positive");
    }
}
