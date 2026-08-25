//! Mage-ViT token-source contract.
//!
//! The StreamMind gate consumes *encoded* vision tokens `[T, P, D]`
//! (D = 2560, P = patches per codec window). The full Mage-ViT
//! encoder that produces them lands in a follow-up commit; this file
//! pins the interface so the gate, the HTTP surface, and tests can be
//! built against it now.
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
