//! Token-level multimodal glue: detecting the placeholder slots
//! inside `input_ids` and scattering the vision encoder's merged
//! tokens into them before the LLM forward pass. Mirrors the
//! reference `MageVLModel.get_placeholder_mask` + `masked_scatter`
//! pair from `modeling_mage_vl.py`.
//!
//! Why a standalone module: the scatter logic depends on a stable
//! `PlaceholderTokens` constant from `config.rs` and has nothing to
//! do with the vision / language model internals — splitting it out
//! keeps the surface testable without spinning up a full model.

use candle::{DType, Device, IndexOp, Result, Tensor};

use super::config::PlaceholderTokens;

/// Locate the placeholder slots inside `input_ids` that the vision
/// encoder should fill in. Returns a flat `[N, 2]` int64 tensor of
/// `(batch, seq_len)` indices suitable for `scatter`.
///
/// `kind = Image` matches `placeholders.image_token_id`;
/// `kind = Video` matches `placeholders.video_token_id`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaceholderKind {
    Image,
    Video,
}

pub fn placeholder_positions(
    input_ids: &Tensor,
    placeholders: &PlaceholderTokens,
    kind: PlaceholderKind,
) -> Result<Tensor> {
    let token_id = match kind {
        PlaceholderKind::Image => placeholders.image_token_id,
        PlaceholderKind::Video => placeholders.video_token_id,
    };
    let target = Tensor::new(&[token_id], input_ids.device())?.to_dtype(input_ids.dtype())?;
    let mask = input_ids.broadcast_eq(&target)?; // [B, L] bool
    // Walk the mask row-major and record each true position as
    // (batch, col). candle doesn't expose nonzero on bool tensors
    // directly, so we iterate the row count and gather via the
    // .i() IndexOp trait. Real workloads touch O(merged_token_count)
    // entries — small relative to sequence length.
    let l = input_ids.dim(1)?;
    let mut rows: Vec<u32> = Vec::new();
    let mut cols: Vec<u32> = Vec::new();
    let flat = mask.flatten_all()?.to_dtype(DType::U8)?;
    let data: Vec<u8> = flat.to_vec1()?;
    for (idx, &v) in data.iter().enumerate() {
        if v != 0 {
            rows.push((idx / l) as u32);
            cols.push((idx % l) as u32);
        }
    }
    let rows = Tensor::from_vec(rows.clone(), (rows.len(),), input_ids.device())?.to_dtype(DType::I64)?;
    let cols = Tensor::from_vec(cols.clone(), (cols.len(),), input_ids.device())?.to_dtype(DType::I64)?;
    Tensor::stack(&[&rows, &cols], 1)
}

/// Scatter `merged` (shape `[N, D]`) into `inputs_embeds` at the
/// positions indicated by `positions` (shape `[N, 2]` int64 of
/// `(batch, seq_len)` indices). Returns a new `inputs_embeds`
/// tensor; the receiver is left untouched.
pub fn scatter_merged(
    inputs_embeds: &Tensor,
    merged: &Tensor,
    positions: &Tensor,
) -> Result<Tensor> {
    let n = positions.dim(0)?;
    let d = merged.dim(1)?;
    let total = inputs_embeds.dim(1)?;
    // Build a flat-index view of merged so we can assign it to the
    // flat index of inputs_embeds.
    let flat = inputs_embeds
        .reshape((inputs_embeds.dim(0)?, total, d))?
        .contiguous()?;
    let flat = flat.transpose(1, 2)?.contiguous()?; // [B, D, L]
    let positions_flat = positions.i((.., 0))?.contiguous()?; // [N] batch idx
    let positions_col = positions.i((.., 1))?.contiguous()?; // [N] col idx
    let _ = total;
    // merged as [B, D, L] view — start from a zero-filled [B, D, L]
    // and use candle's `index_add` equivalent via `scatter`.
let mut out = flat.clone();
    for i in 0..n {
        let b = positions_flat.i(i)?.to_scalar::<i64>()? as usize;
        let c = positions_col.i(i)?.to_scalar::<i64>()? as usize;
        let src = merged.i(i)?.reshape((d, 1))?.unsqueeze(0)?; // [1, D, 1]
        // The scatter is column-wise in the seq axis, so concat
        // along dim=2 (the seq length axis). Use `narrow` rather
        // than `i(..c+1..)` so candle keeps the rank explicit.
        let prefix = out.narrow(2, 0, c)?; // [B, D, c]
        let suffix = out.narrow(2, c + 1, total - c - 1)?; // [B, D, L-c-1]
        out = Tensor::cat(&[&prefix, &src, &suffix], 2)?;
    }
    out.transpose(1, 2)?.contiguous()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholder_positions_finds_image_token_indices() -> Result<()> {
        let placeholder_id: u32 = 151_655;
        let ids = Tensor::from_slice(
            &[1u32, 2u32, placeholder_id, 3u32, placeholder_id],
            (1usize, 5usize),
            &Device::Cpu,
        )?;
        let placeholders = PlaceholderTokens {
            image_token_id: placeholder_id,
            ..PlaceholderTokens::default()
        };
        let positions =
            placeholder_positions(&ids, &placeholders, PlaceholderKind::Image)?;
        assert_eq!(positions.dim(0).unwrap(), 2);
        assert_eq!(positions.dim(1).unwrap(), 2);
        let r0 = positions.i(0)?.to_vec1::<i64>()?;
        let r1 = positions.i(1)?.to_vec1::<i64>()?;
        assert_eq!(r0, vec![0, 2]);
        assert_eq!(r1, vec![0, 4]);
        Ok(())
    }

    #[test]
    fn scatter_merged_replaces_placeholder_columns() -> Result<()> {
        let embeds = Tensor::zeros((1usize, 3usize, 4usize), DType::F32, &Device::Cpu)?;
        let merged = Tensor::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            (2usize, 4usize),
            &Device::Cpu,
        )?;
        let positions = Tensor::from_slice(
            &[0i64, 1i64, 0i64, 2i64],
            (2usize, 2usize),
            &Device::Cpu,
        )?;
        let out = scatter_merged(&embeds, &merged, &positions)?;
        assert_eq!(out.shape(), embeds.shape());
        // out is [B, L, D] = [1, 3, 4]. Read as a flat row-major slice.
        let r0: Vec<f32> = out.flatten_all()?.to_vec1()?;
        assert_eq!(&r0[0..4], &[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(&r0[4..8], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(&r0[8..12], &[5.0, 6.0, 7.0, 8.0]);
        Ok(())
    }
}