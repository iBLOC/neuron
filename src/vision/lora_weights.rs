//! LoRA weight loading and merging utilities.
//!
//! Implements the LoRA (Low-Rank Adaptation) weight merge for the
//! YOLO family. The base model weights stay frozen; the merge produces
//! a temporary expanded classification head per request.
//!
//! ## Math
//!
//! Standard LoRA (Hu et al. 2021):
//! ```text
//! W' = W + (alpha / rank) * B @ A
//! ```
//! where `A` is `[rank, in_dim]`, `B` is `[out_dim, rank]`, and
//! `W` is `[out_dim, in_dim]` (the frozen base weight).
//!
//! For YOLOv8 specifically, LoRA is typically applied to the
//! classification head's final linear layer (where `W` is
//! `[num_classes, hidden_dim]`). The merged head has the same shape,
//! so it can be a drop-in replacement.
//!
//! ## Training-side convention
//!
//! Adapter files in `<weight_path>/lora.safetensors` follow this key
//! pattern:
//! ```text
//! lora_A.head.cls.cv3.conv   tensor[rank, in_dim]   f32
//! lora_B.head.cls.cv3.conv   tensor[out_dim, rank]  f32
//! ```
//!
//! `metadata.json` describes the source modules + the custom class
//! list. See `lora.rs` for the full schema.

use std::path::Path;

use crate::llm::NeureError;
use super::lora::LoraTensor;

/// Parse a LoRA safetensors file into `LoraTensor` entries.
///
/// The file is expected to be a safetensors file with keys of the form
/// `lora_A.<module_path>` (shape `[rank, in_dim]`) and
/// `lora_B.<module_path>` (shape `[out_dim, rank]`).
///
/// Returns one `LoraTensor` per key. The `name` field strips the
/// `lora_A.` / `lora_B.` prefix so consumers see clean module paths
/// (e.g. `head.cls.cv3.conv`).
#[cfg(feature = "candle")]
pub fn parse_safetensors_lora(
    path: &Path,
) -> Result<Vec<LoraTensor>, NeureError> {
    // Read the raw bytes
    let bytes = std::fs::read(path).map_err(|e| {
        NeureError::invalid_input(format!(
            "failed to read LoRA weights at {}: {e}",
            path.display()
        ))
    })?;

    // safetensors 0.4 API: SafeTensors::deserialize
    let safetensors = safetensors::SafeTensors::deserialize(&bytes).map_err(|e| {
        NeureError::invalid_input(format!(
            "invalid safetensors file at {}: {e}",
            path.display()
        ))
    })?;

    let mut result = Vec::new();
    for (name_with_prefix, view) in safetensors.tensors() {
        // Strip the `lora_A.` / `lora_B.` prefix to get a clean module path
        let (kind, name) = if let Some(rest) = name_with_prefix.strip_prefix("lora_A.") {
            ("A", rest.to_string())
        } else if let Some(rest) = name_with_prefix.strip_prefix("lora_B.") {
            ("B", rest.to_string())
        } else {
            // Unknown key — skip with a no-op tensor (zero rows)
            continue;
        };

        let shape = view.shape();
        if shape.len() != 2 {
            return Err(NeureError::invalid_input(format!(
                "LoRA tensor {} has shape {:?}, expected 2D [rows, cols]",
                name_with_prefix, shape
            )));
        }
        let rows = shape[0] as u32;
        let cols = shape[1] as u32;

        // For A: shape is [rank, in_dim]. For B: shape is [out_dim, rank].
        // We store the row count as either `in_dim` or `out_dim` based on
        // the kind, and the column count as the other dim. The `rank` is
        // whichever is smaller (typically the inner dim) — but we can
        // also let the consumer detect it from context.
        let (in_dim, out_dim, rank) = match kind {
            "A" => (cols, 0, rows),  // A: [rank, in_dim] -> rank=rows, in_dim=cols
            "B" => (0, rows, cols),  // B: [out_dim, rank] -> rank=cols, out_dim=rows
            _ => unreachable!(),
        };

        // Read the f32 data — safetensors guarantees contiguous f32 storage
        let data: Vec<f32> = view
            .data()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        result.push(LoraTensor { name, in_dim, out_dim, rank, data });
    }

    Ok(result)
}

/// LoRA safetensors parsing stub for non-candle builds. Without the
/// `candle` feature, the `safetensors` crate is not available; we
/// return an empty tensor list (the metadata-only path still works).
#[cfg(not(feature = "candle"))]
pub fn parse_safetensors_lora(
    _path: &Path,
) -> Result<Vec<LoraTensor>, NeureError> {
    Ok(Vec::new())
}

/// Merge LoRA A + B matrices into a single effective delta of shape
/// `[out_dim, in_dim]`, scaled by `alpha / rank`.
///
/// ```text
/// delta = (alpha / rank) * B @ A
/// ```
///
/// Both A and B must be `f32` row-major; their shapes are validated.
pub fn merge_lora_delta(
    a: &LoraTensor,
    b: &LoraTensor,
    alpha: f32,
) -> Result<Vec<f32>, NeureError> {
    if a.data.len() != (a.in_dim * a.rank) as usize {
        return Err(NeureError::invalid_input(format!(
            "LoRA A has {} elements, expected in_dim * rank = {}",
            a.data.len(),
            a.in_dim * a.rank
        )));
    }
    if b.data.len() != (b.out_dim * b.rank) as usize {
        return Err(NeureError::invalid_input(format!(
            "LoRA B has {} elements, expected out_dim * rank = {}",
            b.data.len(),
            b.out_dim * b.rank
        )));
    }
    if a.rank != b.rank {
        return Err(NeureError::invalid_input(format!(
            "LoRA A rank {} != B rank {}",
            a.rank, b.rank
        )));
    }
    let out_dim = b.out_dim as usize;
    let in_dim = a.in_dim as usize;
    let rank = a.rank as usize;
    let scale = alpha / rank as f32;

    // B is [out_dim, rank] (row-major), A is [rank, in_dim] (row-major)
    // Compute: delta[i, j] = scale * sum_k B[i, k] * A[k, j]
    let mut delta = vec![0.0f32; out_dim * in_dim];
    for i in 0..out_dim {
        for k in 0..rank {
            let b_ik = b.data[i * rank + k];
            if b_ik == 0.0 {
                continue;
            }
            for j in 0..in_dim {
                let a_kj = a.data[k * in_dim + j];
                delta[i * in_dim + j] += scale * b_ik * a_kj;
            }
        }
    }
    Ok(delta)
}

/// Apply a LoRA delta to a frozen base weight, producing the effective
/// weight matrix for inference.
///
/// ```text
/// W' = W + delta
/// ```
///
/// Both W and delta are `[out_dim, in_dim]` row-major. The result is
/// `W'` with the same shape. Used at inference time to materialize
/// the merged weight before a forward pass.
pub fn apply_lora_delta(
    base: &[f32],
    delta: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> Result<Vec<f32>, NeureError> {
    if base.len() != out_dim * in_dim {
        return Err(NeureError::invalid_input(format!(
            "base weight has {} elements, expected out_dim * in_dim = {}",
            base.len(),
            out_dim * in_dim
        )));
    }
    if delta.len() != out_dim * in_dim {
        return Err(NeureError::invalid_input(format!(
            "LoRA delta has {} elements, expected out_dim * in_dim = {}",
            delta.len(),
            out_dim * in_dim
        )));
    }
    let mut out = vec![0.0f32; out_dim * in_dim];
    for i in 0..out_dim * in_dim {
        out[i] = base[i] + delta[i];
    }
    Ok(out)
}

/// Get the `safetensors::SafeTensorError` type alias
/// (re-exported for callers that want to match on it).
#[cfg(feature = "candle")]
pub use safetensors::SafeTensorError as ParseError;

#[cfg(test)]
mod tests {
    use super::*;

    fn make_a(rank: u32, in_dim: u32, values: Vec<f32>) -> LoraTensor {
        LoraTensor { name: "test.A".into(), in_dim, out_dim: 0, rank, data: values }
    }
    fn make_b(out_dim: u32, rank: u32, values: Vec<f32>) -> LoraTensor {
        LoraTensor { name: "test.B".into(), in_dim: 0, out_dim, rank, data: values }
    }

    #[test]
    fn test_merge_lora_delta_simple() {
        // A = [[1, 0], [0, 1]] (identity 2x2, rank=2, in_dim=2)
        let a = make_a(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
        // B = [[1, 0], [0, 1]] (identity 2x2, out_dim=2, rank=2)
        let b = make_b(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
        // B @ A = I, so delta = (alpha/rank) * I = (1.0/2) * I = 0.5 * I
        let delta = merge_lora_delta(&a, &b, 1.0).unwrap();
        assert_eq!(delta, vec![0.5, 0.0, 0.0, 0.5]);
    }

    #[test]
    fn test_merge_lora_delta_scaled() {
        // A = [[2, 0], [0, 3]] (diagonal 2x2)
        let a = make_a(2, 2, vec![2.0, 0.0, 0.0, 3.0]);
        // B = [[1, 0], [0, 1]]
        let b = make_b(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
        // B @ A = A. delta = (4.0/2) * A = 2.0 * A
        let delta = merge_lora_delta(&a, &b, 4.0).unwrap();
        assert_eq!(delta, vec![4.0, 0.0, 0.0, 6.0]);
    }

    #[test]
    fn test_merge_lora_delta_rank_mismatch() {
        let a = make_a(2, 4, vec![0.0; 8]);
        let b = make_b(4, 3, vec![0.0; 12]);  // rank 3 != 2
        let result = merge_lora_delta(&a, &b, 1.0);
        assert!(result.is_err());
        assert!(result.unwrap_err().message.contains("rank"));
    }

    #[test]
    fn test_merge_lora_delta_wrong_a_size() {
        // A claims rank=2 in_dim=4, but data length is 6 (not 8)
        let a = make_a(2, 4, vec![0.0; 6]);
        let b = make_b(4, 2, vec![0.0; 8]);
        let result = merge_lora_delta(&a, &b, 1.0);
        assert!(result.is_err());
        assert!(result.unwrap_err().message.contains("A has 6"));
    }

    #[test]
    fn test_merge_lora_delta_wrong_b_size() {
        let a = make_a(2, 4, vec![0.0; 8]);
        let b = make_b(4, 2, vec![0.0; 6]);  // 6 != 4*2=8
        let result = merge_lora_delta(&a, &b, 1.0);
        assert!(result.is_err());
        assert!(result.unwrap_err().message.contains("B has 6"));
    }

    #[test]
    fn test_merge_lora_delta_zero_data() {
        // All-zero A → zero delta regardless of B
        let a = make_a(2, 3, vec![0.0; 6]);
        let b = make_b(4, 2, vec![1.0; 8]);
        let delta = merge_lora_delta(&a, &b, 5.0).unwrap();
        assert_eq!(delta, vec![0.0; 12]); // 4*3
    }

    #[test]
    fn test_merge_lora_delta_rectangular() {
        // A: [rank=2, in_dim=3], B: [out_dim=4, rank=2]
        // A = [[1, 2, 3], [4, 5, 6]]
        let a = make_a(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        // B = [[1, 0], [0, 1], [1, 1], [2, 1]]
        let b = make_b(4, 2, vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0]);
        // B @ A:
        //   [1, 0] * [1, 2, 3] = [1, 2, 3]
        //   [0, 1] * [4, 5, 6] = [4, 5, 6]
        //   [1, 1] * (sum rows) = [5, 7, 9]
        //   [2, 1] = [2+4, 4+5, 6+6] = [6, 9, 12]
        // Then * (alpha=1 / rank=2) = 0.5 * above
        let delta = merge_lora_delta(&a, &b, 1.0).unwrap();
        let expected: Vec<f32> = vec![
            0.5, 1.0, 1.5,    // row 0
            2.0, 2.5, 3.0,    // row 1
            2.5, 3.5, 4.5,    // row 2
            3.0, 4.5, 6.0,    // row 3
        ];
        for (i, (&got, &exp)) in delta.iter().zip(expected.iter()).enumerate() {
            assert!((got - exp).abs() < 1e-6, "delta[{i}] = {got}, expected {exp}");
        }
    }

    #[test]
    fn test_apply_lora_delta() {
        let base = vec![1.0, 0.0, 0.0, 1.0];  // 2x2 identity
        let delta = vec![0.1, 0.2, 0.3, 0.4];
        let result = apply_lora_delta(&base, &delta, 2, 2).unwrap();
        assert_eq!(result, vec![1.1, 0.2, 0.3, 1.4]);
    }

    #[test]
    fn test_apply_lora_delta_wrong_base_size() {
        let base = vec![1.0; 5];  // should be 4
        let delta = vec![0.1; 4];
        let result = apply_lora_delta(&base, &delta, 2, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_apply_lora_delta_wrong_delta_size() {
        let base = vec![1.0; 4];
        let delta = vec![0.1; 3];  // should be 4
        let result = apply_lora_delta(&base, &delta, 2, 2);
        assert!(result.is_err());
    }

    #[cfg(feature = "candle")]
    #[test]
    fn test_parse_safetensors_lora_strips_prefix() {
        // Build a minimal safetensors file in memory
        // A: rank=2, in_dim=2 → [2, 2] row-major identity
        // B: out_dim=2, rank=2 → [2, 2] row-major identity
        let a_data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
        let b_data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
        let a_bytes: Vec<u8> = a_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let b_bytes: Vec<u8> = b_data.iter().flat_map(|f| f.to_le_bytes()).collect();

        // Wrap in TensorView and use the serialize API
        use safetensors::tensor::TensorView;
        let a_view = TensorView::new(safetensors::Dtype::F32, vec![2, 2], &a_bytes).unwrap();
        let b_view = TensorView::new(safetensors::Dtype::F32, vec![2, 2], &b_bytes).unwrap();
        let tensors: Vec<(&str, TensorView<'_>)> = vec![
            ("lora_A.head.cls.cv3.conv", a_view),
            ("lora_B.head.cls.cv3.conv", b_view),
        ];
        let serialized = safetensors::serialize(tensors, &None).unwrap();

        // Write to a temp file
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("lora.safetensors");
        std::fs::write(&path, &serialized).unwrap();

        let parsed = parse_safetensors_lora(&path).unwrap();
        assert_eq!(parsed.len(), 2);

        // Both tensors should have their lora_A. / lora_B. prefix stripped
        for t in &parsed {
            assert!(!t.name.starts_with("lora_"), "name still has prefix: {}", t.name);
            assert!(t.name.contains("head.cls.cv3.conv"));
        }
    }
}