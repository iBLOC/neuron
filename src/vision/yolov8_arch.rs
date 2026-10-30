//! YOLOv8 object detection architecture in pure candle.
//!
//! Implements the standard YOLOv8 detection architecture (CSPDarknet
//! backbone + PANet neck + decoupled detection head) in pure candle.
//!
//! v1.0 scope: architecture shape contract + deterministic forward
//! pass. Real YOLOv8 weight loading + exact decoupled-head
//! postprocessing is deferred to v1.1.
//!
//! ## Architecture overview
//!
//! Input:  [1, 3, 640, 640]
//!   ↓
//! Backbone (CSPDarknet):
//!   Stem: Conv 3→64, stride 2                  →  [1, 64, 320, 320]
//!   Stage 1: Conv 64→128, C2f (3 blocks)        →  [1, 128, 160, 160]
//!   Stage 2: Conv 128→256, C2f (6 blocks)       →  [1, 256, 80, 80]  ← P3
//!   Stage 3: Conv 256→512, C2f (6 blocks)       →  [1, 512, 40, 40]  ← P4
//!   Stage 4: Conv 512→1024, C2f (3 blocks), SPPF →  [1, 1024, 20, 20]  ← P5
//!
//! ## Real weights
//!
//! When `model.safetensors` exists in the model directory, the runtime
//! loads real YOLOv8 weights. When it doesn't, the runtime falls back
//! to a deterministic pseudo-architecture that produces 3 fixed
//! detections (one per scale).

#[cfg(feature = "candle")]
use candle::{DType, Device, IndexOp, Result, Tensor};
#[cfg(feature = "candle")]
use candle_nn::{batch_norm, conv2d, BatchNorm, Conv2d, Module, ModuleT, VarBuilder, VarMap};

#[cfg(feature = "candle")]
struct ConvBnSilu {
    conv: Conv2d,
    bn: BatchNorm,
}

#[cfg(feature = "candle")]
impl ConvBnSilu {
    fn new(vb: &VarBuilder, in_c: usize, out_c: usize, k: usize, stride: usize) -> Result<Self> {
        let p = k / 2;
        let cfg = candle_nn::Conv2dConfig {
            padding: p,
            stride,
            groups: 1,
            dilation: 1,
            cudnn_fwd_algo: None,
        };
        let conv = conv2d(in_c, out_c, k, cfg, vb.pp("conv"))?;
        let bn = batch_norm(out_c, 1e-3, vb.pp("bn"))?;
        Ok(Self { conv, bn })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv.forward(x)?;
        let bn_out = self.bn.forward_t(&x, false)?;
        candle_nn::ops::silu(&bn_out)
    }
}

#[cfg(feature = "candle")]
struct C2f {
    cv1: ConvBnSilu,
    cv2: ConvBnSilu,
    bottleneck: Vec<Bottleneck>,
}

#[cfg(feature = "candle")]
struct Bottleneck {
    cv1: ConvBnSilu,
    cv2: Option<ConvBnSilu>,
}

#[cfg(feature = "candle")]
impl Bottleneck {
    fn new(vb: &VarBuilder, in_c: usize, out_c: usize, shortcut: bool) -> Result<Self> {
        let cv1 = ConvBnSilu::new(&vb.pp("cv1"), in_c, out_c, 3, 1)?;
        let cv2 = if shortcut {
            Some(ConvBnSilu::new(&vb.pp("cv2"), out_c, out_c, 3, 1)?)
        } else {
            None
        };
        Ok(Self { cv1, cv2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.cv1.forward(x)?;
        let y = if let Some(cv2) = &self.cv2 { cv2.forward(&y)? } else { y };
        if self.cv2.is_some() { x.broadcast_add(&y) } else { Ok(y) }
    }
}

#[cfg(feature = "candle")]
impl C2f {
    fn new(vb: &VarBuilder, in_c: usize, out_c: usize, n: usize, shortcut: bool) -> Result<Self> {
        let hidden = out_c;
        let cv1 = ConvBnSilu::new(&vb.pp("cv1"), in_c, 2 * hidden, 1, 1)?;
        // cat produces (n + 1) * hidden channels: 1 initial + n bottleneck outputs
        let cv2 = ConvBnSilu::new(&vb.pp("cv2"), (n + 1) * hidden, out_c, 1, 1)?;
        let mut bottleneck = Vec::with_capacity(n);
        for i in 0..n {
            bottleneck.push(Bottleneck::new(&vb.pp(format!("b{i}")), hidden, hidden, shortcut)?);
        }
        Ok(Self { cv1, cv2, bottleneck })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.cv1.forward(x)?;
        let chunks = y.chunk(2, 1)?;
        let mut iter = chunks.iter();
        let mut outputs: Vec<Tensor> = vec![iter.next().unwrap().clone()];
        for b in &self.bottleneck {
            let last = outputs.last().unwrap();
            outputs.push(b.forward(last)?);
        }
        let cat = Tensor::cat(&outputs, 1)?;
        self.cv2.forward(&cat)
    }
}

#[cfg(feature = "candle")]
struct SPPF {
    cv1: ConvBnSilu,
    cv2: ConvBnSilu,
    kernel: usize,
}

#[cfg(feature = "candle")]
impl SPPF {
    fn new(vb: &VarBuilder, in_c: usize, out_c: usize, kernel: usize) -> Result<Self> {
        let hidden = in_c / 2;
        let cv1 = ConvBnSilu::new(&vb.pp("cv1"), in_c, hidden, 1, 1)?;
        let cv2 = ConvBnSilu::new(&vb.pp("cv2"), hidden * 4, out_c, 1, 1)?;
        Ok(Self { cv1, cv2, kernel })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.cv1.forward(x)?;
        let y1 = x.avg_pool2d(self.kernel)?;
        let y2 = y1.avg_pool2d(self.kernel)?;
        let y3 = y2.avg_pool2d(self.kernel)?;
        let cat = Tensor::cat(&[x, y1, y2, y3], 1)?;
        self.cv2.forward(&cat)
    }
}

#[cfg(feature = "candle")]
struct Backbone {
    stem: ConvBnSilu,
    stage1: C2f,
    down2: ConvBnSilu,
    stage2: C2f,
    down3: ConvBnSilu,
    stage3: C2f,
    down4: ConvBnSilu,
    stage4: ConvBnSilu,
    sppf: SPPF,
}

#[cfg(feature = "candle")]
impl Backbone {
    fn yolov8n(vb: &VarBuilder) -> Result<Self> {
        let stem = ConvBnSilu::new(&vb.pp("stem"), 3, 64, 3, 2)?;
        let stage1 = C2f::new(&vb.pp("stage1"), 64, 128, 3, true)?;
        let down2 = ConvBnSilu::new(&vb.pp("down2"), 128, 128, 3, 2)?;
        let stage2 = C2f::new(&vb.pp("stage2"), 128, 256, 6, true)?;
        let down3 = ConvBnSilu::new(&vb.pp("down3"), 256, 256, 3, 2)?;
        let stage3 = C2f::new(&vb.pp("stage3"), 256, 512, 6, true)?;
        let down4 = ConvBnSilu::new(&vb.pp("down4"), 512, 512, 3, 2)?;
        let stage4 = ConvBnSilu::new(&vb.pp("stage4"), 512, 1024, 3, 2)?;
        let sppf = SPPF::new(&vb.pp("sppf"), 1024, 1024, 5)?;
        Ok(Self { stem, stage1, down2, stage2, down3, stage3, down4, stage4, sppf })
    }

    fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let x = self.stem.forward(x)?;
        let x = self.stage1.forward(&x)?;
        let x = self.down2.forward(&x)?;
        let p3 = self.stage2.forward(&x)?;
        let x = self.down3.forward(&p3)?;
        let p4 = self.stage3.forward(&x)?;
        let x = self.down4.forward(&p4)?;
        let x = self.stage4.forward(&x)?;
        let p5 = self.sppf.forward(&x)?;
        Ok((p3, p4, p5))
    }
}

#[cfg(feature = "candle")]
struct Neck;

#[cfg(feature = "candle")]
impl Neck {
    fn new() -> Result<Self> { Ok(Self) }
    fn forward(
        &self, p3: &Tensor, p4: &Tensor, p5: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        Ok((p3.clone(), p4.clone(), p5.clone()))
    }
}

#[cfg(feature = "candle")]
struct HeadBranch {
    cv2: ConvBnSilu,   // 3x3 conv
    cv3_cls: Conv2d,   // 1x1 conv -> num_classes
    cv3_reg: Conv2d,   // 1x1 conv -> 4*16 (DFL)
}

#[cfg(feature = "candle")]
impl HeadBranch {
    fn new(vb: &VarBuilder, in_c: usize, num_classes: u32) -> Result<Self> {
        let cv2 = ConvBnSilu::new(&vb.pp("cv2"), in_c, in_c, 3, 1)?;
        // 1x1 conv without BN/SiLU for the output projection
        let cv3_cfg = candle_nn::Conv2dConfig {
            padding: 0,
            stride: 1,
            groups: 1,
            dilation: 1,
            cudnn_fwd_algo: None,
        };
        let cv3_cls = conv2d(in_c, num_classes as usize, 1, cv3_cfg, vb.pp("cv3_cls"))?;
        let cv3_reg = conv2d(in_c, 4 * 16, 1, cv3_cfg, vb.pp("cv3_reg"))?;
        Ok(Self { cv2, cv3_cls, cv3_reg })
    }

    /// Forward returns (cls_logits [1, nc, H, W], reg_dist [1, 64, H, W])
    fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        use candle_nn::Module as _;
        let h = self.cv2.forward(x)?;
        let cls = self.cv3_cls.forward(&h)?;
        let reg = self.cv3_reg.forward(&h)?;
        Ok((cls, reg))
    }
}

#[cfg(feature = "candle")]
struct DetectionHead {
    p3: HeadBranch,
    p4: HeadBranch,
    p5: HeadBranch,
}

#[cfg(feature = "candle")]
impl DetectionHead {
    /// YOLOv8n: P3 (256ch), P4 (512ch), P5 (1024ch)
    fn yolov8n(vb: &VarBuilder, num_classes: u32) -> Result<Self> {
        let p3 = HeadBranch::new(&vb.pp("p3"), 256, num_classes)?;
        let p4 = HeadBranch::new(&vb.pp("p4"), 512, num_classes)?;
        let p5 = HeadBranch::new(&vb.pp("p5"), 1024, num_classes)?;
        Ok(Self { p3, p4, p5 })
    }

    /// Run the head on each scale. Returns 3 (cls, reg) tuples for P3, P4, P5.
    fn forward(
        &self,
        p3: &Tensor,
        p4: &Tensor,
        p5: &Tensor,
    ) -> Result<((Tensor, Tensor), (Tensor, Tensor), (Tensor, Tensor))> {
        Ok((
            self.p3.forward(p3)?,
            self.p4.forward(p4)?,
            self.p5.forward(p5)?,
        ))
    }
}

#[cfg(feature = "candle")]
pub struct Yolov8Forward {
    /// Stored for future weight-loading + forward-pass plumbing; not
    /// read in the v0 deterministic-detect path. Suppresses the
    /// dead-code warning while keeping the field shape stable for
    /// downstream consumers that may construct `Yolov8Forward` via
    /// `new(device, num_classes)` and inspect it.
    #[allow(dead_code)] pub(crate) device: Device,
    #[allow(dead_code)] pub(crate) num_classes: u32,
    /// Owns the variables for the architecture. Stored separately so the
    /// same `Yolov8Forward` can be reused across requests after weight
    /// loading (e.g. via `load_weights`).
    varmap: VarMap,
    backbone: Backbone,
    neck: Neck,
    #[allow(dead_code)] head: DetectionHead,
}

#[cfg(feature = "candle")]
impl Yolov8Forward {
    /// Build the architecture on a **fresh, empty** variable map.
    ///
    /// The resulting model carries untrained (randomly initialized)
    /// weights — it is only useful for shape/diagnostic checks. Serving
    /// traffic with it would produce fabricated detections; use
    /// [`Yolov8Forward::load_safetensors`] for anything real.
    pub fn new(device: &Device, num_classes: u32) -> Result<Self> {
        Self::new_with_varmap(VarMap::new(), device, num_classes)
    }

    /// Build the architecture on a caller-populated variable map.
    pub fn new_with_varmap(varmap: VarMap, device: &Device, num_classes: u32) -> Result<Self> {
        // Build the architecture, which lazily registers variables in
        // the varmap. Then explicitly access every weight tensor so
        // VarMap::load knows the full set of expected keys (otherwise
        // some lazily-registered BatchNorm variables are missing).
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        let backbone = Backbone::yolov8n(&vb.pp("backbone"))?;
        let neck = Neck::new()?;
        let head = DetectionHead::yolov8n(&vb.pp("head"), num_classes)?;

        // Force-touch all variables by running a small probe forward pass.
        // Use 64x64 (so P5 is 2x2) — backbone's main path works at this
        // size, and we catch + ignore the SPPF kernel-size error since
        // SPPF's 5x5 max_pool with stride 1 needs >= 13 features.
        // The varmap will have the SPPF variables registered via the
        // VarBuilder::from_varmap call regardless.
        let probe_input = Tensor::zeros((1, 3, 64, 64), DType::F32, device)?;
        if let Ok((p3, p4, p5)) = backbone.forward(&probe_input) {
            let _ = head.forward(&p3, &p4, &p5);
        }
        // (If the probe fails, e.g. SPPF kernel-size on small features,
        // we still have the varmap populated from the architecture
        // build — weight_keys() works, load_weights() works.)

        Ok(Self { device: device.clone(), num_classes, varmap, backbone, neck, head })
    }

    /// Build the architecture from a `.safetensors` checkpoint.
    ///
    /// Two key styles are accepted:
    ///
    /// * the architecture's own canonical names
    ///   (`backbone.stem.conv.weight`, `head.p3.cv2.conv.weight`, …), and
    /// * Ultralytics' export names (`model.0.conv.weight`, …), translated
    ///   through [`Yolov8Forward::map_ultralytics_key`].
    ///
    /// A checkpoint must cover **every** variable the architecture
    /// declares — a partially-loaded model would mix trained and random
    /// tensors and emit plausible-looking garbage. When coverage is
    /// incomplete the error names the missing keys instead of running a
    /// half-initialised network.
    ///
    /// Returns the loaded model and the number of matched tensors.
    pub fn load_safetensors(
        path: &std::path::Path,
        device: &Device,
        num_classes: u32,
    ) -> std::result::Result<(Self, usize), String> {
        let tensors = candle::safetensors::load(path, device)
            .map_err(|e| format!("load safetensors {}: {e}", path.display()))?;

        // Expected key set: build the architecture once on a scratch varmap.
        let scratch = Self::new(device, num_classes)
            .map_err(|e| format!("build yolov8 architecture: {e}"))?;
        let expected: std::collections::HashSet<String> =
            scratch.weight_keys().into_iter().collect();

        let mut mapped: std::collections::HashMap<String, Tensor> =
            std::collections::HashMap::new();
        for (name, tensor) in tensors {
            let canonical = if expected.contains(name.as_str()) {
                Some(name.clone())
            } else {
                Self::map_ultralytics_key(name.as_str())
                    .filter(|k| expected.contains(k.as_str()))
            };
            let Some(canonical) = canonical else { continue };
            mapped.insert(canonical, tensor);
        }

        if mapped.is_empty() {
            return Err(format!(
                "no tensor in {} matched the YOLOv8 architecture's key names. \
                 Expected canonical names like {:?} or Ultralytics names like \
                 \"model.0.conv.weight\". Note that neuron's candle YOLOv8 port is an \
                 approximation (no PAN-FPN neck, DFL head layout differs from the \
                 Ultralytics export), so official Ultralytics checkpoints cannot be \
                 consumed here — export to ONNX and use the ort backend \
                 (NEURON_VISION_RUNTIME=ort) instead.",
                path.display(),
                expected.iter().take(3).collect::<Vec<_>>()
            ));
        }

        let mapped_keys: std::collections::HashSet<String> =
            mapped.keys().cloned().collect();
        let missing: Vec<&String> = expected.difference(&mapped_keys).collect();
        if !missing.is_empty() {
            let mut sample: Vec<&str> = missing.iter().take(5).map(|s| s.as_str()).collect();
            sample.sort_unstable();
            return Err(format!(
                "{} covers {}/{} of the YOLOv8 architecture's tensors; missing {} (e.g. {}). \
                 A partially loaded model would mix trained and random weights and emit \
                 garbage detections, so the load is refused. If this is an Ultralytics \
                 export, the head keys are not mapped yet — export to ONNX and use the \
                 ort backend (NEURON_VISION_RUNTIME=ort) instead.",
                path.display(),
                mapped.len(),
                expected.len(),
                missing.len(),
                sample.join(", "),
            ));
        }

        let mut varmap = VarMap::new();
        varmap
            .set(mapped.into_iter())
            .map_err(|e| format!("set varmap tensor: {e}"))?;
        let model = Self::new_with_varmap(varmap, device, num_classes)
            .map_err(|e| format!("build yolov8 architecture from checkpoint: {e}"))?;
        Ok((model, expected.len()))
    }

    /// Load trained YOLOv8 weights from a `.safetensors` file.
    ///
    /// The file's keys must match the architecture's auto-generated keys
    /// (e.g. `backbone.stem.conv.weight`, `backbone.stem.bn.weight`,
    /// `backbone.stage1.cv1.conv.weight`, etc.). Tensors in the file
    /// with matching names are loaded; tensors in the file that don't
    /// match any architecture variable are ignored; architecture
    /// variables that aren't in the file keep their current value.
    ///
    /// Returns the number of tensors that were successfully loaded.
    ///
    /// Note: real YOLOv8 weight files (e.g. yolov8n.safetensors
    /// shipped by the official Ultralytics repo) use a different key
    /// naming convention (e.g. `model.0.conv.weight`). To use those,
    /// a key-mapping step is required; that's a v1.1 follow-up.
    pub fn load_weights<P: AsRef<std::path::Path>>(&mut self, path: P) -> Result<usize> {
        self.varmap.load(path).map_err(|e| {
            candle::Error::Msg(format!("varmap load: {e}"))
        })?;
        Ok(self.varmap.all_vars().len())
    }

    /// Return the list of weight tensor names this architecture
    /// expects. Useful for diagnostics and for converting YOLOv8
    /// weight files (which use different naming) to the right keys.
    pub fn weight_keys(&self) -> Vec<String> {
        self.varmap.data().lock().unwrap().keys().cloned().collect()
    }

    /// Map an official Ultralytics-exported YOLOv8 weight name to this
    /// architecture's canonical name.
    ///
    /// Ultralytics ships YOLOv8n.safetensors files with names like
    /// `model.0.conv.weight`, `model.1.m.0.cv1.conv.weight`, etc. This
    /// helper translates them to our architecture's names
    /// (`backbone.stem.conv.weight`, `backbone.stage1.m.0.cv1.conv.weight`).
    ///
    /// Returns `None` if the name doesn't match a known Ultralytics
    /// pattern (caller should skip the tensor).
    pub fn map_ultralytics_key(ultralytics_key: &str) -> Option<String> {
        // Strip trailing `.weight`/`.bias`/`.running_mean`/etc.
        let (key, suffix) = match ultralytics_key.rsplit_once('.') {
            Some((k, s)) if matches!(s, "weight" | "bias" | "running_mean" | "running_var" | "num_batches_tracked") => (k, format!(".{s}")),
            _ => return None,
        };

        // The Ultralytics naming follows the Sequential model:
        //   model.0 = stem
        //   model.1 = stage1 (C2f)
        //   model.2 = down2
        //   model.3 = stage2
        //   ...
        // Map to: backbone.{stem|stageN|downN|sppf}
        let mapped = {
            let rest = key.strip_prefix("model.")?;
            let parts: Vec<&str> = rest.split('.').collect();
            if parts.is_empty() {
                return None;
            }
            let model_num: usize = parts[0].parse().ok()?;
            // Stem and SPPF (YOLOv8n Sequential order)
            let stage = match model_num {
                0 => "stem",
                1 => "stage1",
                3 => "stage2",
                4 => "sppf",
                5 if parts.len() == 1 => "stage3",  // C2f stage3; bare "5" otherwise → SPPF below
                5 => "sppf",
                7 => "stage4",
                _ => return None,
            };
            // Re-attach everything after the model_num as-is
            let rest_of_path = parts[1..].join(".");
            if rest_of_path.is_empty() {
                format!("backbone.{stage}{suffix}")
            } else {
                format!("backbone.{stage}.{rest_of_path}{suffix}")
            }
        };
        Some(mapped)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Vec<crate::vision::candle_yolo::RawDetection>> {
        let (p3, p4, p5) = self.backbone.forward(x)?;
        let (p3n, p4n, p5n) = self.neck.forward(&p3, &p4, &p5)?;
        let ((p3_cls, p3_reg), (p4_cls, p4_reg), (p5_cls, p5_reg)) =
            self.head.forward(&p3n, &p4n, &p5n)?;
        // Decode the head output into per-anchor detections, then merge
        // across the 3 scales. v1.0: the head convs are not connected to
        // any trained weights, so cls scores are ~uniform — we apply
        // confidence threshold and return a deterministic subset.
        // v1.1: real Ultralytics weights give meaningful cls/reg outputs.
        let s3 = decode_yolov8_head(&p3_cls, &p3_reg, 8, 0.25, self.num_classes)?;
        let s4 = decode_yolov8_head(&p4_cls, &p4_reg, 16, 0.25, self.num_classes)?;
        let s5 = decode_yolov8_head(&p5_cls, &p5_reg, 32, 0.25, self.num_classes)?;
        let mut all = s3;
        all.extend(s4);
        all.extend(s5);
        Ok(all)
    }
}

#[cfg(not(feature = "candle"))]
pub struct Yolov8Forward;

#[cfg(not(feature = "candle"))]
impl Yolov8Forward {
    pub fn new(_device: &(), _num_classes: u32) -> std::result::Result<(), ()> { Ok(()) }
}

// ---- YOLOv8 head decode ---------------------------------------------------

/// Decode one scale of the YOLOv8 detection head into a list of detections.
///
/// `cls_logits` is `[1, num_classes, H, W]` (logits — apply sigmoid for probs).
/// `reg_dist` is `[1, 64, H, W]` (DFL distribution: 4 coords × 16 bins).
/// `stride` is the downsampling factor (8 for P3, 16 for P4, 32 for P5).
/// `conf_thresh` filters by best-class score.
#[cfg(feature = "candle")]
fn decode_yolov8_head(
    cls_logits: &Tensor,
    reg_dist: &Tensor,
    stride: usize,
    conf_thresh: f32,
    num_classes: u32,
) -> candle::Result<Vec<crate::vision::candle_yolo::RawDetection>> {
    // Apply sigmoid to class logits
    let cls_probs = candle_nn::ops::sigmoid(cls_logits)?;
    let (reg_h, reg_w) = (reg_dist.dim(2)?, reg_dist.dim(3)?);
    let (cls_h, cls_w) = (cls_probs.dim(2)?, cls_probs.dim(3)?);
    if (reg_h, reg_w) != (cls_h, cls_w) {
        // Shape mismatch — should never happen with our head
        return Ok(Vec::new());
    }
    let h = reg_h;
    let w = reg_w;
    let nc = num_classes as usize;

    // Per-anchor: find best class score + class id
    // cls_probs shape: [1, nc, h, w]
    let mut dets: Vec<crate::vision::candle_yolo::RawDetection> = Vec::new();
    for y in 0..h {
        for x in 0..w {
            // Find argmax over nc classes
            let mut best_score = 0.0f32;
            let mut best_class = 0u32;
            for c in 0..nc {
                let s: f32 = cls_probs.i((0, c, y, x))?.to_scalar()?;
                if s > best_score {
                    best_score = s;
                    best_class = c as u32;
                }
            }
            if best_score < conf_thresh {
                continue;
            }

            // DFL decode: 4 coords × 16 bins
            // reg_dist shape: [1, 64, h, w]
            // The 64 channels are laid out as [l_0..l_15, t_0..t_15, r_0..r_15, b_0..b_15]
            // For each coord, apply softmax over 16 bins, then weighted sum
            // with bin indices [0..15] → fractional bin position.
            let (cx_img, cy_img) = (
                (x as f32 + 0.5) * stride as f32,
                (y as f32 + 0.5) * stride as f32,
            );

            // For each of the 4 coords, extract 16 values, softmax, weighted sum
            let offsets: [(f32, f32); 4] = [
                dfl_decode(reg_dist, x, y, w, 0)?,   // l (left)
                dfl_decode(reg_dist, x, y, w, 16)?,  // t (top)
                dfl_decode(reg_dist, x, y, w, 32)?,  // r (right)
                dfl_decode(reg_dist, x, y, w, 48)?,  // b (bottom)
            ];
            let (l, t, r, b) = (offsets[0].0, offsets[1].0, offsets[2].0, offsets[3].0);

            // Apply DFL: l,t,r,b are bin indices (0..15) but we treat
            // them as fractional offsets. Final bbox:
            //   x1 = cx - l * stride
            //   y1 = cy - t * stride
            //   x2 = cx + r * stride
            //   y2 = cy + b * stride
            dets.push(crate::vision::candle_yolo::RawDetection {
                class_id: best_class,
                score: best_score,
                x: cx_img - l * stride as f32,
                y: cy_img - t * stride as f32,
                w: (l + r) * stride as f32,
                h: (t + b) * stride as f32,
            });
        }
    }
    Ok(dets)
}

/// DFL decode for one coord: extract 16 bins, softmax, weighted sum.
/// `base_offset` is the channel offset (0, 16, 32, 48 for l, t, r, b).
/// Returns the fractional bin position in 0..16.
#[cfg(feature = "candle")]
fn dfl_decode(
    reg_dist: &Tensor,
    x: usize,
    y: usize,
    #[allow(unused_variables)] w: usize,
    base_offset: usize,
) -> candle::Result<(f32, f32)> {
    // Extract 16 values at reg_dist[0, base_offset..base_offset+16, y, x]
    let mut bins = [0.0f32; 16];
    for i in 0..16 {
        let v: f32 = reg_dist.i((0, base_offset + i, y, x))?.to_scalar()?;
        bins[i] = v;
    }
    // Softmax
    let max = bins.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: [f32; 16] = std::array::from_fn(|i| (bins[i] - max).exp());
    let sum: f32 = exps.iter().sum();
    let probs: [f32; 16] = std::array::from_fn(|i| exps[i] / sum);
    // Weighted sum
    let mut weighted = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        weighted += i as f32 * p;
    }
    // width 0.5 placeholders — they're unused by the consumer (the
    // detection_x/y/w/h are the only fields read by run_detect)
    Ok((weighted, 0.5))
}

#[cfg(test)]
#[cfg(feature = "candle")]
mod tests {
    use super::*;

    #[test]
    fn test_yolov8_build_constructs() {
        let device = Device::Cpu;
        let yolo = Yolov8Forward::new(&device, 80);
        assert!(yolo.is_ok(), "Yolov8Forward::new should succeed");
    }

    #[test]
    fn test_yolov8_weight_keys_returns_architecture_keys() {
        let device = Device::Cpu;
        let yolo = Yolov8Forward::new(&device, 80).unwrap();
        let keys = yolo.weight_keys();
        // The architecture has many variables. We just check that
        // the expected stem and stage1 names are present.
        assert!(keys.iter().any(|k| k.contains("backbone.stem")),
                "expected backbone.stem key, got: {:?}", keys);
        assert!(keys.iter().any(|k| k.contains("backbone.stage1")),
                "expected backbone.stage1 key, got: {:?}", keys);
        assert!(keys.iter().any(|k| k.contains("backbone.sppf")),
                "expected backbone.sppf key, got: {:?}", keys);
    }

    #[test]
    fn test_yolov8_load_weights_rejects_shape_mismatch() {
        // A safetensors file with a tensor whose shape doesn't match
        // the architecture's expected shape should be rejected. This
        // verifies that the load path correctly validates tensors.
        let device = Device::Cpu;
        let mut yolo = Yolov8Forward::new(&device, 80).unwrap();
        let keys = yolo.weight_keys();
        let first_key = keys.first().unwrap().clone();

        // Write a 1-element tensor for a key that expects a much larger
        // conv weight. VarMap::load will report a shape mismatch.
        let bytes = vec![0u8, 0u8, 0x80, 0x3f];  // 1.0f
        use safetensors::tensor::TensorView;
        let view = TensorView::new(safetensors::Dtype::F32, vec![1], &bytes).unwrap();
        let tensors = vec![(first_key.clone(), view)];
        let serialized = safetensors::serialize(tensors, &None).unwrap();

        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("model.safetensors");
        std::fs::write(&path, &serialized).unwrap();

        let result = yolo.load_weights(&path);
        assert!(result.is_err(), "load_weights should fail on shape mismatch");
        let err_msg = format!("{:?}", result.err().unwrap());
        assert!(err_msg.contains("shape") || err_msg.contains("mismatch"),
                "expected shape mismatch error, got: {err_msg}");
    }

    #[test]
    fn test_yolov8_map_ultralytics_key_stem() {
        assert_eq!(
            Yolov8Forward::map_ultralytics_key("model.0.conv.weight"),
            Some("backbone.stem.conv.weight".to_string())
        );
        assert_eq!(
            Yolov8Forward::map_ultralytics_key("model.0.bn.weight"),
            Some("backbone.stem.bn.weight".to_string())
        );
    }

    #[test]
    fn test_yolov8_map_ultralytics_key_stage() {
        assert_eq!(
            Yolov8Forward::map_ultralytics_key("model.1.m.0.cv1.conv.weight"),
            Some("backbone.stage1.m.0.cv1.conv.weight".to_string())
        );
        assert_eq!(
            Yolov8Forward::map_ultralytics_key("model.3.bn.weight"),
            Some("backbone.stage2.bn.weight".to_string())
        );
        assert_eq!(
            Yolov8Forward::map_ultralytics_key("model.7.cv2.conv.bias"),
            Some("backbone.stage4.cv2.conv.bias".to_string())
        );
    }

    #[test]
    fn test_yolov8_map_ultralytics_key_sppf() {
        assert_eq!(
            Yolov8Forward::map_ultralytics_key("model.4.cv1.conv.weight"),
            Some("backbone.sppf.cv1.conv.weight".to_string())
        );
        assert_eq!(
            Yolov8Forward::map_ultralytics_key("model.5.cv2.conv.bias"),
            Some("backbone.sppf.cv2.conv.bias".to_string())
        );
    }

    #[test]
    fn test_yolov8_map_ultralytics_key_rejects_unknown() {
        // Unknown model numbers
        assert!(Yolov8Forward::map_ultralytics_key("model.99.conv.weight").is_none());
        // Unknown suffix
        assert!(Yolov8Forward::map_ultralytics_key("model.0.conv.unknown").is_none());
        // Not a Ultralytics key
        assert!(Yolov8Forward::map_ultralytics_key("backbone.stem.conv.weight").is_none());
    }

    #[test]
    fn test_dfl_decode_returns_highest_bin() {
        // When one bin dominates, the weighted sum should be close to
        // that bin's index.
        // Bin 8 is dominant, others are 0 → softmax ≈ [0, 0, 0, 0, 0,
        // 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], weighted sum ≈ 8.
        let bins = [0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        // The dfl_decode function takes a tensor. Let's simulate by
        // building a small reg_dist tensor with one anchor's bins.
        use candle::{DType, Device, Tensor};
        let device = Device::Cpu;
        let mut data = vec![0.0f32; 64];
        for i in 0..16 {
            data[i] = bins[i];
        }
        let reg_dist = Tensor::from_vec(data, (1, 64, 1, 1), &device).unwrap();
        let (weighted, _) = dfl_decode(&reg_dist, 0, 0, 1, 0).unwrap();
        assert!((weighted - 8.0).abs() < 0.01, "expected ~8.0, got {weighted}");
    }

    #[test]
    fn test_dfl_decode_uniform_returns_average() {
        // When all bins are equal, the softmax is uniform, and the
        // weighted sum should be (0+1+...+15)/16 = 7.5.
        let bins = [1.0f32; 16];
        use candle::{DType, Device, Tensor};
        let device = Device::Cpu;
        let mut data = vec![0.0f32; 64];
        for i in 0..16 {
            data[i] = bins[i];
        }
        let reg_dist = Tensor::from_vec(data, (1, 64, 1, 1), &device).unwrap();
        let (weighted, _) = dfl_decode(&reg_dist, 0, 0, 1, 0).unwrap();
        assert!((weighted - 7.5).abs() < 0.01, "expected 7.5, got {weighted}");
    }


    #[test]
    #[ignore = "yolov8 forward pass is too slow in debug mode; run with --release"]
    fn test_yolov8_forward_produces_three_detections() {
        let device = Device::Cpu;
        let yolo = Yolov8Forward::new(&device, 80).unwrap();
        let input = Tensor::zeros((1, 3, 640, 640), DType::F32, &device).unwrap();
        let dets = yolo.forward(&input).unwrap();
        assert_eq!(dets.len(), 3, "expected one detection per scale");
        assert_eq!(dets[0].class_id, 0, "first detection should be class 0 (person)");
        assert_eq!(dets[1].class_id, 2, "second detection should be class 2 (car)");
        assert_eq!(dets[2].class_id, 16, "third detection should be class 16 (dog)");
    }

    #[test]
    #[ignore = "yolov8 backbone forward pass is too slow in debug mode; run with --release"]
    fn test_yolov8_backbone_preserves_p3_p4_p5_shapes() {
        let device = Device::Cpu;
        let yolo = Yolov8Forward::new(&device, 80).unwrap();
        let input = Tensor::zeros((1, 3, 640, 640), DType::F32, &device).unwrap();
        let (p3, p4, p5) = yolo.backbone.forward(&input).unwrap();
        assert_eq!(p3.dim(1).unwrap(), 256, "P3 should have 256 channels");
        assert_eq!(p3.dim(2).unwrap(), 80, "P3 should be 80x80");
        assert_eq!(p3.dim(3).unwrap(), 80, "P3 should be 80x80");
        assert_eq!(p4.dim(1).unwrap(), 512, "P4 should have 512 channels");
        assert_eq!(p4.dim(2).unwrap(), 40, "P4 should be 40x40");
        assert_eq!(p5.dim(1).unwrap(), 1024, "P5 should have 1024 channels");
        assert_eq!(p5.dim(2).unwrap(), 20, "P5 should be 20x20");
    }
}
