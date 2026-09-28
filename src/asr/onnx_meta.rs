//! Minimal ONNX `metadata_props` reader -- protobuf wire-format walk, no
//! `onnx` crate.
//!
//! sherpa-onnx exported graphs carry their preprocessing contract (LFR
//! window/shift, CMVN statistics, language and ITN ids) in the model's
//! top-level `metadata_props` map. sherpa-onnx reads that map at runtime
//! through onnxruntime; tract discards it when it parses the graph. To
//! run the same graphs under [`super::sensevoice`] the map has to be
//! recovered from the file itself, and pulling in the full ONNX
//! protobuf toolchain just for that would be heavy.
//!
//! Wire layout used here (ONNX proto `ModelProto`):
//!
//! | field | type | meaning |
//! |---|---|---|
//! | `7` | message | `graph` (skipped, may be huge) |
//! | `14` | repeated message | `metadata_props` (`StringStringEntryProto`) |
//!
//! `StringStringEntryProto` carries `key` (field 1) and `value` (field 2),
//! both length-delimited strings. Every other field type -- including the
//! graph body -- is skipped by length, so this reads O(metadata) bytes
//! worth of interpretation while still streaming over the whole file.

use std::collections::HashMap;

/// The parsed `metadata_props` map of one ONNX model file.
#[derive(Debug, Clone, Default)]
pub struct OnnxMetadata {
    entries: HashMap<String, String>,
}

impl OnnxMetadata {
    /// Parse `metadata_props` out of an ONNX `ModelProto` blob.
    ///
    /// A file with no metadata parses to an empty map rather than an
    /// error: the contract (which keys are required) is the caller's,
    /// and models without the sherpa-onnx metadata block are simply not
    /// SenseVoice exports.
    pub fn parse(file: &[u8]) -> Result<Self, String> {
        let mut entries = HashMap::new();
        let mut walker = ProtoWalker { buf: file, pos: 0 };
        while let Some((field, wire)) = walker.next_field()? {
            if field == 14 && wire == WireType::Len {
                let entry = walker.read_len_payload()?;
                let mut kv = ProtoWalker { buf: entry, pos: 0 };
                let mut key: Option<&[u8]> = None;
                let mut value: Option<&[u8]> = None;
                while let Some((f, w)) = kv.next_field()? {
                    if w == WireType::Len {
                        match f {
                            1 => key = Some(kv.read_len_payload()?),
                            2 => value = Some(kv.read_len_payload()?),
                            _ => kv.skip_len_payload()?,
                        }
                    } else {
                        kv.skip_value(w)?;
                    }
                }
                if let (Some(k), Some(v)) = (key, value) {
                    entries.insert(
                        String::from_utf8_lossy(k).into_owned(),
                        String::from_utf8_lossy(v).into_owned(),
                    );
                }
            } else if wire == WireType::Len {
                walker.skip_len_payload()?;
            } else {
                walker.skip_value(wire)?;
            }
        }
        Ok(Self { entries })
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.entries.get(key).map(|s| s.as_str())
    }

    /// Parse a comma-separated f32 list (the `neg_mean` / `inv_stddev`
    /// layout the sherpa-onnx exporter emits).
    pub fn get_f32_vec(&self, key: &str) -> Option<Result<Vec<f32>, String>> {
        self.entries.get(key).map(|raw| {
            raw.split(',')
                .map(|s| {
                    s.trim()
                        .parse::<f32>()
                        .map_err(|e| format!("metadata {key}: bad float {s:?}: {e}"))
                })
                .collect()
        })
    }

    pub fn get_u64(&self, key: &str) -> Option<Result<u64, String>> {
        self.entries.get(key).map(|raw| {
            raw.trim()
                .parse::<u64>()
                .map_err(|e| format!("metadata {key}: bad integer {raw:?}: {e}"))
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WireType {
    Varint,
    Fixed64,
    Len,
    Fixed32,
}

struct ProtoWalker<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> ProtoWalker<'a> {
    fn next_field(&mut self) -> Result<Option<(u64, WireType)>, String> {
        if self.pos >= self.buf.len() {
            return Ok(None);
        }
        let key = self.read_varint()?;
        let field = key >> 3;
        let wire = match key & 0x7 {
            0 => WireType::Varint,
            1 => WireType::Fixed64,
            2 => WireType::Len,
            5 => WireType::Fixed32,
            w => return Err(format!("unsupported protobuf wire type {w}")),
        };
        if field == 0 {
            return Err("protobuf field number 0 is invalid".to_string());
        }
        Ok(Some((field, wire)))
    }

    fn read_varint(&mut self) -> Result<u64, String> {
        let mut value = 0u64;
        let mut shift = 0u32;
        loop {
            let Some(&b) = self.buf.get(self.pos) else {
                return Err("truncated varint".to_string());
            };
            self.pos += 1;
            value |= u64::from(b & 0x7f) << shift;
            if b & 0x80 == 0 {
                return Ok(value);
            }
            shift += 7;
            if shift >= 64 {
                return Err("varint longer than 10 bytes".to_string());
            }
        }
    }

    fn read_len_payload(&mut self) -> Result<&'a [u8], String> {
        let len = self.read_varint()? as usize;
        let end = self
            .pos
            .checked_add(len)
            .filter(|&e| e <= self.buf.len())
            .ok_or_else(|| format!("length-delimited field of {len} bytes overruns the buffer"))?;
        let payload = &self.buf[self.pos..end];
        self.pos = end;
        Ok(payload)
    }

    fn skip_len_payload(&mut self) -> Result<(), String> {
        self.read_len_payload().map(|_| ())
    }

    fn skip_value(&mut self, wire: WireType) -> Result<(), String> {
        match wire {
            WireType::Varint => self.read_varint().map(|_| ()),
            WireType::Fixed64 => self.skip_bytes(8),
            WireType::Fixed32 => self.skip_bytes(4),
            WireType::Len => self.skip_len_payload(),
        }
    }

    fn skip_bytes(&mut self, n: usize) -> Result<(), String> {
        self.pos = self
            .pos
            .checked_add(n)
            .filter(|&p| p <= self.buf.len())
            .ok_or_else(|| "fixed-width field overruns the buffer".to_string())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-encode a minimal ModelProto carrying two metadata entries plus
    /// a skipped graph field, so the walker is exercised without the onnx
    /// crate.
    fn encode_len(field: u64, payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut key = field << 3 | 2;
        while key >= 0x80 {
            out.push(0x80 | (key & 0x7f) as u8);
            key >>= 7;
        }
        out.push(key as u8);
        let mut len = payload.len() as u64;
        while len >= 0x80 {
            out.push(0x80 | (len & 0x7f) as u8);
            len >>= 7;
        }
        out.push(len as u8);
        out.extend_from_slice(payload);
        out
    }

    fn entry(key: &str, value: &str) -> Vec<u8> {
        let mut e = encode_len(1, key.as_bytes());
        e.extend(encode_len(2, value.as_bytes()));
        e
    }

    #[test]
    fn test_parse_metadata_entries() {
        let mut buf = Vec::new();
        // field 7 (graph) is skipped wholesale, including its inner fields
        buf.extend(encode_len(7, b"\x08\x96\x01some graph body"));
        buf.extend(encode_len(14, &entry("lfr_window_size", "7")));
        buf.extend(encode_len(14, &entry("model_type", "sense_voice_ctc")));
        // an unknown top-level varint field must be skipped too
        buf.extend_from_slice(&[0x78, 0x2a]); // field 15 varint 42

        let meta = OnnxMetadata::parse(&buf).unwrap();
        assert_eq!(meta.get("lfr_window_size"), Some("7"));
        assert_eq!(meta.get("model_type"), Some("sense_voice_ctc"));
        assert_eq!(meta.get("missing"), None);
    }

    #[test]
    fn test_parse_float_and_int_helpers() {
        let mut buf = Vec::new();
        buf.extend(encode_len(14, &entry("neg_mean", "-8.311879,-8.600912, 0.5")));
        buf.extend(encode_len(14, &entry("vocab_size", "25055")));

        let meta = OnnxMetadata::parse(&buf).unwrap();
        let mean = meta.get_f32_vec("neg_mean").unwrap().unwrap();
        assert_eq!(mean, vec![-8.311879, -8.600912, 0.5]);
        assert_eq!(meta.get_u64("vocab_size").unwrap().unwrap(), 25055);
        // a non-numeric entry surfaces as Err, not as a silent empty vec
        let mut bad = Vec::new();
        bad.extend(encode_len(14, &entry("neg_mean", "1.0,oops")));
        let meta = OnnxMetadata::parse(&bad).unwrap();
        assert!(meta.get_f32_vec("neg_mean").unwrap().is_err());
    }

    #[test]
    fn test_parse_empty_and_truncated() {
        assert!(OnnxMetadata::parse(&[]).unwrap().entries.is_empty());
        // length runs past the end -> error, not panic
        let buf = [0x72u8, 0xff, 0x01];
        assert!(OnnxMetadata::parse(&buf).is_err());
    }
}
