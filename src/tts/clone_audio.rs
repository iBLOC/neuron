//! Voice-clone reference-audio decoding for the TTS wire format.
//!
//! `SpeechRequest.reference_audio` accepts a base64 payload (optionally a
//! `data:` URI) or a server-local file path. Payloads decode through the
//! shared ASR audio stack (WAV always; MP3/FLAC/OGG behind `asr-audio`) to
//! mono 16 kHz f32 — exactly what the VoxCPM2 VAE encoder consumes.

use std::path::PathBuf;

/// 30 s cap on reference audio, matching upstream guidance for clone prompts.
const MAX_REF_SAMPLES: usize = 16000 * 30;

/// Decode a `reference_audio` field to mono 16 kHz f32 samples.
pub fn decode_reference_payload(audio: &str) -> Result<Vec<f32>, String> {
    let bytes: Vec<u8> = if let Some(rest) = audio.strip_prefix("data:") {
        decode_base64(rest.split(',').last().unwrap_or(rest))?
    } else if PathBuf::from(audio).exists() {
        std::fs::read(audio).map_err(|e| format!("failed to read reference_audio file: {e}"))?
    } else {
        decode_base64(audio)?
    };

    // Full multi-format decode (WAV/MP3/FLAC/OGG + resample) when the
    // `asr-audio` feature is on; otherwise the always-available manual WAV
    // decoder, which requires the payload to already be 16 kHz mono.
    #[cfg(feature = "asr-audio")]
    let (samples, _sr) = crate::asr::audio::decode_to_pcm16k_mono(&bytes)
        .map_err(|e| format!("failed to decode reference_audio: {e}"))?;
    #[cfg(not(feature = "asr-audio"))]
    let (samples, _sr) = {
        let (s, sr) = crate::asr::wav::decode_wav(&bytes)
            .map_err(|e| format!("failed to decode reference_audio WAV: {e}"))?;
        if sr != 16000 {
            return Err(format!(
                "reference_audio WAV is {sr} Hz; build with --features asr-audio \
                 to decode/resample other formats (or supply 16 kHz mono WAV)"
            ));
        }
        (s, sr)
    };

    if samples.is_empty() {
        return Err("reference_audio decoded to zero samples".to_string());
    }
    if samples.len() > MAX_REF_SAMPLES {
        return Err(format!(
            "reference_audio too long: {} samples (max {MAX_REF_SAMPLES})",
            samples.len()
        ));
    }
    Ok(samples)
}

fn decode_base64(s: &str) -> Result<Vec<u8>, String> {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD
        .decode(s.trim())
        .map_err(|e| format!("invalid base64 reference_audio: {e}"))
}

#[cfg(all(test, feature = "voxcpm"))]
mod tests {
    use super::*;

    #[test]
    fn test_decode_reference_payload_rejects_garbage() {
        assert!(decode_reference_payload("!!!not base64!!!").is_err());
    }

    #[test]
    fn test_decode_reference_payload_wav_base64() {
        // 16-bit mono PCM WAV, 8 samples at 16 kHz, built with hound.
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut cursor = std::io::Cursor::new(Vec::new());
        {
            let mut writer = hound::WavWriter::new(&mut cursor, spec).unwrap();
            for i in 0..8 {
                writer.write_sample((i * 1000) as i16).unwrap();
            }
            writer.finalize().unwrap();
        }
        use base64::Engine as _;
        let b64 = base64::engine::general_purpose::STANDARD.encode(cursor.into_inner());
        let samples = decode_reference_payload(&b64).expect("decodes");
        assert_eq!(samples.len(), 8);
        assert!((samples[4] - (4000.0 / 32768.0)).abs() < 1e-4);
    }
}
