//! Minimal RIFF/WAVE decoder shared by the ASR runtimes.
//!
//! Accepts 16-bit PCM at 16 kHz, mono or stereo (stereo downmixed by
//! averaging), and hands back f32 samples in [-1, 1]. This is the same
//! contract the Whisper runtime has always shipped. Multi-format
//! decoding (MP3 / FLAC / OGG/Vorbis) lives in [`super::audio`] behind
//! the `asr-audio` feature.

pub fn decode_wav(data: &[u8]) -> Result<(Vec<f32>, u32), String> {
    if data.len() < 44 || &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        return Err("not a valid WAV file".to_string());
    }
    let mut pos = 12;
    let mut sample_rate = 0u32;
    let mut num_channels = 0u16;
    let mut bits_per_sample = 0u16;
    let mut audio_data_pos = 0;
    let mut audio_data_len = 0;
    while pos + 8 <= data.len() {
        let chunk_id = &data[pos..pos + 4];
        let chunk_size = u32::from_le_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]]);
        if chunk_id == b"fmt " {
            num_channels = u16::from_le_bytes([data[pos + 10], data[pos + 11]]);
            sample_rate = u32::from_le_bytes([data[pos + 12], data[pos + 13], data[pos + 14], data[pos + 15]]);
            bits_per_sample = u16::from_le_bytes([data[pos + 22], data[pos + 23]]);
        } else if chunk_id == b"data" {
            audio_data_pos = pos + 8;
            audio_data_len = chunk_size as usize;
            break;
        }
        pos += 8 + chunk_size as usize;
    }
    if audio_data_pos == 0 {
        return Err("no audio data chunk found".to_string());
    }
    if sample_rate != 16000 {
        return Err(format!("unsupported sample rate {} (expected 16000)", sample_rate));
    }
    if bits_per_sample != 16 {
        return Err(format!("unsupported bits per sample {} (only 16-bit supported)", bits_per_sample));
    }
    let mut samples = Vec::with_capacity(audio_data_len / 2);
    for i in (0..audio_data_len).step_by(2) {
        let s = i16::from_le_bytes([data[audio_data_pos + i], data[audio_data_pos + i + 1]]);
        samples.push(s as f32 / 32768.0);
    }
    if num_channels == 2 {
        let mut mono = Vec::with_capacity(samples.len() / 2);
        for i in (0..samples.len()).step_by(2) {
            mono.push((samples[i] + samples[i + 1]) / 2.0);
        }
        samples = mono;
    }
    Ok((samples, sample_rate))
}
