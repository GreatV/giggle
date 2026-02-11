//! Edge case tests for boundary conditions and unusual inputs.
//!
//! Tests cover:
//! - Empty audio
//! - Single sample
//! - Very long files (simulated)

use giggle::{effects, feature, spectrum, window};

// Empty Audio Tests

#[test]
fn stft_empty_input() {
    let y: Vec<f32> = Vec::new();
    let cfg = spectrum::StftConfig::default();
    assert!(spectrum::stft(&y, &cfg).is_err());
}

#[test]
fn istft_empty_matrix() {
    let stft = ndarray::Array2::<num_complex::Complex32>::zeros((0, 0));
    let cfg = spectrum::StftConfig::default();
    assert!(spectrum::istft(&stft, &cfg, None).is_err());
}

#[test]
fn mel_spectrogram_empty() {
    let y: Vec<f32> = Vec::new();
    assert!(feature::mel::melspectrogram(&y, 22050, 512, 128, 40).is_err());
}

#[test]
fn mfcc_empty() {
    let y: Vec<f32> = Vec::new();
    assert!(feature::mfcc::mfcc(&y, 22050, 13, 512, 128, 40).is_err());
}

#[test]
fn chroma_empty() {
    let y: Vec<f32> = Vec::new();
    assert!(feature::chroma::chroma_stft(&y, 22050, 512, 128, 12, 0.0).is_err());
}

#[test]
fn zcr_empty() {
    let y: Vec<f32> = Vec::new();
    // With center=false, empty input should produce no frames
    let zcr = feature::basic::zero_crossing_rate_frames(&y, 2048, 512, false).unwrap();
    assert!(zcr.is_empty());
}

#[test]
fn rms_empty() {
    let y: Vec<f32> = Vec::new();
    // With center=false, empty input should produce no frames
    let rms = feature::basic::rms_frames(&y, 2048, 512, false).unwrap();
    assert!(rms.is_empty());
}

#[test]
fn trim_empty() {
    let y: Vec<f32> = Vec::new();
    let (trimmed, _) = effects::trim::trim(&y, 60.0);
    assert!(trimmed.is_empty());
}

#[test]
fn split_empty() {
    let y: Vec<f32> = Vec::new();
    let intervals = effects::trim::split(&y, 60.0, 1);
    assert!(intervals.is_empty());
}

#[test]
fn remix_empty() {
    let y: Vec<f32> = Vec::new();
    let intervals = vec![(0, 10)];
    let remixed = effects::remix::remix(&y, intervals, false);
    assert!(remixed.is_empty());
}

// Single Sample Tests

#[test]
fn stft_single_sample() {
    let y = vec![1.0f32];
    let cfg = spectrum::StftConfig {
        n_fft: 4,
        hop_length: 2,
        win_length: 4,
        window: window::hann(4),
        ..Default::default()
    };
    let stft = spectrum::stft(&y, &cfg).unwrap();
    // Should still produce output (with padding)
    assert!(stft.shape()[0] == 3); // n_fft/2 + 1
}

#[test]
fn mel_spectrogram_single_sample() {
    let y = vec![1.0f32];
    let mel = feature::mel::melspectrogram(&y, 22050, 512, 128, 40).unwrap();
    // With centering, single sample should produce at least one frame
    assert!(mel.shape()[0] == 40);
}

#[test]
fn zcr_single_sample() {
    let y = vec![1.0f32];
    let zcr = feature::basic::zero_crossing_rate_frames(&y, 4, 2, true).unwrap();
    // May or may not produce frames depending on centering
    assert!(zcr.len() <= 2);
}

#[test]
fn trim_single_sample() {
    let y = vec![0.5f32];
    let (trimmed, (start, end)) = effects::trim::trim(&y, 60.0);
    assert_eq!(trimmed.len(), 1);
    assert_eq!(start, 0);
    assert_eq!(end, 1);
}

#[test]
fn remix_single_sample() {
    let y = vec![1.0f32];
    let intervals = vec![(0, 1)];
    let remixed = effects::remix::remix(&y, intervals, false);
    assert_eq!(remixed.len(), 1);
    assert_eq!(remixed[0], 1.0);
}

// Very Long File Tests (Simulated)
// Note: We use smaller sizes than actual 1-hour files to keep tests fast,
// but test the same code paths.

#[test]
fn stft_large_input() {
    // Simulate a large file: 10 seconds at 22050 Hz
    let sr = 22050;
    let duration_sec = 10;
    let n_samples = sr * duration_sec;

    // Generate a simple signal
    let y: Vec<f32> = (0..n_samples)
        .map(|i| ((i as f32) * 440.0 * 2.0 * std::f32::consts::PI / sr as f32).sin())
        .collect();

    let mut cfg = spectrum::StftConfig::default();
    cfg.n_fft = 2048;
    cfg.hop_length = 512;
    cfg.window = window::hann(cfg.n_fft);

    let stft = spectrum::stft(&y, &cfg).unwrap();

    // With centering, padded length = n_samples + n_fft
    // n_frames = (padded_len - n_fft) / hop_length + 1
    let padded_len = n_samples + cfg.n_fft;
    let expected_frames = (padded_len - cfg.n_fft) / cfg.hop_length + 1;
    let actual_frames = stft.shape()[1];
    assert!(
        actual_frames > 0,
        "STFT should produce frames for large input"
    );
    assert_eq!(actual_frames, expected_frames, "Frame count mismatch");
}

#[test]
fn mel_spectrogram_large_input() {
    // 5 seconds at 22050 Hz
    let sr = 22050;
    let n_samples = sr * 5;

    let y: Vec<f32> = (0..n_samples)
        .map(|i| ((i as f32) * 440.0 * 2.0 * std::f32::consts::PI / sr as f32).sin())
        .collect();

    let mel = feature::mel::melspectrogram(&y, sr as u32, 2048, 512, 128).unwrap();

    assert_eq!(mel.shape()[0], 128);
    assert!(mel.shape()[1] > 0);
}

#[test]
fn mfcc_large_input() {
    // 3 seconds at 22050 Hz
    let sr = 22050;
    let n_samples = sr * 3;

    let y: Vec<f32> = (0..n_samples)
        .map(|i| ((i as f32) * 440.0 * 2.0 * std::f32::consts::PI / sr as f32).sin())
        .collect();

    let mfcc = feature::mfcc::mfcc(&y, sr as u32, 13, 2048, 512, 128).unwrap();

    assert_eq!(mfcc.shape()[0], 13);
    assert!(mfcc.shape()[1] > 0);
}

#[test]
fn chroma_large_input() {
    // 2 seconds at 22050 Hz
    let sr = 22050;
    let n_samples = sr * 2;

    let y: Vec<f32> = (0..n_samples)
        .map(|i| ((i as f32) * 440.0 * 2.0 * std::f32::consts::PI / sr as f32).sin())
        .collect();

    let chroma = feature::chroma::chroma_stft(&y, sr as u32, 2048, 512, 12, 0.0).unwrap();

    assert_eq!(chroma.shape()[0], 12);
    assert!(chroma.shape()[1] > 0);
}

// Numerical Stability Tests

#[test]
fn stft_all_zeros() {
    let y = vec![0.0f32; 4096];
    let cfg = spectrum::StftConfig::default();
    let stft = spectrum::stft(&y, &cfg).unwrap();

    // All coefficients should be zero
    for v in stft.iter() {
        assert!(v.norm() < 1e-10);
    }
}

#[test]
fn mel_filterbank_zero_mels() {
    let fb = feature::mel::mel_filterbank(22050, 2048, 0, 0.0, 11025.0);
    assert_eq!(fb.shape()[0], 0);
}

#[test]
fn zcr_constant_signal() {
    // Constant positive signal should have zero crossings = 0
    let y = vec![1.0f32; 1000];
    let zcr = feature::basic::zero_crossing_rate_frames(&y, 256, 128, true).unwrap();
    for &v in &zcr {
        assert!(v < 1e-6, "ZCR should be zero for constant signal");
    }
}

#[test]
fn rms_silent_signal() {
    let y = vec![0.0f32; 1000];
    let rms = feature::basic::rms_frames(&y, 256, 128, true).unwrap();
    for &v in &rms {
        assert!(v < 1e-10, "RMS should be zero for silent signal");
    }
}

#[test]
fn trim_all_silence() {
    let y = vec![0.0f32; 1000];
    let (trimmed, _) = effects::trim::trim(&y, 60.0);
    assert!(trimmed.is_empty());
}

#[test]
fn split_all_silence() {
    let y = vec![0.0f32; 1000];
    let intervals = effects::trim::split(&y, 60.0, 1);
    assert!(intervals.is_empty());
}
