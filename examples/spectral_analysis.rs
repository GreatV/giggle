//! Spectral Analysis Example
//!
//! This example demonstrates spectral analysis techniques using STFT.

use giggle::{io, spectrum};
use log::info;
use ndarray::Array2;

fn main() {
    env_logger::init();
    info!("Spectral Analysis Example");

    // Generate a chirp signal
    let sr = 22050;
    let duration = 2.0;
    info!("Generating chirp signal (100 Hz to 1000 Hz)...");
    let chirp = io::chirp(100.0, 1000.0, sr, duration);
    info!("Generated {} samples\n", chirp.len());

    // STFT Computation
    info!("STFT Computation");

    let n_fft = 2048;
    let hop_length = 512;

    // Create STFT config
    let config = spectrum::StftConfig {
        n_fft,
        hop_length,
        win_length: n_fft,
        center: true,
        window: giggle::window::hann(n_fft),
        pad_mode: spectrum::PadMode::Reflect,
    };

    // Compute STFT
    let stft_result = spectrum::stft(&chirp, &config).unwrap();
    info!("STFT shape: {:?}", stft_result.shape());
    info!("  - Frequency bins: {}", stft_result.shape()[0]);
    info!("  - Time frames: {}", stft_result.shape()[1]);

    // Compute magnitude spectrogram
    let mut mag_spec = Array2::zeros((stft_result.shape()[0], stft_result.shape()[1]));
    for ((i, j), val) in mag_spec.indexed_iter_mut() {
        *val = stft_result[[i, j]].norm();
    }

    // Inverse STFT
    info!("Inverse STFT");

    let reconstructed = spectrum::istft(&stft_result, &config, Some(chirp.len())).unwrap();
    info!("Reconstructed {} samples", reconstructed.len());

    // Calculate reconstruction error
    let min_len = chirp.len().min(reconstructed.len());
    let error: f32 = chirp
        .iter()
        .zip(reconstructed.iter())
        .take(min_len)
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / min_len as f32;
    info!("  - Mean absolute error: {:.6}", error);

    // Griffin-Lim Reconstruction
    info!("Griffin-Lim Phase Reconstruction");

    let griffin_lim_result =
        spectrum::griffinlim(&mag_spec, &config, 32, Some(chirp.len()), 0.0).unwrap();

    info!("Reconstructed {} samples", griffin_lim_result.len());

    let min_len_gl = chirp.len().min(griffin_lim_result.len());
    let error_gl: f32 = chirp
        .iter()
        .zip(griffin_lim_result.iter())
        .take(min_len_gl)
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / min_len_gl as f32;
    info!("  - Mean absolute error: {:.6}", error_gl);
}
