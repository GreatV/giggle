//! Harmonic Analysis Example
//!
//! This example demonstrates how to extract the harmonic spectrum from an audio signal.
//! The basic idea is to estimate the fundamental frequency (f0) at each time step,
//! and extract the energy at integer multiples of f0 (the harmonics).
//!
//! Based on librosa's plot_spectral_harmonics.py example.

use giggle::convert::fft_frequencies;
use giggle::harmonic::{self, InterpKind};
use giggle::io;
use giggle::spectrum::{StftConfig, stft};
use log::info;
use ndarray::Array2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    info!("Harmonic Analysis Example");

    let sr = 22050;

    // Generate a test signal with harmonics
    info!("Generating Test Signal");

    // Create a signal with fundamental and overtones
    let duration = 2.0;
    let f0 = 440.0; // A4

    info!("Fundamental frequency: {} Hz", f0);

    // Generate fundamental + harmonics
    let fundamental = io::tone(f0, sr, duration);
    let h2 = io::tone(f0 * 2.0, sr, duration);
    let h3 = io::tone(f0 * 3.0, sr, duration);
    let h4 = io::tone(f0 * 4.0, sr, duration);

    // Mix with decreasing amplitudes
    let signal: Vec<f32> = fundamental
        .iter()
        .zip(h2.iter())
        .zip(h3.iter())
        .zip(h4.iter())
        .map(|(((f, h2), h3), h4)| f * 1.0 + h2 * 0.5 + h3 * 0.33 + h4 * 0.25)
        .collect();

    info!("Generated {} samples", signal.len());
    info!("  - Fundamental: {} Hz (amplitude 1.0)", f0);
    info!("  - Harmonic 2: {} Hz (amplitude 0.5)", f0 * 2.0);
    info!("  - Harmonic 3: {} Hz (amplitude 0.33)", f0 * 3.0);
    info!("  - Harmonic 4: {} Hz (amplitude 0.25)", f0 * 4.0);

    // Compute STFT
    info!("Computing STFT");

    let cfg = StftConfig::default();
    let stft_result = stft(&signal, &cfg)?;

    let n_freq = stft_result.shape()[0];
    let n_frames = stft_result.shape()[1];

    info!("STFT shape: [{}, {}]", n_freq, n_frames);

    // Convert to magnitude spectrogram
    let mut mag = Array2::<f32>::zeros((n_freq, n_frames));
    for f in 0..n_freq {
        for t in 0..n_frames {
            mag[(f, t)] = stft_result[(f, t)].norm();
        }
    }

    // Harmonic Interpolation
    info!("Harmonic Interpolation");

    let freqs = fft_frequencies(sr, cfg.n_fft);
    let harmonics = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    info!("Computing energy at harmonics: {:?}", harmonics);

    let harm_energy = harmonic::interp_harmonics(&mag, &freqs, &harmonics, InterpKind::Linear, 0.0);

    info!("Harmonic energy shape: {:?}", harm_energy.shape());
    info!("  - {} harmonics", harm_energy.shape()[0]);
    info!("  - {} frequency bins", harm_energy.shape()[1]);
    info!("  - {} time frames", harm_energy.shape()[2]);

    // Show average energy per harmonic at the fundamental frequency
    let mid_frame = n_frames / 2;
    info!("\nEnergy at each harmonic (frame {}):", mid_frame);
    for (i, h) in harmonics.iter().enumerate() {
        // Find bin closest to fundamental
        let f0_bin = freqs
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| (*a - f0).abs().partial_cmp(&(*b - f0).abs()).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let energy = harm_energy[(i, f0_bin, mid_frame)];
        info!("  Harmonic {} ({} Hz): {:.4}", h, f0 * h, energy);
    }

    // Harmonic Salience
    info!("Harmonic Salience");

    let weights = vec![1.0, 0.5, 0.33, 0.25, 0.2];
    let salience = harmonic::salience(
        &mag,
        &freqs,
        &harmonics,
        Some(&weights),
        false,
        0.0,
        InterpKind::Linear,
    );

    info!("Salience shape: {:?}", salience.shape());

    // Find frequency with maximum salience
    let mut max_salience = 0.0f32;
    let mut max_freq_idx = 0;
    for f in 0..n_freq {
        let avg_sal: f32 = salience.row(f).iter().sum::<f32>() / n_frames as f32;
        if avg_sal > max_salience {
            max_salience = avg_sal;
            max_freq_idx = f;
        }
    }

    info!(
        "\nMaximum salience at {:.2} Hz (value: {:.4})",
        freqs[max_freq_idx], max_salience
    );

    // f0-based Harmonic Extraction
    info!("f0-based Harmonic Extraction");

    // Assume constant f0 of 440 Hz
    let f0_vec: Vec<f32> = vec![f0; n_frames];
    let f0_harmonics: Vec<f32> = (1..=12).map(|i| i as f32).collect();

    let f0_harm = harmonic::f0_harmonics(
        &mag,
        &f0_vec,
        &freqs,
        &f0_harmonics,
        InterpKind::Linear,
        0.0,
    )
    .unwrap();

    info!("f0-harmonic shape: {:?}", f0_harm.shape());
    info!("  - {} harmonics", f0_harm.shape()[0]);
    info!("  - {} time frames", f0_harm.shape()[1]);

    // Show energy at each harmonic
    info!(
        "\nEnergy at harmonics of f0={} Hz (frame {}):",
        f0, mid_frame
    );
    for i in 0..f0_harmonics.len().min(6) {
        let energy = f0_harm[(i, mid_frame)];
        let harmonic_freq = f0 * f0_harmonics[i];
        info!(
            "  H{} ({} Hz): {:.4}",
            f0_harmonics[i] as i32, harmonic_freq, energy
        );
    }

    // Timbre Analysis
    info!("Timbre Analysis");

    // Compute harmonic energy ratios (timbre feature)
    let h1_energy = f0_harm[(0, mid_frame)];
    if h1_energy > 1e-10 {
        info!("Harmonic energy ratios (relative to fundamental):");
        for i in 1..f0_harmonics.len().min(6) {
            let ratio = f0_harm[(i, mid_frame)] / h1_energy;
            info!("  H{}/H1: {:.4}", f0_harmonics[i] as i32, ratio);
        }
    }

    // Compute spectral centroid of harmonics
    let mut weighted_sum = 0.0f32;
    let mut total_energy = 0.0f32;
    for i in 0..f0_harmonics.len() {
        let energy = f0_harm[(i, mid_frame)];
        let freq = f0 * f0_harmonics[i];
        weighted_sum += energy * freq;
        total_energy += energy;
    }
    let centroid = if total_energy > 1e-10 {
        weighted_sum / total_energy
    } else {
        0.0
    };

    info!("\nHarmonic spectral centroid: {:.2} Hz", centroid);

    Ok(())
}
