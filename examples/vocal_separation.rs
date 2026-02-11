//! Vocal Separation Example
//!
//! This example demonstrates a simple technique for separating vocals (and
//! other sporadic foreground signals) from accompanying instrumentation.
//!
//! Based on the "REPET-SIM" method of Rafii and Pardo, 2012, with modifications
//! from Fitzgerald, 2012 for soft-masking.

use giggle::io;
use giggle::spectrum::{StftConfig, istft, stft};
use giggle::utils::softmask;
use log::info;
use ndarray::Array2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    info!("Vocal Separation Example");

    let sr = 22050;

    // Generate synthetic mixture
    info!("Generating Synthetic Mixture");

    let duration = 4.0;

    // Background: harmonic instrumentation (chord progression)
    let c_major = create_chord(&[261.63, 329.63, 392.00], duration, sr); // C4, E4, G4
    let f_major = create_chord(&[349.23, 440.00, 523.25], duration, sr); // F4, A4, C5
    let g_major = create_chord(&[392.00, 493.88, 587.33], duration, sr); // G4, B4, D5

    // Create repeating chord pattern
    let background = c_major
        .iter()
        .zip(f_major.iter())
        .zip(g_major.iter())
        .map(|((c, f), g)| (c + f + g) / 3.0 * 0.5)
        .collect::<Vec<f32>>();

    // Foreground: vocal-like melody (simpler, more sparse)
    let melody_notes = [
        (523.25, 0.0, 0.5), // C5 at 0s
        (587.33, 0.5, 0.5), // D5 at 0.5s
        (523.25, 1.0, 0.5), // C5 at 1s
        (659.25, 1.5, 0.5), // E5 at 1.5s
        (587.33, 2.0, 0.5), // D5 at 2s
        (523.25, 2.5, 0.5), // C5 at 2.5s
        (493.88, 3.0, 0.5), // B4 at 3s
        (523.25, 3.5, 0.5), // C5 at 3.5s
    ];

    let mut foreground = vec![0.0f32; (duration * sr as f32) as usize];
    for (freq, start_time, note_duration) in &melody_notes {
        let start_sample = (*start_time * sr as f32) as usize;
        let note = io::tone(*freq, sr, *note_duration);
        for (i, sample) in note.iter().enumerate() {
            if start_sample + i < foreground.len() {
                foreground[start_sample + i] += sample * 0.8;
            }
        }
    }

    // Mix them
    let mixture: Vec<f32> = background
        .iter()
        .zip(foreground.iter())
        .map(|(b, f)| b + f)
        .collect();

    info!("Generated {} samples", mixture.len());

    // Compute STFT
    info!("Computing STFT");

    let cfg = StftConfig::default();
    let stft_result = stft(&mixture, &cfg)?;

    let n_freq = stft_result.shape()[0];
    let n_frames = stft_result.shape()[1];

    info!("STFT shape: [{}, {}]", n_freq, n_frames);

    // Get magnitude and phase
    let mut mag = Array2::<f32>::zeros((n_freq, n_frames));
    let mut phase = Array2::<f32>::zeros((n_freq, n_frames));
    for f in 0..n_freq {
        for t in 0..n_frames {
            mag[(f, t)] = stft_result[(f, t)].norm();
            phase[(f, t)] = stft_result[(f, t)].arg();
        }
    }

    // Non-local Filtering (REPET-SIM inspired)
    info!("Non-local Filtering");

    // Simple median filtering along time axis to get background estimate
    // (In a full implementation, this would use recurrence-based similarity)
    let mut background_mag = Array2::<f32>::zeros((n_freq, n_frames));
    let window_size = 5;

    for f in 0..n_freq {
        for t in 0..n_frames {
            let mut window = Vec::new();
            for dt in -(window_size as isize)..=window_size as isize {
                let tt = (t as isize + dt).clamp(0, n_frames as isize - 1) as usize;
                window.push(mag[(f, tt)]);
            }
            window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            background_mag[(f, t)] = window[window.len() / 2];
        }
    }

    // Take minimum of original and filtered (signals are additive)
    let background_mag = background_mag.mapv(|v| v.min(mag[(0, 0)]));

    info!("Background magnitude shape: {:?}", background_mag.shape());

    // Soft Masking
    info!("Soft Masking");

    let _margin_i = 2.0; // Instrument margin
    let _margin_v = 10.0; // Vocal margin
    let power = 2.0;

    // Create masks
    let mask_background = softmask(
        &background_mag,
        &[&(mag.mapv(|v| v - background_mag[(0, 0)]))],
        power,
        true,
    );

    let mask_foreground = softmask(
        &(mag.mapv(|v| v - background_mag[(0, 0)])),
        &[&background_mag],
        power,
        true,
    );

    info!("Background mask shape: {:?}", mask_background.shape());
    info!("Foreground mask shape: {:?}", mask_foreground.shape());

    // Apply masks
    let mut sep_background = Array2::<f32>::zeros((n_freq, n_frames));
    let mut sep_foreground = Array2::<f32>::zeros((n_freq, n_frames));

    for f in 0..n_freq {
        for t in 0..n_frames {
            sep_background[(f, t)] = mag[(f, t)] * mask_background[(f, t)];
            sep_foreground[(f, t)] = mag[(f, t)] * mask_foreground[(f, t)];
        }
    }

    // Reconstruct Audio
    info!("Reconstructing Audio");

    // Reconstruct background (instrumentation)
    let mut stft_background = stft_result.clone();
    for f in 0..n_freq {
        for t in 0..n_frames {
            let magnitude = sep_background[(f, t)];
            let phase_val = phase[(f, t)];
            stft_background[(f, t)] = num_complex::Complex32::from_polar(magnitude, phase_val);
        }
    }
    let audio_background = istft(&stft_background, &cfg, Some(mixture.len()))?;

    // Reconstruct foreground (vocals)
    let mut stft_foreground = stft_result.clone();
    for f in 0..n_freq {
        for t in 0..n_frames {
            let magnitude = sep_foreground[(f, t)];
            let phase_val = phase[(f, t)];
            stft_foreground[(f, t)] = num_complex::Complex32::from_polar(magnitude, phase_val);
        }
    }
    let audio_foreground = istft(&stft_foreground, &cfg, Some(mixture.len()))?;

    info!(
        "Reconstructed background: {} samples",
        audio_background.len()
    );
    info!(
        "Reconstructed foreground: {} samples",
        audio_foreground.len()
    );

    // Quality Analysis
    info!("Quality Analysis");

    // Compute energy ratios
    let mix_energy: f32 = mixture.iter().map(|s| s * s).sum();
    let bg_energy: f32 = audio_background.iter().map(|s| s * s).sum();
    let fg_energy: f32 = audio_foreground.iter().map(|s| s * s).sum();

    info!("Energy analysis:");
    info!("  Mixture:    {:.4}", mix_energy);
    info!(
        "  Background: {:.4} ({:.1}%)",
        bg_energy,
        bg_energy / mix_energy * 100.0
    );
    info!(
        "  Foreground: {:.4} ({:.1}%)",
        fg_energy,
        fg_energy / mix_energy * 100.0
    );
    info!(
        "  Total:      {:.4} ({:.1}%)",
        bg_energy + fg_energy,
        (bg_energy + fg_energy) / mix_energy * 100.0
    );

    // Compute separation quality (simplified)
    let bg_sparsity = compute_sparsity(&audio_background);
    let fg_sparsity = compute_sparsity(&audio_foreground);

    info!("\nSparsity (higher = more concentrated energy):");
    info!("  Background: {:.4}", bg_sparsity);
    info!("  Foreground: {:.4}", fg_sparsity);

    Ok(())
}

/// Create a chord from multiple frequencies
fn create_chord(frequencies: &[f32], duration: f32, sr: u32) -> Vec<f32> {
    let mut result = vec![0.0f32; (duration * sr as f32) as usize];

    for freq in frequencies {
        let tone = io::tone(*freq, sr, duration);
        for (i, sample) in tone.iter().enumerate() {
            result[i] += sample / frequencies.len() as f32;
        }
    }

    result
}

/// Compute sparsity measure (Gini coefficient approximation)
fn compute_sparsity(signal: &[f32]) -> f32 {
    let energy: Vec<f32> = signal.iter().map(|s| s * s).collect();
    let total: f32 = energy.iter().sum();

    if total < 1e-10 {
        return 0.0;
    }

    let mean = total / energy.len() as f32;
    let variance = energy.iter().map(|e| (e - mean).powi(2)).sum::<f32>() / energy.len() as f32;

    // Coefficient of variation as sparsity proxy
    variance.sqrt() / mean
}
