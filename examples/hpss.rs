//! Harmonic-Percussive Source Separation Example
//!
//! This example demonstrates separating an audio signal into its harmonic
//! and percussive components using median filtering (HPSS).
//!
//! Based on the approach of Fitzgerald, 2010 and its margin-based extension
//! by Dreidger, Mueller and Disch, 2014.

use giggle::{effects, io, spectrum};
use log::info;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    info!("Harmonic-Percussive Source Separation");

    // Generate a signal with both harmonic and percussive components
    info!("Generating synthetic signal...");
    let sr = 22050;
    let duration = 4.0;

    // Create harmonic component: a musical chord
    let c4 = io::tone(261.63, sr, duration);
    let e4 = io::tone(329.63, sr, duration);
    let g4 = io::tone(392.00, sr, duration);

    let harmonic_signal: Vec<f32> = c4
        .iter()
        .zip(e4.iter())
        .zip(g4.iter())
        .map(|((c, e), g)| (c + e + g) / 3.0 * 0.5)
        .collect();

    // Create percussive component: clicks at regular intervals
    let click_times: Vec<f32> = (0..16).map(|i| i as f32 * 0.25).collect();
    let percussive_signal = io::clicks(
        &click_times,
        sr,
        Some((duration * sr as f32) as usize),
        0.02,
        800.0,
    );

    // Mix the components
    let mixed_signal: Vec<f32> = harmonic_signal
        .iter()
        .zip(percussive_signal.iter())
        .map(|(h, p)| h + p * 0.3)
        .collect();

    info!("Generated {} samples", mixed_signal.len());

    let n_fft = 2048;
    let hop_length = 512;

    // Compute STFT
    info!("Computing STFT");

    let cfg = spectrum::StftConfig {
        n_fft,
        hop_length,
        ..Default::default()
    };

    let stft = spectrum::stft(&mixed_signal, &cfg)?;
    info!("STFT shape: {:?}", stft.shape());
    info!("  - Frequency bins: {}", stft.shape()[0]);
    info!("  - Time frames: {}", stft.shape()[1]);

    // HPSS Separation
    info!("HPSS Separation");

    // Separate with default parameters
    let (harmonic_stft, percussive_stft) = effects::hpss::hpss(
        &stft,
        (31, 17), // kernel sizes: (harmonic, percussive)
        2.0,      // power for Wiener-like masking
        1.0,      // margin in dB
    );

    info!("  - Harmonic STFT shape: {:?}", harmonic_stft.shape());
    info!("  - Percussive STFT shape: {:?}", percussive_stft.shape());

    // Reconstruct Time-Domain Signals
    info!("Reconstructing Time-Domain Signals");

    let harmonic_reconstructed = spectrum::istft(&harmonic_stft, &cfg, Some(mixed_signal.len()))?;
    let percussive_reconstructed =
        spectrum::istft(&percussive_stft, &cfg, Some(mixed_signal.len()))?;

    info!("Reconstructed signal lengths:");
    info!("  - Harmonic: {} samples", harmonic_reconstructed.len());
    info!("  - Percussive: {} samples", percussive_reconstructed.len());

    // Compare with Different Margins
    info!("Effect of Different Margin Values");

    let margins = [1.0, 2.0, 4.0, 8.0];

    for margin in &margins {
        let (h, p) = effects::hpss::hpss(&stft, (31, 17), 2.0, *margin);

        // Calculate energy distribution
        let h_energy: f32 = h.iter().map(|c| c.norm_sqr()).sum();
        let p_energy: f32 = p.iter().map(|c| c.norm_sqr()).sum();
        let total_energy = h_energy + p_energy;

        let h_ratio = if total_energy > 0.0 {
            h_energy / total_energy
        } else {
            0.0
        };
        let p_ratio = if total_energy > 0.0 {
            p_energy / total_energy
        } else {
            0.0
        };

        info!(
            "  Margin {:4.1} dB: Harmonic={:.1}%, Percussive={:.1}%",
            margin,
            h_ratio * 100.0,
            p_ratio * 100.0
        );
    }

    // Direct Harmonic/ Percussive Extraction
    info!("Direct Component Extraction");

    // Using the convenience functions
    let h_direct = effects::hpss::harmonic(&stft, 31);
    let p_direct = effects::hpss::percussive(&stft, 17);

    let h_direct_energy: f32 = h_direct.iter().map(|c| c.norm_sqr()).sum();
    let p_direct_energy: f32 = p_direct.iter().map(|c| c.norm_sqr()).sum();

    info!("Direct extraction results:");
    info!("  - Harmonic energy: {:.4}", h_direct_energy);
    info!("  - Percussive energy: {:.4}", p_direct_energy);

    // Energy Analysis
    info!("Energy Analysis");

    let original_energy: f32 = stft.iter().map(|c| c.norm_sqr()).sum();
    let h_sep_energy: f32 = harmonic_stft.iter().map(|c| c.norm_sqr()).sum();
    let p_sep_energy: f32 = percussive_stft.iter().map(|c| c.norm_sqr()).sum();

    info!("Energy comparison:");
    info!("  - Original signal: {:.4}", original_energy);
    info!(
        "  - Harmonic component: {:.4} ({:.1}%)",
        h_sep_energy,
        h_sep_energy / original_energy * 100.0
    );
    info!(
        "  - Percussive component: {:.4} ({:.1}%)",
        p_sep_energy,
        p_sep_energy / original_energy * 100.0
    );
    info!(
        "  - Total separated: {:.4} ({:.1}%)",
        h_sep_energy + p_sep_energy,
        (h_sep_energy + p_sep_energy) / original_energy * 100.0
    );

    Ok(())
}
