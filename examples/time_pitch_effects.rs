//! Time Stretching and Pitch Shifting Example
//!
//! This example demonstrates time stretching and pitch shifting effects
//! using the phase vocoder technique.

use giggle::{effects, io, spectrum};
use log::info;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    info!("Time Stretching and Pitch Shifting");

    let sr = 22050;

    // Generate Test Signal
    info!("Generating Test Signal");

    // Create a melodic phrase: C4 - E4 - G4 - C5
    let note_duration = 0.5;
    let notes = [261.63, 329.63, 392.00, 523.25]; // C4, E4, G4, C5
    let note_names = ["C4", "E4", "G4", "C5"];

    let mut original_signal = Vec::new();
    for (freq, name) in notes.iter().zip(note_names.iter()) {
        let tone = io::tone(*freq, sr, note_duration);
        original_signal.extend(tone);
        info!(
            "Added {} ({:.2} Hz): {} samples",
            name,
            freq,
            original_signal.len()
        );
    }

    info!(
        "\nOriginal signal: {} samples ({:.2}s)",
        original_signal.len(),
        original_signal.len() as f32 / sr as f32
    );

    let n_fft = 2048;
    let hop_length = 512;

    // Time Stretching
    info!("Time Stretching");

    let stretch_factors = [0.5, 1.0, 1.5, 2.0];

    for rate in &stretch_factors {
        let stretched =
            effects::phase_vocoder::time_stretch(&original_signal, *rate, n_fft, hop_length)?;

        let original_duration = original_signal.len() as f32 / sr as f32;
        let new_duration = stretched.len() as f32 / sr as f32;

        info!(
            "  Rate {:.1}x: {} samples ({:.2}s -> {:.2}s)",
            rate,
            stretched.len(),
            original_duration,
            new_duration
        );
    }

    // Pitch Shifting
    info!("Pitch Shifting");

    let shift_semitones = [-12.0, -5.0, 0.0, 5.0, 12.0];

    for n_steps in &shift_semitones {
        let shifted =
            effects::time_pitch::pitch_shift(&original_signal, sr, *n_steps, n_fft, hop_length)?;

        let direction = if *n_steps < 0.0 {
            "down"
        } else if *n_steps > 0.0 {
            "up"
        } else {
            "unchanged"
        };

        info!(
            "  {:+5.1} semitones ({:4}): {} samples",
            n_steps,
            direction,
            shifted.len()
        );
    }

    // Detailed Analysis: Phase Vocoder
    info!("Phase Vocoder Analysis");

    // Compute STFT
    let cfg = spectrum::StftConfig {
        n_fft,
        hop_length,
        ..Default::default()
    };

    let stft = spectrum::stft(&original_signal, &cfg)?;
    info!("Input STFT: {:?}", stft.shape());

    // Apply phase vocoder at different rates
    let vocoder_rates = [0.5, 1.0, 2.0];

    for rate in &vocoder_rates {
        let modified_stft = effects::phase_vocoder::phase_vocoder(&stft, *rate, hop_length);
        info!(
            "  Rate {:.1}x: output shape {:?}",
            rate,
            modified_stft.shape()
        );

        // Reconstruct and check length
        let reconstructed = spectrum::istft(&modified_stft, &cfg, None)?;
        let expected_duration = original_signal.len() as f32 / rate;
        let _actual_duration = reconstructed.len() as f32;

        info!(
            "    Reconstructed: {} samples (expected ~{:.0})",
            reconstructed.len(),
            expected_duration
        );
    }

    // Combined Effects
    info!("Combined Effects");

    // Example: Slow down by 2x and shift up by 7 semitones (perfect fifth)
    info!("Example: Slow down 2x + Pitch up 7 semitones");

    let slow = effects::phase_vocoder::time_stretch(&original_signal, 0.5, n_fft, hop_length)?;
    let slow_and_shifted = effects::time_pitch::pitch_shift(&slow, sr, 7.0, n_fft, hop_length)?;

    info!("  Original: {} samples", original_signal.len());
    info!("  After time stretch (0.5x): {} samples", slow.len());
    info!(
        "  After pitch shift (+7st): {} samples",
        slow_and_shifted.len()
    );

    // Example: Create harmonized version (original + shifted versions)
    info!("\nExample: Harmonized version (original + 3rd + 5th)");

    let third = effects::time_pitch::pitch_shift(&original_signal, sr, 4.0, n_fft, hop_length)?;
    let fifth = effects::time_pitch::pitch_shift(&original_signal, sr, 7.0, n_fft, hop_length)?;

    // Mix the harmonized versions
    let harmonized: Vec<f32> = original_signal
        .iter()
        .zip(third.iter())
        .zip(fifth.iter())
        .map(|((o, t), f)| (o + t * 0.7 + f * 0.7) / 2.4)
        .collect();

    info!("  Original: {} samples", original_signal.len());
    info!("  Third (+4st): {} samples", third.len());
    info!("  Fifth (+7st): {} samples", fifth.len());
    info!("  Harmonized mix: {} samples", harmonized.len());

    // Quality Comparison
    info!("Quality Metrics");

    // Test pitch shifting quality by measuring frequency preservation
    // Generate a pure tone, shift it, and compare
    let test_freq = 440.0;
    let test_tone = io::tone(test_freq, sr, 1.0);

    info!("Pitch shift quality test ({} Hz tone):", test_freq);

    for n_steps in [0.0, 4.0, 7.0, 12.0] {
        let shifted = effects::time_pitch::pitch_shift(&test_tone, sr, n_steps, n_fft, hop_length)?;

        // Expected frequency
        let expected_freq = test_freq * 2.0f32.powf(n_steps / 12.0);

        // Calculate energy preservation
        let original_energy: f32 = test_tone.iter().map(|s| s * s).sum();
        let shifted_energy: f32 = shifted.iter().map(|s| s * s).sum();
        let energy_ratio = if original_energy > 0.0 {
            shifted_energy / original_energy
        } else {
            0.0
        };

        info!(
            "  {:+4.1}st -> {:7.2} Hz: energy ratio = {:.3}",
            n_steps, expected_freq, energy_ratio
        );
    }

    // Performance Notes

    Ok(())
}
