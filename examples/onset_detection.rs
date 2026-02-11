//! Onset Detection Example
//!
//! This example demonstrates onset detection methods.

use giggle::onset;
use log::info;

fn main() {
    env_logger::init();
    info!("Onset Detection Example");

    // Generate a signal with regular onsets
    let sr = 22050;
    let duration = 5.0;
    let onset_interval = (sr as f32 * 0.5) as usize; // 2 onsets per second

    info!(
        "Generating rhythmic signal with onsets every {} samples...",
        onset_interval
    );
    let mut signal = vec![0.0f32; (sr as f32 * duration) as usize];

    // Add transient onsets
    for i in 0..(signal.len() / onset_interval) {
        let pos = i * onset_interval;
        for j in 0..500 {
            if pos + j < signal.len() {
                let j_f32 = j as f32;
                let decay = (-j_f32 / 100.0).exp();
                signal[pos + j] += 0.5 * decay;
            }
        }
    }

    info!("Generated {} samples\n", signal.len());

    let hop_length = 512;
    let n_fft = 2048;

    // Onset Strength Envelope
    info!("Onset Strength Envelope");

    let onset_env = onset::strength::onset_strength(&signal, n_fft, hop_length).unwrap();

    info!("Onset strength envelope: {} frames", onset_env.len());

    let mean_strength: f32 = onset_env.iter().sum::<f32>() / onset_env.len() as f32;
    let max_strength = onset_env.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    info!("  - Mean strength: {:.4}", mean_strength);
    info!("  - Max strength: {:.4}", max_strength);

    // Onset Detection
    info!("Onset Detection");

    let threshold = 0.5;
    let onsets = onset::detect::onset_detect(&signal, n_fft, hop_length, threshold).unwrap();

    info!("Detected {} onsets", onsets.len());

    // Convert onset frames to times
    let onset_times: Vec<f32> = onsets
        .iter()
        .map(|&frame| frame as f32 * hop_length as f32 / sr as f32)
        .collect();

    info!("\nFirst 10 onset times (seconds):");
    for (i, time) in onset_times.iter().take(10).enumerate() {
        info!("  Onset {}: {:.3}s", i + 1, time);
    }
}
