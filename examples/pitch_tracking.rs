//! Pitch Tracking Example
//!
//! This example demonstrates pitch tracking with the YIN algorithm.

use giggle::{io, pitch};
use log::{error, info};

fn main() {
    env_logger::init();
    info!("Pitch Tracking Example");

    // Generate a sequence of notes
    let sr = 22050;

    let notes = vec![
        (220.0, 1.0),  // A3
        (261.63, 1.0), // C4
        (329.63, 1.0), // E4
        (440.0, 1.0),  // A4
    ];

    info!("Generating sequence of notes:");
    let mut signal = Vec::new();
    for (freq, duration) in &notes {
        info!("  - {:.2} Hz for {} seconds", freq, duration);
        let tone = io::tone(*freq, sr, *duration);
        signal.extend(tone);
    }

    info!("\nGenerated {} samples", signal.len());

    let hop_length = 512;
    let frame_length = 2048;

    // YIN Pitch Tracking
    info!("YIN Pitch Tracking");

    let yin_freqs = pitch::yin(
        &signal,
        sr,
        frame_length,
        hop_length,
        40.0,
        sr as f32 / 4.0,
        0.1,
    )
    .unwrap_or_else(|e| {
        error!("Failed to compute YIN pitch: {}", e);
        vec![]
    });

    info!("YIN output: {} frames", yin_freqs.len());

    // Calculate average pitch for each note segment
    let frames_per_note = (sr as f32 * 1.0 / hop_length as f32) as usize;
    info!("\nAverage pitch per note (YIN):");
    for (i, (expected_freq, _)) in notes.iter().enumerate() {
        let start = i * frames_per_note;
        let end = ((i + 1) * frames_per_note).min(yin_freqs.len());

        if start < yin_freqs.len() {
            let valid_freqs: Vec<f32> = yin_freqs[start..end]
                .iter()
                .filter(|&&f| f > 0.0)
                .cloned()
                .collect();
            if !valid_freqs.is_empty() {
                let avg_freq: f32 = valid_freqs.iter().sum::<f32>() / valid_freqs.len() as f32;
                let error = (avg_freq - expected_freq).abs() / expected_freq * 100.0;
                info!(
                    "  Note {}: {:.2} Hz (expected {:.2} Hz, error {:.1}%)",
                    i + 1,
                    avg_freq,
                    expected_freq,
                    error
                );
            } else {
                info!("  Note {}: No pitch detected", i + 1);
            }
        }
    }
}
