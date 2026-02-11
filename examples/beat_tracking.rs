//! Beat Tracking Example
//!
//! This example demonstrates tempo estimation and beat tracking.

use giggle::beat;
use log::info;

fn main() {
    env_logger::init();
    info!("Beat Tracking Example");

    // Generate a rhythmic signal with a clear tempo
    let sr = 22050;
    let bpm = 120.0;
    let duration = 8.0;
    let beats_per_second = bpm / 60.0;
    let samples_per_beat = (sr as f32 / beats_per_second) as usize;

    info!("Generating rhythmic signal at {} BPM...", bpm);
    let mut signal = vec![0.0f32; (sr as f32 * duration) as usize];

    // Add kick drum on beats
    for beat in 0..(signal.len() / samples_per_beat) {
        let pos = beat * samples_per_beat;
        for i in 0..1000 {
            if pos + i < signal.len() {
                let t = i as f32 / sr as f32;
                let freq = 150.0 * (1.0 - t * 10.0).exp();
                signal[pos + i] += 0.5 * (2.0 * std::f32::consts::PI * freq * t).sin();
            }
        }
    }

    info!("Generated {} samples\n", signal.len());

    let hop_length = 512;

    // Beat Tracking
    info!("Beat Tracking");

    let (tempo, beats) = beat::beat_track(&signal, sr, None, hop_length, None).unwrap();

    info!("Estimated tempo: {:.1} BPM", tempo);
    info!("Expected tempo: {} BPM", bpm as i32);
    info!("Error: {:.1} BPM", (tempo - bpm).abs());

    info!("\nDetected {} beats", beats.len());

    // Convert beat frames to times
    let beat_times: Vec<f32> = beats
        .iter()
        .map(|&frame| frame as f32 * hop_length as f32 / sr as f32)
        .collect();

    info!("\nFirst 10 beat times (seconds):");
    for (i, time) in beat_times.iter().take(10).enumerate() {
        info!("  Beat {}: {:.3}s", i + 1, time);
    }

    // PLP Analysis
    info!("PLP (Predominant Local Pulse)");

    let plp = beat::plp(&signal, sr, None, hop_length, 384, 30.0, 300.0).unwrap();

    info!("PLP envelope: {} frames", plp.len());

    let plp_mean: f32 = plp.iter().sum::<f32>() / plp.len() as f32;
    info!("  - Mean PLP: {:.4}", plp_mean);
}
