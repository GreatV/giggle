//! Chroma Features Example
//!
//! This example demonstrates various chroma feature extraction techniques.

use giggle::{feature, io};
use log::info;

fn main() {
    env_logger::init();
    info!("Chroma Features Example");

    // Generate a C major chord
    let sr = 22050;
    let duration = 2.0;

    info!("Generating C major chord (C4, E4, G4)...");
    let c4 = io::tone(261.63, sr, duration);
    let e4 = io::tone(329.63, sr, duration);
    let g4 = io::tone(392.00, sr, duration);

    let signal: Vec<f32> = c4
        .iter()
        .zip(e4.iter())
        .zip(g4.iter())
        .map(|((c, e), g)| (c + e + g) / 3.0)
        .collect();

    info!("Generated {} samples\n", signal.len());

    let n_fft = 2048;
    let hop_length = 512;

    // Chroma STFT
    info!("Chroma STFT");

    let chroma_stft =
        feature::chroma::chroma_stft(&signal, sr, n_fft, hop_length, 12, 0.0).unwrap();
    info!("Chroma STFT shape: {:?}", chroma_stft.shape());

    // Show average chroma values
    let pitch_classes = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];
    info!("\nAverage chroma values (C major chord):");
    for (i, name) in pitch_classes.iter().enumerate() {
        let avg: f32 = chroma_stft.row(i).iter().sum::<f32>() / chroma_stft.shape()[1] as f32;
        let marker = match i {
            0 => " <-- C (root)",
            4 => " <-- E (major 3rd)",
            7 => " <-- G (perfect 5th)",
            _ => "",
        };
        info!("  - {}: {:.4}{}", name, avg, marker);
    }

    // Chroma CQT
    info!("Chroma CQT");

    let chroma_cqt = feature::chroma::chroma_cqt(
        &signal, sr, hop_length, 32.7, 12, 7, 36, 0.0, -1, 0.0, "full",
    )
    .unwrap();
    info!("Chroma CQT shape: {:?}", chroma_cqt.shape());

    // Chroma CENS
    info!("Chroma CENS");

    let chroma_cens =
        feature::chroma::chroma_cens(&signal, sr, hop_length, 32.7, 12, 7, 36, 0.0, 41, "full")
            .unwrap();
    info!("Chroma CENS shape: {:?}", chroma_cens.shape());
}
