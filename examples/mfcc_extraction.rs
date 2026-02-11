//! MFCC Extraction Example
//!
//! This example demonstrates Mel-Frequency Cepstral Coefficient (MFCC) extraction.

use giggle::{feature, io};
use log::info;

fn main() {
    env_logger::init();
    info!("MFCC Extraction Example");

    // Generate a test signal
    let sr = 22050;
    let duration = 2.0;

    info!("Generating test signal...");
    let signal = io::tone(440.0, sr, duration);
    info!("Generated {} samples\n", signal.len());

    let n_fft = 2048;
    let hop_length = 512;

    // MFCC Extraction
    info!("MFCC Extraction");

    let n_mfcc = 13;
    let mfcc = feature::mfcc::mfcc(&signal, sr, n_mfcc, n_fft, hop_length, 128).unwrap();

    info!("MFCC shape: {:?}", mfcc.shape());
    info!("  - Coefficients: {}", mfcc.shape()[0]);
    info!("  - Time frames: {}", mfcc.shape()[1]);

    // Show statistics for each coefficient
    info!("\nMFCC statistics:");
    for i in 0..n_mfcc.min(5) {
        let row = mfcc.row(i);
        let mean: f32 = row.iter().sum::<f32>() / row.len() as f32;
        info!("  MFCC[{}]: mean={:.4}", i, mean);
    }

    // Different Numbers of Coefficients
    info!("Different Numbers of Coefficients");

    for n in [5, 13, 20] {
        let mfcc_n = feature::mfcc::mfcc(&signal, sr, n, n_fft, hop_length, 128).unwrap();
        info!("n_mfcc={}: shape {:?}", n, mfcc_n.shape());
    }

    // Mel Spectrogram
    info!("Mel Spectrogram");

    let n_mels = 128;
    let mel_spec = feature::mel::melspectrogram(&signal, sr, n_fft, hop_length, n_mels).unwrap();

    info!("Mel spectrogram shape: {:?}", mel_spec.shape());
}
