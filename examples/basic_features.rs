//! Basic Feature Extraction Example
//!
//! This example demonstrates basic feature extraction using giggle:
//! - Generating synthetic tones
//! - Computing mel spectrogram
//! - Computing MFCCs
//! - Computing chroma features

use giggle::{feature, io};
use log::info;

fn main() {
    env_logger::init();
    info!("Basic Feature Extraction Example");

    // Generate a 440 Hz tone (A4) at 22050 Hz sample rate, 2 seconds duration
    let sr = 22050;
    let duration = 2.0;
    info!("Generating {} Hz tone...", 440.0);
    let signal = io::tone(440.0, sr, duration);
    info!("Generated {} samples\n", signal.len());

    // Mel Spectrogram
    info!("Mel Spectrogram");

    let n_fft = 2048;
    let hop_length = 512;
    let n_mels = 128;

    let mel_spec = feature::mel::melspectrogram(&signal, sr, n_fft, hop_length, n_mels).unwrap();
    info!("Mel spectrogram shape: {:?}", mel_spec.shape());
    info!("  - Mel bands: {}", mel_spec.shape()[0]);
    info!("  - Time frames: {}", mel_spec.shape()[1]);

    // MFCC Extraction
    info!("MFCC Extraction");

    let n_mfcc = 13;
    let mfcc = feature::mfcc::mfcc(&signal, sr, n_mfcc, n_fft, hop_length, 128).unwrap();
    info!("MFCC shape: {:?}", mfcc.shape());
    info!("  - Coefficients: {}", mfcc.shape()[0]);
    info!("  - Time frames: {}", mfcc.shape()[1]);

    // Chroma Features
    info!("Chroma Features");

    let chroma = feature::chroma::chroma_stft(&signal, sr, n_fft, hop_length, 12, 0.0).unwrap();
    info!("Chroma STFT shape: {:?}", chroma.shape());
    info!("  - Pitch classes: {}", chroma.shape()[0]);
    info!("  - Time frames: {}", chroma.shape()[1]);
}
