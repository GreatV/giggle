//! Spectral Contrast Example
//!
//! This example demonstrates spectral contrast extraction.

use giggle::feature::contrast;
use giggle::io;
use log::info;

fn main() {
    env_logger::init();
    info!("Spectral Contrast Example");

    // Generate different types of signals
    let sr = 22050;
    let duration = 2.0;

    // Harmonic signal (tone)
    info!("Generating harmonic signal (440 Hz tone)...");
    let harmonic = io::tone(440.0, sr, duration);

    let hop_length = 512;
    let n_fft = 2048;
    let n_bands = 6;

    // Spectral Contrast for Harmonic Signal
    info!("Spectral Contrast - Harmonic Signal");

    let fmin = 200.0; // Minimum frequency for contrast analysis
    let contrast_harm =
        contrast::spectral_contrast(&harmonic, sr, n_fft, hop_length, n_bands, fmin).unwrap();

    info!("Contrast shape: {:?}", contrast_harm.shape());

    // Show average contrast values
    info!("\nAverage spectral contrast (harmonic):");
    for i in 0..contrast_harm.shape()[0] {
        let avg: f32 = contrast_harm.row(i).iter().sum::<f32>() / contrast_harm.shape()[1] as f32;
        info!("  Band {}: {:.4}", i, avg);
    }
}
