use giggle::{feature, spectrum, window};

#[test]
fn default_stft_shapes() {
    let y = vec![0.0f32; 4096];
    let cfg = spectrum::StftConfig::default();
    let stft = spectrum::stft(&y, &cfg).unwrap();
    assert_eq!(stft.shape()[0], cfg.n_fft / 2 + 1);
}

#[test]
fn mel_frequency_bounds() {
    let freqs = feature::mel::mel_frequencies(10, 0.0, 8000.0);
    assert!(!freqs.is_empty());
    assert!(freqs.first().unwrap() >= &0.0);
    assert!(*freqs.last().unwrap() <= 8000.0 * 1.02);
}

#[test]
fn window_length_matches() {
    let w = window::hann(128);
    assert_eq!(w.len(), 128);
}
