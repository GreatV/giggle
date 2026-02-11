use giggle::{fft, frame, io, spectrum, utils, window};
use ndarray::Array2;
use rand::Rng;

#[test]
fn window_lengths() {
    assert_eq!(window::hann(0).len(), 0);
    assert_eq!(window::hann(8).len(), 8);
    assert_eq!(window::hamming(8).len(), 8);
    assert_eq!(window::blackman(8).len(), 8);
    assert_eq!(window::bartlett(8).len(), 8);
}

#[test]
fn stft_istft_roundtrip() {
    let mut rng = rand::thread_rng();
    let y: Vec<f32> = (0..16000).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect();

    let mut config = spectrum::StftConfig::default();
    config.n_fft = 512;
    config.win_length = 512;
    config.hop_length = 128;
    config.window = window::hann(config.win_length);

    let stft = spectrum::stft(&y, &config).unwrap();
    let y_rec = spectrum::istft(&stft, &config, Some(y.len())).unwrap();

    let mse = utils::mse(&y, &y_rec);
    assert!(mse < 1e-3, "mse too high: {mse}");
}

#[test]
fn wav_roundtrip() {
    let sample_rate = 16000u32;
    let mut rng = rand::thread_rng();
    let frames = 1024;
    let mut data = Array2::<f32>::zeros((1, frames));
    for i in 0..frames {
        data[(0, i)] = rng.gen_range(-0.8f32..0.8f32);
    }

    let path = std::env::temp_dir().join("giggle_test.wav");
    io::save_wav(&path, &data, sample_rate).unwrap();
    let (loaded, spec) = io::load_wav(&path, None, None).unwrap();
    std::fs::remove_file(&path).ok();

    assert_eq!(spec.sample_rate, sample_rate);
    assert_eq!(loaded.shape(), data.shape());

    let mse = utils::mse(data.as_slice().unwrap(), loaded.as_slice().unwrap());
    assert!(mse < 2e-4, "mse too high: {mse}");
}

#[test]
fn resample_length_scales() {
    let sr_in = 16000u32;
    let sr_out = 22050u32;
    let frames = 16000;
    let mut data = Array2::<f32>::zeros((1, frames));
    for i in 0..frames {
        data[(0, i)] = (i as f32 / 100.0).sin();
    }

    let out = io::resample(&data, sr_in, sr_out).unwrap();
    let expected = (frames as f64 * sr_out as f64 / sr_in as f64) as usize;
    let got = out.shape()[1];

    let diff = expected.abs_diff(got);
    assert!(
        diff <= 2,
        "resample length off: expected {expected}, got {got}"
    );
}

#[test]
fn frame_count_centered() {
    let y = vec![0.0f32; 10];
    let frames = frame::frame_signal(&y, 4, 2, true).unwrap();
    assert_eq!(frames.len(), 6);
}

#[test]
fn rfft_length() {
    let y = vec![0.0f32; 8];
    let out = fft::rfft(&y);
    assert_eq!(out.len(), 5);
}
