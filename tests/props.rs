use giggle::spectrum::{StftConfig, istft, stft};
use giggle::window;
use proptest::prelude::*;

fn energy_time(x: &[f32]) -> f32 {
    x.iter().map(|v| v * v).sum()
}

proptest! {
    #[test]
    fn stft_istft_roundtrip_prop(len in 512usize..4096) {
        let mut cfg = StftConfig::default();
        cfg.n_fft = 256;
        cfg.win_length = 256;
        cfg.hop_length = 64;
        cfg.window = window::hann(cfg.win_length);

        let y: Vec<f32> = (0..len).map(|i| ((i as f32) * 0.01).sin()).collect();
        let s = stft(&y, &cfg).unwrap();
        let y_rec = istft(&s, &cfg, Some(y.len())).unwrap();
        let mut mse = 0.0f32;
        for i in 0..y.len() {
            let d = y[i] - y_rec[i];
            mse += d * d;
        }
        mse /= y.len() as f32;
        prop_assert!(mse < 1e-3);
    }

    #[test]
    fn parseval_energy_consistency(len in 512usize..4096) {
        let mut cfg = StftConfig::default();
        cfg.n_fft = 512;
        cfg.win_length = 512;
        cfg.hop_length = 128;
        cfg.window = window::hann(cfg.win_length);

        let y: Vec<f32> = (0..len).map(|i| ((i as f32) * 0.02).cos()).collect();
        let s = stft(&y, &cfg).unwrap();
        let y_rec = istft(&s, &cfg, Some(y.len())).unwrap();

        let e_time = energy_time(&y);
        let e_rec = energy_time(&y_rec);
        let ratio = (e_time - e_rec).abs() / e_time.max(1e-9);
        prop_assert!(ratio < 0.05);
    }
}
