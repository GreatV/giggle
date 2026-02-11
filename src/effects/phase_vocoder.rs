use crate::spectrum::{StftConfig, istft, stft};
use crate::window;
use ndarray::Array2;
use num_complex::Complex32;

pub fn phase_vocoder(
    stft_matrix: &Array2<Complex32>,
    rate: f32,
    hop_length: usize,
) -> Array2<Complex32> {
    let n_freq = stft_matrix.shape().first().copied().unwrap_or(0);
    let n_frames = stft_matrix.shape().get(1).copied().unwrap_or(0);
    if n_freq == 0 || n_frames == 0 || rate <= 0.0 {
        return Array2::<Complex32>::zeros((0, 0));
    }

    let n_fft = (n_freq - 1) * 2;
    let hop = hop_length.max(1) as f32;
    let omega: Vec<f32> = (0..n_freq)
        .map(|i| 2.0 * std::f32::consts::PI * i as f32 / n_fft as f32)
        .collect();

    let mut time_steps = Vec::new();
    let mut t = 0.0f32;
    while t < (n_frames - 1) as f32 {
        time_steps.push(t);
        t += rate;
    }

    let out_frames = time_steps.len();
    let mut out = Array2::<Complex32>::zeros((n_freq, out_frames));
    let mut phase_acc: Vec<f32> = vec![0.0; n_freq];
    let mut last_phase: Vec<f32> = vec![0.0; n_freq];

    for (out_idx, step) in time_steps.iter().enumerate() {
        let idx = step.floor() as usize;
        let frac = step - idx as f32;
        if idx + 1 >= n_frames {
            break;
        }

        for f in 0..n_freq {
            let a = stft_matrix[(f, idx)];
            let b = stft_matrix[(f, idx + 1)];
            let mag = (1.0 - frac) * a.norm() + frac * b.norm();

            let phase_a = a.arg();
            let phase_b = b.arg();

            let mut delta = phase_b - phase_a - omega[f] * hop;
            delta = (delta + std::f32::consts::PI) % (2.0 * std::f32::consts::PI)
                - std::f32::consts::PI;
            let true_freq = omega[f] + delta / hop;

            if out_idx == 0 {
                last_phase[f] = phase_a;
                phase_acc[f] = phase_a;
            } else {
                phase_acc[f] += true_freq * hop;
                last_phase[f] = phase_b;
            }

            out[(f, out_idx)] = Complex32::from_polar(mag, phase_acc[f]);
        }
    }

    out
}

pub fn time_stretch(
    y: &[f32],
    rate: f32,
    n_fft: usize,
    hop_length: usize,
) -> crate::Result<Vec<f32>> {
    if y.is_empty() || rate <= 0.0 {
        return Ok(Vec::new());
    }
    let mut cfg = StftConfig::default();
    cfg.n_fft = n_fft;
    cfg.win_length = n_fft;
    cfg.hop_length = hop_length.max(1);
    cfg.window = window::hann(cfg.win_length);

    let stft_matrix = stft(y, &cfg)?;
    let stretched = phase_vocoder(&stft_matrix, rate, cfg.hop_length);
    istft(&stretched, &cfg, None)
}
