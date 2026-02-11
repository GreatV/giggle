use crate::spectrum::{StftConfig, stft};
use crate::window;
use ndarray::Array2;

/// Compute spectral contrast.
///
/// Spectral contrast is defined as the difference in magnitude between
/// peaks and valleys in the frequency spectrum, partitioned into bands.
///
/// # Arguments
/// * `y` - Input audio signal
/// * `sr` - Sample rate
/// * `n_fft` - FFT window size
/// * `hop_length` - Number of samples between frames
/// * `n_bands` - Number of contrast bands
/// * `fmin` - Minimum frequency for the first band (must be > 0)
///
/// # Returns
/// Spectral contrast matrix of shape (n_bands + 1, n_frames)
///
/// # Errors
/// Returns `Error::EmptyAudio` if input is empty
/// Returns `Error::InvalidSize` if n_bands or n_fft is 0
/// Returns `Error::InvalidFrequencyRange` if fmin <= 0
pub fn spectral_contrast(
    y: &[f32],
    sr: u32,
    n_fft: usize,
    hop_length: usize,
    n_bands: usize,
    fmin: f32,
) -> crate::Result<Array2<f32>> {
    if y.is_empty() {
        return Err(crate::Error::EmptyAudio);
    }
    if n_bands == 0 {
        return Err(crate::Error::InvalidSize {
            name: "n_bands",
            value: 0,
            reason: "must be > 0",
        });
    }
    if n_fft == 0 {
        return Err(crate::Error::InvalidSize {
            name: "n_fft",
            value: 0,
            reason: "must be > 0",
        });
    }
    if fmin <= 0.0 {
        return Err(crate::Error::InvalidFrequencyRange {
            fmin,
            fmax: sr as f32 / 2.0,
            reason: "fmin must be > 0".to_string(),
        });
    }

    let mut cfg = StftConfig::default();
    cfg.n_fft = n_fft;
    cfg.win_length = n_fft;
    cfg.hop_length = hop_length.max(1);
    cfg.window = window::hann(cfg.win_length);

    // Match librosa's default pad_mode="constant" for spectral_contrast.
    let pad = n_fft / 2;
    let mut padded = vec![0.0f32; y.len() + 2 * pad];
    padded[pad..pad + y.len()].copy_from_slice(y);
    cfg.center = false;
    let stft_matrix = stft(&padded, &cfg)?;
    let n_freq = stft_matrix.shape()[0];
    let n_frames = stft_matrix.shape()[1];
    if n_freq == 0 || n_frames == 0 {
        return Ok(Array2::<f32>::zeros((0, 0)));
    }

    let nyquist = sr as f32 / 2.0;
    if nyquist <= fmin {
        return Ok(Array2::<f32>::zeros((0, 0)));
    }

    let mut octa = vec![0.0f32; n_bands + 2];
    for i in 0..=n_bands {
        octa[i + 1] = fmin * 2.0f32.powf(i as f32);
    }

    let mut freqs = vec![0.0f32; n_freq];
    for (k, freq) in freqs.iter_mut().enumerate().take(n_freq) {
        *freq = k as f32 * sr as f32 / n_fft as f32;
    }

    let mut valley = Array2::<f32>::zeros((n_bands + 1, n_frames));
    let mut peak = Array2::<f32>::zeros((n_bands + 1, n_frames));

    for k in 0..=n_bands {
        let f_low = octa[k];
        let f_high = octa[k + 1];

        let mut current_band = vec![false; n_freq];
        for (i, f) in freqs.iter().enumerate() {
            if *f >= f_low && *f <= f_high {
                current_band[i] = true;
            }
        }

        let mut idxs: Vec<usize> = current_band
            .iter()
            .enumerate()
            .filter_map(|(i, v)| if *v { Some(i) } else { None })
            .collect();
        if idxs.is_empty() {
            continue;
        }

        if k > 0 && idxs[0] > 0 {
            current_band[idxs[0] - 1] = true;
            idxs.insert(0, idxs[0] - 1);
        }
        if k == n_bands {
            let last = *idxs.last().unwrap();
            for (i, band_val) in current_band
                .iter_mut()
                .enumerate()
                .take(n_freq)
                .skip(last + 1)
            {
                *band_val = true;
                idxs.push(i);
            }
        }

        if k < n_bands && !idxs.is_empty() {
            idxs.pop();
        }

        let count = current_band.iter().filter(|v| **v).count();
        let mut q = (0.02 * count as f32).round() as usize;
        if q < 1 {
            q = 1;
        }

        for t in 0..n_frames {
            let mut mags = Vec::with_capacity(idxs.len());
            for &f in &idxs {
                let v = stft_matrix[(f, t)];
                mags.push((v.re * v.re + v.im * v.im).sqrt());
            }
            mags.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let low = mags.iter().take(q).sum::<f32>() / q as f32;
            let high = mags.iter().rev().take(q).sum::<f32>() / q as f32;
            valley[(k, t)] = low.max(1e-10);
            peak[(k, t)] = high.max(1e-10);
        }
    }

    let mut out = Array2::<f32>::zeros((n_bands + 1, n_frames));
    let mut max_db = f32::NEG_INFINITY;
    for k in 0..=n_bands {
        for t in 0..n_frames {
            let p = peak[(k, t)];
            let p_db = 10.0 * p.max(1e-10).log10();
            if p_db > max_db {
                max_db = p_db;
            }
        }
    }
    let min_db = max_db - 80.0;
    for k in 0..=n_bands {
        for t in 0..n_frames {
            let p = peak[(k, t)];
            let v = valley[(k, t)];
            let mut p_db = 10.0 * p.max(1e-10).log10();
            let mut v_db = 10.0 * v.max(1e-10).log10();
            if p_db < min_db {
                p_db = min_db;
            }
            if v_db < min_db {
                v_db = min_db;
            }
            out[(k, t)] = p_db - v_db;
        }
    }

    Ok(out)
}
