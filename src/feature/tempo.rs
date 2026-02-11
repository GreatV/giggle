use crate::fft::FftPlan;
use crate::onset::strength::onset_strength;
use ndarray::Array2;
use num_complex::Complex32;

/// Compute tempogram from onset strength envelope.
///
/// A tempogram represents the distribution of rhythmic periodicities over time.
/// This implementation uses local autocorrelation of the onset strength envelope.
///
/// # Arguments
/// * `onset_env` - Onset strength envelope
/// * `sr` - Sample rate
/// * `hop_length` - Hop length used for STFT
/// * `win_length` - Window length for local autocorrelation (in frames)
///
/// # Returns
/// Tempogram matrix of shape (win_length, n_frames)
pub fn tempogram(
    onset_env: &[f32],
    _sr: u32,
    _hop_length: usize,
    win_length: usize,
) -> Array2<f32> {
    if onset_env.is_empty() || win_length == 0 {
        return Array2::<f32>::zeros((0, 0));
    }

    let n_frames = onset_env.len();
    let max_lag = win_length.min(n_frames);

    let n_out_frames = if n_frames >= win_length {
        n_frames - win_length + 1
    } else {
        0
    };

    let mut tempogram = Array2::<f32>::zeros((max_lag, n_out_frames));

    for t in 0..n_out_frames {
        let window = &onset_env[t..t + win_length];

        for lag in 0..max_lag {
            let mut corr = 0.0f32;
            for i in 0..(win_length - lag) {
                corr += window[i] * window[i + lag];
            }
            tempogram[(lag, t)] = corr;
        }
    }

    tempogram
}

/// Estimate the tempo (BPM) of an audio signal.
///
/// This function estimates the tempo by analyzing the periodicity of the
/// onset strength envelope using autocorrelation.
///
/// # Arguments
/// * `y` - Input audio signal
/// * `sr` - Sample rate in Hz
/// * `n_fft` - FFT size for onset strength computation
/// * `hop_length` - Hop length for onset strength computation
///
/// # Returns
/// Estimated tempo in beats per minute (BPM), or 0.0 if tempo cannot be determined
///
/// # Example
/// ```
/// use giggle::feature::tempo::tempo;
///
/// let signal = vec![0.1f32; 22050];
/// let bpm = tempo(&signal, 22050, 2048, 512).unwrap();
/// assert!(bpm >= 0.0);
/// ```
pub fn tempo(y: &[f32], sr: u32, n_fft: usize, hop_length: usize) -> crate::Result<f32> {
    let env = onset_strength(y, n_fft, hop_length)?;
    if env.len() < 3 || hop_length == 0 || sr == 0 {
        return Ok(0.0);
    }

    let max_lag = env.len().min((sr as usize / hop_length).max(1));
    let min_lag = (sr as f32 / hop_length as f32 / 4.0).max(1.0) as usize; // up to 240 BPM

    let mut best_lag = 0usize;
    let mut best_corr = 0.0f32;

    for lag in min_lag..max_lag {
        let mut sum = 0.0f32;
        for i in 0..(env.len() - lag) {
            sum += env[i] * env[i + lag];
        }
        if sum > best_corr {
            best_corr = sum;
            best_lag = lag;
        }
    }

    if best_lag == 0 {
        return Ok(0.0);
    }

    let period_sec = best_lag as f32 * hop_length as f32 / sr as f32;
    Ok(60.0 / period_sec)
}

/// Compute Fourier tempogram using FFT.
///
/// # Arguments
/// * `onset_env` - Onset strength envelope
/// * `sr` - Sample rate
/// * `hop_length` - Hop length used for STFT
/// * `win_length` - Window length for local FFT (in frames)
///
/// # Returns
/// Fourier tempogram (tempo_bins x time_frames)
pub fn fourier_tempogram(
    onset_env: &[f32],
    _sr: u32,
    _hop_length: usize,
    win_length: usize,
) -> Array2<f32> {
    if onset_env.is_empty() || win_length == 0 {
        return Array2::<f32>::zeros((0, 0));
    }

    let n_frames = onset_env.len();
    if n_frames < win_length {
        return Array2::<f32>::zeros((0, 0));
    }

    let n_out_frames = n_frames - win_length + 1;
    let fft_size = win_length.next_power_of_two();
    let n_freq = fft_size / 2 + 1;

    let mut tempogram = Array2::<f32>::zeros((n_freq, n_out_frames));
    let fft = FftPlan::new(fft_size);

    for t in 0..n_out_frames {
        let window = &onset_env[t..t + win_length];

        // Prepare FFT buffer
        let mut buffer = vec![Complex32::new(0.0, 0.0); fft_size];
        for i in 0..win_length {
            buffer[i].re = window[i];
        }

        // Apply Hann window
        for (i, buf) in buffer.iter_mut().enumerate().take(win_length) {
            let w = 0.5
                * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (win_length - 1) as f32).cos());
            buf.re *= w;
        }

        fft.forward(&mut buffer);

        // Store magnitude
        for f in 0..n_freq {
            tempogram[(f, t)] = buffer[f].norm();
        }
    }

    tempogram
}

/// Convert tempogram frequency bins to BPM values.
///
/// # Arguments
/// * `bin_idx` - Frequency bin index
/// * `sr` - Sample rate
/// * `hop_length` - Hop length
/// * `win_length` - Window length used in tempogram
///
/// # Returns
/// BPM value for the given bin
pub fn tempo_frequencies(n_bins: usize, sr: u32, hop_length: usize, win_length: usize) -> Vec<f32> {
    if win_length == 0 {
        return Vec::new();
    }

    let fft_size = win_length.next_power_of_two();
    let mut bpms = Vec::with_capacity(n_bins);

    for bin in 0..n_bins {
        // Frequency in Hz
        let freq_hz = bin as f32 * sr as f32 / (hop_length as f32 * fft_size as f32);
        // Convert to BPM
        let bpm = freq_hz * 60.0;
        bpms.push(bpm);
    }

    bpms
}

/// Compute ratio of local tempogram to global tempogram.
///
/// This identifies rhythmic patterns by comparing the local tempogram
/// to a global reference (typically the mean across time). Values > 1
/// indicate stronger-than-average rhythmic energy at that tempo/time.
///
/// # Arguments
/// * `tempogram` - Input tempogram (tempo_bins x time_frames)
///
/// # Returns
/// Tempogram ratio with same shape as input
///
/// # Example
/// ```
/// use giggle::feature::tempo::{tempogram, tempogram_ratio};
/// let onset_env = vec![0.1, 0.5, 0.3, 0.8, 0.2, 0.6, 0.4, 0.9];
/// let tg = tempogram(&onset_env, 22050, 512, 4);
/// let ratio = tempogram_ratio(&tg);
/// assert_eq!(ratio.shape(), tg.shape());
/// ```
pub fn tempogram_ratio(tempogram: &Array2<f32>) -> Array2<f32> {
    let shape = tempogram.shape();
    if shape[0] == 0 || shape[1] == 0 {
        return Array2::<f32>::zeros((shape[0], shape[1]));
    }

    let (n_bins, n_frames) = (shape[0], shape[1]);

    // Compute global tempogram (mean across time)
    let mut global = vec![0.0f32; n_bins];
    for bin in 0..n_bins {
        let mut sum = 0.0;
        for frame in 0..n_frames {
            sum += tempogram[(bin, frame)];
        }
        global[bin] = sum / n_frames as f32;
    }

    // Compute ratio: local / global
    let mut ratio = Array2::<f32>::zeros((n_bins, n_frames));
    for bin in 0..n_bins {
        let g = global[bin];
        for frame in 0..n_frames {
            let local = tempogram[(bin, frame)];
            if g > 1e-10 {
                ratio[(bin, frame)] = local / g;
            } else {
                // If global is zero, set ratio to 0
                ratio[(bin, frame)] = 0.0;
            }
        }
    }

    ratio
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tempogram_shape() {
        let onset_env = vec![0.1, 0.5, 0.3, 0.8, 0.2, 0.6, 0.4, 0.9, 0.1, 0.7];
        let tg = tempogram(&onset_env, 22050, 512, 5);

        assert_eq!(tg.shape()[0], 5);
        assert_eq!(tg.shape()[1], onset_env.len() - 5 + 1);
    }

    #[test]
    fn test_tempogram_values() {
        let onset_env = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let tg = tempogram(&onset_env, 22050, 512, 6);

        assert!(tg.shape()[0] > 0);
        assert!(tg.iter().any(|&v| v > 0.0));
    }

    #[test]
    fn test_tempogram_empty() {
        let onset_env = vec![];
        let tg = tempogram(&onset_env, 22050, 512, 5);

        assert_eq!(tg.shape(), &[0, 0]);
    }

    #[test]
    fn test_tempogram_autocorr_properties() {
        let onset_env = vec![1.0, 0.5, 0.2, 0.8, 0.3, 0.9, 0.1, 0.7];
        let tg = tempogram(&onset_env, 22050, 512, 6);

        for col in 0..tg.shape()[1] {
            assert!(tg[(0, col)] >= 0.0);
        }
    }

    #[test]
    fn test_fourier_tempogram_shape() {
        let onset_env = vec![0.1, 0.5, 0.3, 0.8, 0.2, 0.6, 0.4, 0.9, 0.1, 0.7];
        let ft = fourier_tempogram(&onset_env, 22050, 512, 6);

        assert!(ft.shape()[0] > 0); // Should have frequency bins
        assert_eq!(ft.shape()[1], onset_env.len() - 6 + 1); // Sliding window
    }

    #[test]
    fn test_fourier_tempogram_empty() {
        let onset_env = vec![];
        let ft = fourier_tempogram(&onset_env, 22050, 512, 6);

        assert_eq!(ft.shape(), &[0, 0]);
    }

    #[test]
    fn test_tempo_frequencies() {
        let bpms = tempo_frequencies(10, 22050, 512, 384);

        assert_eq!(bpms.len(), 10);
        assert_eq!(bpms[0], 0.0); // DC component
        assert!(bpms[1] > 0.0); // Positive BPM
        assert!(bpms[5] > bpms[1]); // Increasing
    }

    #[test]
    fn test_tempogram_ratio_shape() {
        let onset_env = vec![0.1, 0.5, 0.3, 0.8, 0.2, 0.6, 0.4, 0.9, 0.1, 0.7];
        let tg = tempogram(&onset_env, 22050, 512, 5);
        let ratio = tempogram_ratio(&tg);

        assert_eq!(ratio.shape(), tg.shape());
    }

    #[test]
    fn test_tempogram_ratio_mean() {
        use approx::assert_relative_eq;

        let onset_env = vec![0.1, 0.5, 0.3, 0.8, 0.2, 0.6, 0.4, 0.9, 0.1, 0.7];
        let tg = tempogram(&onset_env, 22050, 512, 5);
        let ratio = tempogram_ratio(&tg);

        // Mean of ratios across time for each bin should be close to 1.0
        for bin in 0..ratio.shape()[0] {
            let mut sum = 0.0;
            for frame in 0..ratio.shape()[1] {
                sum += ratio[(bin, frame)];
            }
            let mean = sum / ratio.shape()[1] as f32;

            // Skip bins where original tempogram was all zeros
            let mut tg_sum = 0.0;
            for frame in 0..tg.shape()[1] {
                tg_sum += tg[(bin, frame)];
            }

            if tg_sum > 1e-6 {
                assert_relative_eq!(mean, 1.0, epsilon = 0.01);
            }
        }
    }

    #[test]
    fn test_tempogram_ratio_empty() {
        let tg = Array2::<f32>::zeros((0, 0));
        let ratio = tempogram_ratio(&tg);

        assert_eq!(ratio.shape(), &[0, 0]);
    }

    #[test]
    fn test_tempogram_ratio_properties() {
        let onset_env = vec![0.1, 0.5, 0.3, 0.8, 0.2, 0.6, 0.4, 0.9, 0.1, 0.7];
        let tg = tempogram(&onset_env, 22050, 512, 5);
        let ratio = tempogram_ratio(&tg);

        // All values should be non-negative and finite
        for &v in ratio.iter() {
            assert!(v >= 0.0);
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_tempogram_ratio_constant() {
        use approx::assert_relative_eq;

        // Create constant tempogram
        let tg = Array2::from_elem((5, 10), 2.0);
        let ratio = tempogram_ratio(&tg);

        // All ratios should be 1.0 for constant input
        for &v in ratio.iter() {
            assert_relative_eq!(v, 1.0, epsilon = 1e-6);
        }
    }
}
