use crate::spectrum::{StftConfig, stft};
use crate::window;
use ndarray::Array2;

pub fn onset_strength(y: &[f32], n_fft: usize, hop_length: usize) -> crate::Result<Vec<f32>> {
    if y.is_empty() || n_fft == 0 {
        return Ok(Vec::new());
    }
    let mut cfg = StftConfig::default();
    cfg.n_fft = n_fft;
    cfg.win_length = n_fft;
    cfg.hop_length = hop_length.max(1);
    cfg.window = window::hann(cfg.win_length);

    let stft_matrix = stft(y, &cfg)?;
    let n_freq = stft_matrix.shape()[0];
    let n_frames = stft_matrix.shape()[1];
    if n_freq == 0 || n_frames == 0 {
        return Ok(Vec::new());
    }

    let mut env = vec![0.0f32; n_frames];
    let mut prev_mag = vec![0.0f32; n_freq];

    for t in 0..n_frames {
        let mut sum = 0.0f32;
        for f in 0..n_freq {
            let v = stft_matrix[(f, t)];
            let mag = (v.re * v.re + v.im * v.im).sqrt();
            let diff = (mag - prev_mag[f]).max(0.0);
            sum += diff;
            prev_mag[f] = mag;
        }
        env[t] = sum;
    }

    Ok(env)
}

/// Compute onset strength across multiple frequency bands.
///
/// This function computes onset strength separately for different frequency
/// bands, allowing for more detailed analysis of rhythmic content across
/// the spectrum. This is useful for separating different types of onsets
/// (e.g., harmonic vs percussive).
///
/// # Arguments
/// * `y` - Input audio signal
/// * `sr` - Sample rate
/// * `n_fft` - FFT size
/// * `hop_length` - Hop length
/// * `bands` - Frequency bands as (low_freq, high_freq) pairs in Hz
///
/// # Returns
/// Onset strength matrix (n_bands x n_frames)
///
/// # Example
/// ```
/// use giggle::onset::strength::onset_strength_multi;
///
/// let signal = vec![0.0; 22050]; // 1 second of silence
/// let sr = 22050;
/// let bands = vec![(0.0, 200.0), (200.0, 2000.0), (2000.0, 11025.0)];
/// let onset_env = onset_strength_multi(&signal, sr, 2048, 512, &bands).unwrap();
/// assert_eq!(onset_env.shape()[0], 3); // 3 bands
/// ```
pub fn onset_strength_multi(
    y: &[f32],
    sr: u32,
    n_fft: usize,
    hop_length: usize,
    bands: &[(f32, f32)],
) -> crate::Result<Array2<f32>> {
    if y.is_empty() || n_fft == 0 || bands.is_empty() {
        return Ok(Array2::<f32>::zeros((bands.len(), 0)));
    }

    let mut cfg = StftConfig::default();
    cfg.n_fft = n_fft;
    cfg.win_length = n_fft;
    cfg.hop_length = hop_length.max(1);
    cfg.window = window::hann(cfg.win_length);

    let stft_matrix = stft(y, &cfg)?;
    let n_freq = stft_matrix.shape()[0];
    let n_frames = stft_matrix.shape()[1];

    if n_freq == 0 || n_frames == 0 {
        return Ok(Array2::<f32>::zeros((bands.len(), n_frames)));
    }

    // Convert frequency bands to bin indices
    let freq_per_bin = sr as f32 / n_fft as f32;
    let band_bins: Vec<(usize, usize)> = bands
        .iter()
        .map(|(low, high)| {
            let low_bin = ((low / freq_per_bin).round() as usize).min(n_freq);
            let high_bin = ((high / freq_per_bin).round() as usize).min(n_freq);
            (low_bin, high_bin)
        })
        .collect();

    let mut onset_env = Array2::<f32>::zeros((bands.len(), n_frames));
    let mut prev_mag = vec![vec![0.0f32; n_freq]; bands.len()];

    for (band_idx, &(low_bin, high_bin)) in band_bins.iter().enumerate() {
        for t in 0..n_frames {
            let mut sum = 0.0f32;
            for f in low_bin..high_bin {
                let v = stft_matrix[(f, t)];
                let mag = (v.re * v.re + v.im * v.im).sqrt();
                let diff = (mag - prev_mag[band_idx][f]).max(0.0);
                sum += diff;
                prev_mag[band_idx][f] = mag;
            }
            onset_env[(band_idx, t)] = sum;
        }
    }

    Ok(onset_env)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onset_strength_multi_shape() {
        let signal = vec![0.1; 22050]; // 1 second
        let sr = 22050;
        let bands = vec![(0.0, 200.0), (200.0, 2000.0), (2000.0, 11025.0)];
        let onset_env = onset_strength_multi(&signal, sr, 2048, 512, &bands).unwrap();

        assert_eq!(onset_env.shape()[0], 3); // 3 bands
        assert!(onset_env.shape()[1] > 0); // Has frames
    }

    #[test]
    fn test_onset_strength_multi_empty() {
        let signal = vec![];
        let sr = 22050;
        let bands = vec![(0.0, 1000.0)];
        let onset_env = onset_strength_multi(&signal, sr, 2048, 512, &bands).unwrap();

        assert_eq!(onset_env.shape()[0], 1);
        assert_eq!(onset_env.shape()[1], 0);
    }

    #[test]
    fn test_onset_strength_multi_single_band_matches_full() {
        use crate::io;

        // Generate a simple signal
        let signal = io::tone(440.0, 22050, 0.5);
        let sr = 22050;

        // Full spectrum band
        let bands = vec![(0.0, 11025.0)];
        let onset_multi = onset_strength_multi(&signal, sr, 2048, 512, &bands).unwrap();

        // Regular onset strength
        let onset_single = onset_strength(&signal, 2048, 512).unwrap();

        // Should have same number of frames
        assert_eq!(onset_multi.shape()[1], onset_single.len());

        // Values should be similar (may not be exactly equal due to rounding)
        for t in 0..onset_single.len() {
            let ratio = if onset_single[t] > 1e-6 {
                onset_multi[(0, t)] / onset_single[t]
            } else {
                1.0
            };
            assert!(
                ratio > 0.9 && ratio < 1.1,
                "Mismatch at frame {}: multi={}, single={}",
                t,
                onset_multi[(0, t)],
                onset_single[t]
            );
        }
    }

    #[test]
    fn test_onset_strength_multi_properties() {
        let signal = vec![0.1, 0.5, 0.3, 0.8, 0.2, 0.6, 0.4, 0.9];
        let signal: Vec<f32> = signal.into_iter().cycle().take(22050).collect();
        let sr = 22050;
        let bands = vec![(0.0, 500.0), (500.0, 2000.0)];
        let onset_env = onset_strength_multi(&signal, sr, 2048, 512, &bands).unwrap();

        // All values should be non-negative
        for &v in onset_env.iter() {
            assert!(v >= 0.0);
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_onset_strength_multi_no_bands() {
        let signal = vec![0.1; 1000];
        let sr = 22050;
        let bands = vec![];
        let onset_env = onset_strength_multi(&signal, sr, 2048, 512, &bands).unwrap();

        assert_eq!(onset_env.shape()[0], 0);
    }

    #[test]
    fn test_onset_strength_multi_band_independence() {
        use crate::io;

        // Generate signal with two distinct frequency components
        let sr = 22050;
        let low_freq = io::tone(100.0, sr, 0.5); // Low frequency
        let high_freq = io::tone(5000.0, sr, 0.5); // High frequency

        // Test with two non-overlapping bands
        let bands = vec![(0.0, 500.0), (4000.0, 6000.0)];

        // Low frequency signal should have more energy in first band
        let onset_low = onset_strength_multi(&low_freq, sr, 2048, 512, &bands).unwrap();
        let low_band_0: f32 = (0..onset_low.shape()[1]).map(|t| onset_low[(0, t)]).sum();
        let low_band_1: f32 = (0..onset_low.shape()[1]).map(|t| onset_low[(1, t)]).sum();
        assert!(
            low_band_0 > low_band_1 * 2.0,
            "Low freq should dominate band 0"
        );

        // High frequency signal should have more energy in second band
        let onset_high = onset_strength_multi(&high_freq, sr, 2048, 512, &bands).unwrap();
        let high_band_0: f32 = (0..onset_high.shape()[1]).map(|t| onset_high[(0, t)]).sum();
        let high_band_1: f32 = (0..onset_high.shape()[1]).map(|t| onset_high[(1, t)]).sum();
        assert!(
            high_band_1 > high_band_0 * 2.0,
            "High freq should dominate band 1"
        );
    }
}
