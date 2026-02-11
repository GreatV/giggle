use crate::spectrum::{StftConfig, griffinlim, stft};
use crate::window;
use ndarray::Array2;

/// Convert frequency in Hz to mel scale.
///
/// The mel scale is a perceptual scale of pitches judged by listeners to be
/// equally spaced from one another.
///
/// # Arguments
/// * `hz` - Frequency in Hz
///
/// # Returns
/// Frequency in mel scale
///
/// # Example
/// ```
/// use giggle::feature::mel::hz_to_mel;
///
/// let mel = hz_to_mel(440.0);  // A4
/// assert!(mel > 6.0 && mel < 7.0);
/// ```
pub fn hz_to_mel(hz: f32) -> f32 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f32).ln() / 27.0;
    if hz < min_log_hz {
        hz / f_sp
    } else {
        min_log_mel + (hz / min_log_hz).ln() / logstep
    }
}

/// Convert frequency from mel scale to Hz.
///
/// This is the inverse of `hz_to_mel`.
///
/// # Arguments
/// * `mel` - Frequency in mel scale
///
/// # Returns
/// Frequency in Hz
///
/// # Example
/// ```
/// use giggle::feature::mel::{hz_to_mel, mel_to_hz};
///
/// let hz = 440.0;
/// let mel = hz_to_mel(hz);
/// assert!((mel_to_hz(mel) - hz).abs() < 0.01);
/// ```
pub fn mel_to_hz(mel: f32) -> f32 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f32).ln() / 27.0;
    if mel < min_log_mel {
        mel * f_sp
    } else {
        min_log_hz * (logstep * (mel - min_log_mel)).exp()
    }
}

/// Generate an array of mel frequencies.
///
/// # Arguments
/// * `n_mels` - Number of mel bands
/// * `fmin` - Minimum frequency in Hz
/// * `fmax` - Maximum frequency in Hz
///
/// # Returns
/// Vector of `n_mels` frequencies evenly spaced in the mel scale
///
/// # Example
/// ```
/// use giggle::feature::mel::mel_frequencies;
///
/// let mels = mel_frequencies(10, 0.0, 22050.0);
/// assert_eq!(mels.len(), 10);
/// assert_eq!(mels[0], 0.0); // fmin
/// ```
pub fn mel_frequencies(n_mels: usize, fmin: f32, fmax: f32) -> Vec<f32> {
    if n_mels == 0 {
        return Vec::new();
    }
    let mel_min = hz_to_mel(fmin.max(0.0));
    let mel_max = hz_to_mel(fmax.max(fmin));
    let step = (mel_max - mel_min) / (n_mels as f32 - 1.0).max(1.0);
    (0..n_mels)
        .map(|i| mel_to_hz(mel_min + step * i as f32))
        .collect()
}

/// Create a mel filterbank matrix.
///
/// This creates a linear transformation matrix to convert FFT frequency bins
/// to mel-frequency bins. Each row is a filter that responds to a different
/// frequency range.
///
/// # Arguments
/// * `sr` - Sample rate in Hz
/// * `n_fft` - FFT window size
/// * `n_mels` - Number of mel bands
/// * `fmin` - Minimum frequency in Hz
/// * `fmax` - Maximum frequency in Hz (will be clipped to Nyquist)
///
/// # Returns
/// Mel filterbank matrix of shape (n_mels, n_fft / 2 + 1)
///
/// # Example
/// ```
/// use giggle::feature::mel::mel_filterbank;
///
/// let fb = mel_filterbank(22050, 2048, 128, 0.0, 11025.0);
/// assert_eq!(fb.shape(), &[128, 1025]); // n_mels x (n_fft/2 + 1)
/// ```
pub fn mel_filterbank(sr: u32, n_fft: usize, n_mels: usize, fmin: f32, fmax: f32) -> Array2<f32> {
    let n_freq = n_fft / 2 + 1;
    let mut fb = Array2::<f32>::zeros((n_mels, n_freq));
    if n_mels == 0 || n_fft == 0 {
        return fb;
    }

    let fmax = fmax.min(sr as f32 / 2.0).max(fmin);
    let mel_points = mel_frequencies(n_mels + 2, fmin, fmax);

    let mut fft_freqs = vec![0.0f32; n_freq];
    for (i, freq) in fft_freqs.iter_mut().enumerate().take(n_freq) {
        *freq = i as f32 * sr as f32 / n_fft as f32;
    }

    let mut fb64 = vec![0.0f64; n_mels * n_freq];
    for m in 0..n_mels {
        let f_m_minus = mel_points[m];
        let f_m = mel_points[m + 1];
        let f_m_plus = mel_points[m + 2];
        let denom_left = (f_m - f_m_minus).max(1e-8) as f64;
        let denom_right = (f_m_plus - f_m).max(1e-8) as f64;

        for (k, freq) in fft_freqs.iter().enumerate() {
            let freq = *freq;
            let lower = (freq - f_m_minus) as f64 / denom_left;
            let upper = (f_m_plus - freq) as f64 / denom_right;
            let w = lower.min(upper).max(0.0);
            fb64[m * n_freq + k] = w;
        }

        let enorm = 2.0 / (f_m_plus - f_m_minus).max(1e-8) as f64;
        for k in 0..n_freq {
            fb64[m * n_freq + k] *= enorm;
        }
    }

    for m in 0..n_mels {
        for k in 0..n_freq {
            fb[(m, k)] = fb64[m * n_freq + k] as f32;
        }
    }

    fb
}

/// Compute a mel-scaled spectrogram.
///
/// This function computes a mel-frequency spectrogram, which represents
/// the short-term power spectrum of a sound on a mel scale.
///
/// # Arguments
/// * `y` - Input audio signal (mono)
/// * `sr` - Sample rate in Hz
/// * `n_fft` - FFT window size
/// * `hop_length` - Number of samples between frames
/// * `n_mels` - Number of mel bands
///
/// # Returns
/// Mel spectrogram matrix of shape (n_mels, n_frames)
///
/// # Example
/// ```
/// use giggle::feature::mel::melspectrogram;
///
/// let signal = vec![0.1f32; 22050];
/// let mel = melspectrogram(&signal, 22050, 2048, 512, 128).unwrap();
/// assert_eq!(mel.shape(), &[128, 44]); // 128 mels, 44 frames
/// ```
pub fn melspectrogram(
    y: &[f32],
    sr: u32,
    n_fft: usize,
    hop_length: usize,
    n_mels: usize,
) -> crate::Result<Array2<f32>> {
    let mut cfg = StftConfig::default();
    cfg.n_fft = n_fft;
    cfg.win_length = n_fft;
    cfg.hop_length = hop_length;
    cfg.window = window::hann(cfg.win_length);

    let stft_matrix = stft(y, &cfg)?;
    let n_freq = stft_matrix.shape()[0];
    let n_frames = stft_matrix.shape()[1];

    let fb = mel_filterbank(sr, n_fft, n_mels, 0.0, sr as f32 / 2.0);
    let mut mel_spec = Array2::<f32>::zeros((n_mels, n_frames));

    for t in 0..n_frames {
        for f in 0..n_freq {
            let v = stft_matrix[(f, t)];
            let power = (v.re * v.re + v.im * v.im) as f64;
            for m in 0..n_mels {
                let w = fb[(m, f)] as f64;
                if w > 0.0 {
                    mel_spec[(m, t)] += (w * power) as f32;
                }
            }
        }
    }

    Ok(mel_spec)
}

/// Convert mel spectrogram to linear STFT magnitude spectrogram.
///
/// This function reconstructs an approximate linear-frequency magnitude
/// spectrogram from a mel spectrogram using the transpose of the mel
/// filterbank. Note that this is an approximate inverse since the mel
/// transformation is lossy (information is compressed).
///
/// # Arguments
/// * `mel_spec` - Mel spectrogram (n_mels x n_frames)
/// * `sr` - Sample rate
/// * `n_fft` - FFT size (determines output frequency resolution)
/// * `fmin` - Minimum frequency for mel filterbank
/// * `fmax` - Maximum frequency for mel filterbank
///
/// # Returns
/// Magnitude STFT matrix (n_freq x n_frames) where n_freq = n_fft / 2 + 1
///
/// # Example
/// ```
/// use giggle::feature::mel::{melspectrogram, mel_to_stft};
/// use ndarray::Array2;
///
/// let signal = vec![0.1; 22050];
/// let mel_spec = melspectrogram(&signal, 22050, 2048, 512, 128).unwrap();
/// let stft_mag = mel_to_stft(&mel_spec, 22050, 2048, 0.0, 11025.0);
/// assert_eq!(stft_mag.shape()[0], 2048 / 2 + 1);
/// ```
pub fn mel_to_stft(
    mel_spec: &Array2<f32>,
    sr: u32,
    n_fft: usize,
    fmin: f32,
    fmax: f32,
) -> Array2<f32> {
    let shape = mel_spec.shape();
    let (n_mels, n_frames) = (shape[0], shape[1]);

    if n_mels == 0 || n_frames == 0 || n_fft == 0 {
        let n_freq = n_fft / 2 + 1;
        return Array2::<f32>::zeros((n_freq, n_frames));
    }

    // Generate mel filterbank (n_mels x n_freq)
    let fb = mel_filterbank(sr, n_fft, n_mels, fmin, fmax);
    let n_freq = n_fft / 2 + 1;

    // Compute transpose reconstruction: stft_mag ≈ fb^T @ mel_spec
    let mut stft_mag = Array2::<f32>::zeros((n_freq, n_frames));

    for t in 0..n_frames {
        for f in 0..n_freq {
            let mut sum = 0.0f32;
            for m in 0..n_mels {
                sum += fb[(m, f)] * mel_spec[(m, t)];
            }
            stft_mag[(f, t)] = sum;
        }
    }

    stft_mag
}

/// Convert mel spectrogram to audio waveform.
///
/// This function reconstructs an audio signal from a mel spectrogram using:
/// 1. mel_to_stft to reconstruct magnitude spectrogram
/// 2. Griffin-Lim algorithm for phase reconstruction
/// 3. ISTFT to convert to time domain
///
/// # Arguments
/// * `mel_spec` - Mel spectrogram (n_mels x n_frames)
/// * `sr` - Sample rate
/// * `n_fft` - FFT size
/// * `hop_length` - Hop length for STFT/ISTFT
/// * `fmin` - Minimum frequency for mel filterbank
/// * `fmax` - Maximum frequency for mel filterbank
/// * `n_iter` - Number of Griffin-Lim iterations (default: 32)
/// * `length` - Optional output length to match original signal
///
/// # Returns
/// Reconstructed audio signal
///
/// # Example
/// ```
/// use giggle::feature::mel::{melspectrogram, mel_to_audio};
///
/// let signal = vec![0.1; 22050];
/// let mel_spec = melspectrogram(&signal, 22050, 2048, 512, 128).unwrap();
/// let reconstructed = mel_to_audio(&mel_spec, 22050, 2048, 512, 0.0, 11025.0, 16, None).unwrap();
/// assert!(reconstructed.len() > 0);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn mel_to_audio(
    mel_spec: &Array2<f32>,
    sr: u32,
    n_fft: usize,
    hop_length: usize,
    fmin: f32,
    fmax: f32,
    n_iter: usize,
    length: Option<usize>,
) -> crate::Result<Vec<f32>> {
    // Convert mel spectrogram to linear STFT magnitude
    let stft_mag = mel_to_stft(mel_spec, sr, n_fft, fmin, fmax);

    if stft_mag.shape()[0] == 0 || stft_mag.shape()[1] == 0 {
        return Ok(Vec::new());
    }

    // Reconstruct phase using Griffin-Lim
    let mut config = StftConfig::default();
    config.n_fft = n_fft;
    config.win_length = n_fft;
    config.hop_length = hop_length;
    config.window = window::hann(config.win_length);
    config.center = true;

    // Use Griffin-Lim with momentum for faster convergence

    griffinlim(&stft_mag, &config, n_iter, length, 0.99)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_mel_to_stft_shape() {
        let mel_spec = Array2::<f32>::zeros((128, 100));
        let stft_mag = mel_to_stft(&mel_spec, 22050, 2048, 0.0, 11025.0);

        assert_eq!(stft_mag.shape()[0], 2048 / 2 + 1); // n_freq
        assert_eq!(stft_mag.shape()[1], 100); // n_frames
    }

    #[test]
    fn test_mel_to_stft_empty() {
        let mel_spec = Array2::<f32>::zeros((0, 0));
        let stft_mag = mel_to_stft(&mel_spec, 22050, 2048, 0.0, 11025.0);

        assert_eq!(stft_mag.shape()[0], 2048 / 2 + 1);
        assert_eq!(stft_mag.shape()[1], 0);
    }

    #[test]
    fn test_mel_to_stft_roundtrip() {
        use crate::io;

        // Generate a test signal
        let signal = io::tone(440.0, 22050, 0.5);

        // Forward: signal -> mel spectrogram
        let mel_spec = melspectrogram(&signal, 22050, 2048, 512, 128).unwrap();

        // Backward: mel -> STFT magnitude
        let stft_reconstructed = mel_to_stft(&mel_spec, 22050, 2048, 0.0, 11025.0);

        // Check shape
        assert_eq!(stft_reconstructed.shape()[0], 2048 / 2 + 1);
        assert_eq!(stft_reconstructed.shape()[1], mel_spec.shape()[1]);

        // Check that reconstruction is non-trivial (has energy)
        let energy: f32 = stft_reconstructed.iter().map(|&x| x * x).sum();
        assert!(energy > 0.0, "Reconstructed STFT should have energy");
    }

    #[test]
    fn test_mel_to_stft_properties() {
        let mut mel_spec = Array2::<f32>::zeros((128, 50));

        // Set some non-zero values
        for m in 0..128 {
            for t in 0..50 {
                mel_spec[(m, t)] = (m as f32 + 1.0) * 0.1;
            }
        }

        let stft_mag = mel_to_stft(&mel_spec, 22050, 2048, 0.0, 11025.0);

        // All values should be non-negative and finite
        for &v in stft_mag.iter() {
            assert!(v >= 0.0);
            assert!(v.is_finite());
        }

        // Should have some non-zero values
        let has_nonzero = stft_mag.iter().any(|&v| v > 0.0);
        assert!(has_nonzero, "STFT should have non-zero values");
    }

    #[test]
    fn test_mel_to_stft_zero_input() {
        let mel_spec = Array2::<f32>::zeros((128, 50));
        let stft_mag = mel_to_stft(&mel_spec, 22050, 2048, 0.0, 11025.0);

        // Zero input should give zero output
        for &v in stft_mag.iter() {
            assert_relative_eq!(v, 0.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_mel_to_audio_basic() {
        use crate::io;

        // Generate test signal
        let signal = io::tone(440.0, 22050, 0.5);
        let signal_len = signal.len();

        // Convert to mel spectrogram
        let mel_spec = melspectrogram(&signal, 22050, 2048, 512, 128).unwrap();

        // Convert back to audio
        let reconstructed = mel_to_audio(
            &mel_spec,
            22050,
            2048,
            512,
            0.0,
            11025.0,
            8,
            Some(signal_len),
        )
        .unwrap();

        // Check length
        assert_eq!(reconstructed.len(), signal_len);

        // Check energy
        let energy: f32 = reconstructed.iter().map(|&x| x * x).sum();
        assert!(energy > 0.1, "Reconstructed signal should have energy");
    }

    #[test]
    fn test_mel_to_audio_empty() {
        let mel_spec = Array2::<f32>::zeros((0, 0));
        let audio = mel_to_audio(&mel_spec, 22050, 2048, 512, 0.0, 11025.0, 8, None).unwrap();

        assert_eq!(audio.len(), 0);
    }

    #[test]
    fn test_mel_to_audio_properties() {
        use crate::io;

        let signal = io::tone(880.0, 22050, 0.3);
        let mel_spec = melspectrogram(&signal, 22050, 2048, 512, 80).unwrap();

        let audio = mel_to_audio(&mel_spec, 22050, 2048, 512, 0.0, 11025.0, 16, None).unwrap();

        // Audio should have reasonable length
        assert!(!audio.is_empty());

        // All values should be finite
        for &v in &audio {
            assert!(v.is_finite());
        }

        // Should have some variation (not all zeros or constant)
        let min_val = audio.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = audio.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(max_val > min_val + 0.01);
    }

    #[test]
    fn test_mel_to_audio_with_length() {
        use crate::io;

        let signal = io::tone(440.0, 22050, 0.5);
        let target_len = 10000;

        let mel_spec = melspectrogram(&signal, 22050, 2048, 512, 128).unwrap();
        let audio = mel_to_audio(
            &mel_spec,
            22050,
            2048,
            512,
            0.0,
            11025.0,
            8,
            Some(target_len),
        )
        .unwrap();

        assert_eq!(audio.len(), target_len);
    }
}
