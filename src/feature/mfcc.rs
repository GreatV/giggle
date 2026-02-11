use crate::feature::mel::{mel_to_audio, melspectrogram};
use ndarray::Array2;

/// Configuration for MFCC computation.
///
/// This struct provides a builder pattern for configuring MFCC parameters
/// instead of using many positional arguments.
///
/// # Example
/// ```
/// use giggle::feature::mfcc::MfccConfig;
///
/// let config = MfccConfig::new(22050)
///     .with_n_mfcc(13)
///     .with_n_mels(128);
/// ```
#[derive(Debug, Clone)]
pub struct MfccConfig {
    /// Sample rate
    pub sr: u32,
    /// FFT window size
    pub n_fft: usize,
    /// Number of samples between frames
    pub hop_length: usize,
    /// Number of MFCC coefficients to return
    pub n_mfcc: usize,
    /// Number of mel bands
    pub n_mels: usize,
}

impl MfccConfig {
    /// Create a new MFCC configuration with defaults.
    ///
    /// # Arguments
    /// * `sr` - Sample rate
    pub fn new(sr: u32) -> Self {
        Self {
            sr,
            n_fft: 2048,
            hop_length: 512,
            n_mfcc: 20,
            n_mels: 128,
        }
    }

    /// Set the FFT window size.
    pub fn with_n_fft(mut self, n_fft: usize) -> Self {
        self.n_fft = n_fft;
        self
    }

    /// Set the hop length.
    pub fn with_hop_length(mut self, hop_length: usize) -> Self {
        self.hop_length = hop_length;
        self
    }

    /// Set the number of MFCC coefficients.
    pub fn with_n_mfcc(mut self, n_mfcc: usize) -> Self {
        self.n_mfcc = n_mfcc;
        self
    }

    /// Set the number of mel bands.
    pub fn with_n_mels(mut self, n_mels: usize) -> Self {
        self.n_mels = n_mels;
        self
    }

    /// Compute MFCC with this configuration.
    ///
    /// # Arguments
    /// * `y` - Input audio signal (mono)
    ///
    /// # Returns
    /// MFCC matrix of shape (n_mfcc, n_frames)
    pub fn compute(&self, y: &[f32]) -> crate::Result<Array2<f32>> {
        mfcc(
            y,
            self.sr,
            self.n_mfcc,
            self.n_fft,
            self.hop_length,
            self.n_mels,
        )
    }
}

impl Default for MfccConfig {
    fn default() -> Self {
        Self {
            sr: 22050,
            n_fft: 2048,
            hop_length: 512,
            n_mfcc: 20,
            n_mels: 128,
        }
    }
}

/// Configuration for MFCC to audio reconstruction.
///
/// This struct provides a builder pattern for configuring MFCC to audio
/// reconstruction parameters.
///
/// # Example
/// ```
/// use giggle::feature::mfcc::MfccToAudioConfig;
///
/// let config = MfccToAudioConfig::new(22050)
///     .with_n_iter(16);
/// ```
#[derive(Debug, Clone)]
pub struct MfccToAudioConfig {
    /// Sample rate
    pub sr: u32,
    /// FFT window size
    pub n_fft: usize,
    /// Hop length for STFT/ISTFT
    pub hop_length: usize,
    /// Minimum frequency for mel filterbank
    pub fmin: f32,
    /// Maximum frequency for mel filterbank
    pub fmax: f32,
    /// Number of Griffin-Lim iterations
    pub n_iter: usize,
}

impl MfccToAudioConfig {
    /// Create a new MFCC to audio configuration with defaults.
    ///
    /// # Arguments
    /// * `sr` - Sample rate
    pub fn new(sr: u32) -> Self {
        Self {
            sr,
            n_fft: 2048,
            hop_length: 512,
            fmin: 0.0,
            fmax: sr as f32 / 2.0,
            n_iter: 32,
        }
    }

    /// Set the FFT window size.
    pub fn with_n_fft(mut self, n_fft: usize) -> Self {
        self.n_fft = n_fft;
        self
    }

    /// Set the hop length.
    pub fn with_hop_length(mut self, hop_length: usize) -> Self {
        self.hop_length = hop_length;
        self
    }

    /// Set the minimum frequency.
    pub fn with_fmin(mut self, fmin: f32) -> Self {
        self.fmin = fmin;
        self
    }

    /// Set the maximum frequency.
    pub fn with_fmax(mut self, fmax: f32) -> Self {
        self.fmax = fmax;
        self
    }

    /// Set the number of Griffin-Lim iterations.
    pub fn with_n_iter(mut self, n_iter: usize) -> Self {
        self.n_iter = n_iter;
        self
    }

    /// Reconstruct audio from MFCC with this configuration.
    ///
    /// # Arguments
    /// * `mfcc` - MFCC coefficients (n_mfcc x n_frames)
    /// * `n_mels` - Number of mel bands to use for reconstruction
    /// * `length` - Optional output length to match original signal
    ///
    /// # Returns
    /// Reconstructed audio signal
    pub fn compute(
        &self,
        mfcc: &Array2<f32>,
        n_mels: usize,
        length: Option<usize>,
    ) -> crate::Result<Vec<f32>> {
        mfcc_to_audio(
            mfcc,
            n_mels,
            self.sr,
            self.n_fft,
            self.hop_length,
            self.fmin,
            self.fmax,
            self.n_iter,
            length,
        )
    }
}

impl Default for MfccToAudioConfig {
    fn default() -> Self {
        Self {
            sr: 22050,
            n_fft: 2048,
            hop_length: 512,
            fmin: 0.0,
            fmax: 11025.0,
            n_iter: 32,
        }
    }
}

/// Compute the Discrete Cosine Transform (DCT) Type-II.
///
/// DCT-II is the most common form of the DCT and is widely used in signal
/// processing, particularly for MFCC computation.
///
/// # Arguments
/// * `x` - Input signal
/// * `n_out` - Number of output coefficients
///
/// # Returns
/// DCT coefficients
///
/// # Example
/// ```
/// use giggle::feature::mfcc::dct_type_ii;
///
/// let signal = vec![1.0f32, 2.0, 3.0, 4.0];
/// let dct = dct_type_ii(&signal, 4);
/// assert_eq!(dct.len(), 4);
/// ```
pub fn dct_type_ii(x: &[f32], n_out: usize) -> Vec<f32> {
    let n = x.len() as f32;
    if n == 0.0 || n_out == 0 {
        return Vec::new();
    }
    let mut out = vec![0.0f32; n_out];
    for (k, out_val) in out.iter_mut().enumerate().take(n_out) {
        let mut sum = 0.0f32;
        for (i, v) in x.iter().enumerate() {
            let angle = std::f32::consts::PI / n * (i as f32 + 0.5) * k as f32;
            sum += v * angle.cos();
        }
        let scale = if k == 0 {
            (1.0 / n).sqrt()
        } else {
            (2.0 / n).sqrt()
        };
        *out_val = sum * scale;
    }
    out
}

fn power_to_db(x: f32, ref_value: f32, _top_db: f32) -> f32 {
    let amin = 1e-10f32;
    let ref_db = 10.0 * ref_value.max(amin).log10();

    10.0 * x.max(amin).log10() - ref_db
}

/// Compute Mel-Frequency Cepstral Coefficients (MFCCs).
///
/// MFCCs are commonly used as features in speech and audio processing.
/// They represent the short-term power spectrum of sound based on a
/// mel-scale warped frequency axis.
///
/// # Arguments
/// * `y` - Input audio signal (mono)
/// * `sr` - Sample rate in Hz
/// * `n_mfcc` - Number of MFCC coefficients to return
/// * `n_fft` - FFT window size
/// * `hop_length` - Number of samples between frames
/// * `n_mels` - Number of mel bands to use
///
/// # Returns
/// MFCC matrix of shape (n_mfcc, n_frames)
///
/// # Example
/// ```
/// use giggle::feature::mfcc::mfcc;
///
/// let signal = vec![0.1f32; 22050];
/// let coeffs = mfcc(&signal, 22050, 13, 2048, 512, 128).unwrap();
/// assert_eq!(coeffs.shape(), &[13, 44]);
/// ```
pub fn mfcc(
    y: &[f32],
    sr: u32,
    n_mfcc: usize,
    n_fft: usize,
    hop_length: usize,
    n_mels: usize,
) -> crate::Result<Array2<f32>> {
    let mel = melspectrogram(y, sr, n_fft, hop_length, n_mels)?;
    let n_frames = mel.shape()[1];
    let mut out = Array2::<f32>::zeros((n_mfcc, n_frames));

    for t in 0..n_frames {
        let mut log_mel = vec![0.0f32; n_mels];
        let mut max_db = f32::NEG_INFINITY;
        for m in 0..n_mels {
            let db = power_to_db(mel[(m, t)], 1.0, 80.0);
            log_mel[m] = db;
            if db > max_db {
                max_db = db;
            }
        }
        let floor = max_db - 80.0;
        for v in &mut log_mel {
            if *v < floor {
                *v = floor;
            }
        }
        let coeffs = dct_type_ii(&log_mel, n_mfcc);
        for k in 0..n_mfcc {
            out[(k, t)] = coeffs[k];
        }
    }

    Ok(out)
}

/// DCT Type-III (inverse DCT).
///
/// This is the inverse of DCT-II and is used to reconstruct mel spectrogram
/// from MFCC coefficients.
///
/// # Arguments
/// * `x` - Input DCT coefficients
/// * `n_out` - Number of output samples
///
/// # Returns
/// Reconstructed signal
pub fn dct_type_iii(x: &[f32], n_out: usize) -> Vec<f32> {
    let n = n_out as f32;
    if x.is_empty() || n_out == 0 {
        return Vec::new();
    }

    let mut out = vec![0.0f32; n_out];
    for (i, out_val) in out.iter_mut().enumerate().take(n_out) {
        let mut sum = 0.0f32;
        for (k, &coeff) in x.iter().enumerate() {
            let angle = std::f32::consts::PI / n * (i as f32 + 0.5) * k as f32;
            let scale = if k == 0 {
                (1.0 / n).sqrt()
            } else {
                (2.0 / n).sqrt()
            };
            sum += coeff * scale * angle.cos();
        }
        *out_val = sum;
    }
    out
}

/// Convert MFCC coefficients to mel spectrogram.
///
/// This function inverts the MFCC transformation by:
/// 1. Applying inverse DCT (DCT-III) to get log mel spectrogram
/// 2. Converting from dB scale to power
///
/// # Arguments
/// * `mfcc` - MFCC coefficients (n_mfcc x n_frames)
/// * `n_mels` - Number of mel bands in output spectrogram
///
/// # Returns
/// Reconstructed mel spectrogram (n_mels x n_frames)
///
/// # Example
/// ```
/// use giggle::feature::mfcc::{mfcc, mfcc_to_mel};
///
/// let signal = vec![0.1; 22050];
/// let mfcc_coeffs = mfcc(&signal, 22050, 13, 2048, 512, 128).unwrap();
/// let mel_spec = mfcc_to_mel(&mfcc_coeffs, 128);
/// assert_eq!(mel_spec.shape()[0], 128);
/// ```
pub fn mfcc_to_mel(mfcc: &Array2<f32>, n_mels: usize) -> Array2<f32> {
    let shape = mfcc.shape();
    let (n_mfcc, n_frames) = (shape[0], shape[1]);

    if n_mfcc == 0 || n_frames == 0 || n_mels == 0 {
        return Array2::<f32>::zeros((n_mels, n_frames));
    }

    let mut mel_spec = Array2::<f32>::zeros((n_mels, n_frames));

    for t in 0..n_frames {
        // Extract MFCC coefficients for this frame
        let coeffs: Vec<f32> = (0..n_mfcc).map(|k| mfcc[(k, t)]).collect();

        // Apply inverse DCT to get log mel spectrogram
        let log_mel = dct_type_iii(&coeffs, n_mels);

        // Convert from dB to power: power = 10^(db / 10)
        for m in 0..n_mels {
            let db = log_mel[m];
            let power = 10.0f32.powf(db / 10.0);
            mel_spec[(m, t)] = power;
        }
    }

    mel_spec
}

/// Convert MFCC coefficients to audio waveform.
///
/// This function reconstructs an audio signal from MFCC coefficients by:
/// 1. Converting MFCC to mel spectrogram (mfcc_to_mel)
/// 2. Converting mel spectrogram to audio (mel_to_audio with Griffin-Lim)
///
/// # Arguments
/// * `mfcc` - MFCC coefficients (n_mfcc x n_frames)
/// * `n_mels` - Number of mel bands to use for reconstruction
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
/// use giggle::feature::mfcc::{mfcc, mfcc_to_audio};
///
/// let signal = vec![0.1; 22050];
/// let mfcc_coeffs = mfcc(&signal, 22050, 13, 2048, 512, 128).unwrap();
/// let reconstructed = mfcc_to_audio(&mfcc_coeffs, 128, 22050, 2048, 512, 0.0, 11025.0, 16, None).unwrap();
/// assert!(reconstructed.len() > 0);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn mfcc_to_audio(
    mfcc: &Array2<f32>,
    n_mels: usize,
    sr: u32,
    n_fft: usize,
    hop_length: usize,
    fmin: f32,
    fmax: f32,
    n_iter: usize,
    length: Option<usize>,
) -> crate::Result<Vec<f32>> {
    // Convert MFCC to mel spectrogram
    let mel_spec = mfcc_to_mel(mfcc, n_mels);

    if mel_spec.shape()[0] == 0 || mel_spec.shape()[1] == 0 {
        return Ok(Vec::new());
    }

    // Convert mel spectrogram to audio
    mel_to_audio(&mel_spec, sr, n_fft, hop_length, fmin, fmax, n_iter, length)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dct_type_iii_basic() {
        // DCT-III should invert DCT-II
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let dct2 = dct_type_ii(&x, x.len());
        let reconstructed = dct_type_iii(&dct2, x.len());

        for i in 0..x.len() {
            assert!(
                (x[i] - reconstructed[i]).abs() < 1e-5,
                "DCT roundtrip failed at index {}",
                i
            );
        }
    }

    #[test]
    fn test_mfcc_to_mel_shape() {
        let mfcc = Array2::<f32>::zeros((13, 100));
        let mel_spec = mfcc_to_mel(&mfcc, 128);

        assert_eq!(mel_spec.shape()[0], 128); // n_mels
        assert_eq!(mel_spec.shape()[1], 100); // n_frames
    }

    #[test]
    fn test_mfcc_to_mel_empty() {
        let mfcc = Array2::<f32>::zeros((0, 0));
        let mel_spec = mfcc_to_mel(&mfcc, 128);

        assert_eq!(mel_spec.shape()[0], 128);
        assert_eq!(mel_spec.shape()[1], 0);
    }

    #[test]
    fn test_mfcc_to_mel_properties() {
        use crate::io;

        // Generate test signal and compute MFCC
        let signal = io::tone(440.0, 22050, 0.5);
        let mfcc_coeffs = mfcc(&signal, 22050, 13, 2048, 512, 128).unwrap();

        // Convert back to mel
        let mel_reconstructed = mfcc_to_mel(&mfcc_coeffs, 128);

        // Check shape
        assert_eq!(mel_reconstructed.shape()[0], 128);
        assert_eq!(mel_reconstructed.shape()[1], mfcc_coeffs.shape()[1]);

        // All values should be non-negative and finite
        for &v in mel_reconstructed.iter() {
            assert!(v >= 0.0);
            assert!(v.is_finite());
        }

        // Should have some non-zero values
        let has_nonzero = mel_reconstructed.iter().any(|&v| v > 0.0);
        assert!(has_nonzero, "Reconstructed mel should have non-zero values");
    }

    #[test]
    fn test_mfcc_to_mel_roundtrip_lossy() {
        use crate::io;

        let signal = io::tone(880.0, 22050, 0.3);

        // Forward: signal -> mel -> mfcc (with only 13 coefficients)
        let mel_original =
            crate::feature::mel::melspectrogram(&signal, 22050, 2048, 512, 128).unwrap();
        let mfcc_coeffs = mfcc(&signal, 22050, 13, 2048, 512, 128).unwrap();

        // Backward: mfcc -> mel
        let mel_reconstructed = mfcc_to_mel(&mfcc_coeffs, 128);

        // Shapes should match
        assert_eq!(mel_reconstructed.shape(), mel_original.shape());

        // Reconstruction is lossy (we only used 13 MFCC coefficients from 128 mel bands)
        // But should still capture general structure
        let orig_energy: f32 = mel_original.iter().map(|&x| x * x).sum();
        let recon_energy: f32 = mel_reconstructed.iter().map(|&x| x * x).sum();

        assert!(orig_energy > 0.0);
        assert!(recon_energy > 0.0);

        // Energies should be in same ballpark (within 5 orders of magnitude)
        // Note: MFCC is very lossy (13 coeffs from 128 mels), so we expect large differences
        let ratio = (orig_energy / recon_energy).log10().abs();
        assert!(ratio < 5.0, "Energy ratio too different: 10^{}", ratio);
    }

    #[test]
    fn test_mfcc_to_audio_basic() {
        use crate::io;

        // Generate test signal
        let signal = io::tone(440.0, 22050, 0.5);
        let signal_len = signal.len();

        // Convert to MFCC
        let mfcc_coeffs = mfcc(&signal, 22050, 13, 2048, 512, 128).unwrap();

        // Convert back to audio
        let reconstructed = mfcc_to_audio(
            &mfcc_coeffs,
            128,
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
    fn test_mfcc_to_audio_empty() {
        let mfcc = Array2::<f32>::zeros((0, 0));
        let audio = mfcc_to_audio(&mfcc, 128, 22050, 2048, 512, 0.0, 11025.0, 8, None).unwrap();

        assert_eq!(audio.len(), 0);
    }

    #[test]
    fn test_mfcc_to_audio_properties() {
        use crate::io;

        let signal = io::tone(880.0, 22050, 0.3);
        let mfcc_coeffs = mfcc(&signal, 22050, 20, 2048, 512, 80).unwrap();

        let audio =
            mfcc_to_audio(&mfcc_coeffs, 80, 22050, 2048, 512, 0.0, 11025.0, 16, None).unwrap();

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
}
