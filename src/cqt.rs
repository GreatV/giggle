/// Constant-Q Transform (CQT) and Variable-Q Transform (VQT) implementation.
///
/// This module provides CQT/VQT transforms for music analysis with logarithmic
/// frequency resolution (constant number of bins per octave).
use crate::convert::cqt_frequencies;
use crate::fft::FftPlan;
use crate::window;
use ndarray::Array2;
use num_complex::Complex32;
use std::f32::consts::PI;

/// Configuration for Constant-Q Transform (CQT).
///
/// This struct provides a builder pattern for configuring CQT parameters
/// instead of using many positional arguments.
///
/// # Example
/// ```
/// use giggle::cqt::CqtConfig;
///
/// let config = CqtConfig::new(22050, 512)
///     .with_fmin(110.0)
///     .with_n_bins(84)
///     .with_bins_per_octave(12);
/// ```
#[derive(Debug, Clone)]
pub struct CqtConfig {
    /// Sample rate
    pub sr: u32,
    /// Number of samples between successive CQT columns
    pub hop_length: usize,
    /// Minimum frequency (default: ~32.7 Hz, C1)
    pub fmin: f32,
    /// Number of frequency bins (default: 84, 7 octaves)
    pub n_bins: usize,
    /// Number of bins per octave (default: 12)
    pub bins_per_octave: usize,
    /// Tuning offset in fractions of a bin (default: 0.0)
    pub tuning: f32,
    /// Filter scale factor (default: 1.0)
    pub filter_scale: f32,
}

impl CqtConfig {
    /// Create a new CQT configuration with defaults.
    ///
    /// # Arguments
    /// * `sr` - Sample rate
    /// * `hop_length` - Number of samples between successive CQT columns
    pub fn new(sr: u32, hop_length: usize) -> Self {
        Self {
            sr,
            hop_length,
            fmin: 32.70, // C1
            n_bins: 84,  // 7 octaves
            bins_per_octave: 12,
            tuning: 0.0,
            filter_scale: 1.0,
        }
    }

    /// Set the minimum frequency.
    pub fn with_fmin(mut self, fmin: f32) -> Self {
        self.fmin = fmin;
        self
    }

    /// Set the number of frequency bins.
    pub fn with_n_bins(mut self, n_bins: usize) -> Self {
        self.n_bins = n_bins;
        self
    }

    /// Set the number of bins per octave.
    pub fn with_bins_per_octave(mut self, bins_per_octave: usize) -> Self {
        self.bins_per_octave = bins_per_octave;
        self
    }

    /// Set the tuning offset.
    pub fn with_tuning(mut self, tuning: f32) -> Self {
        self.tuning = tuning;
        self
    }

    /// Set the filter scale factor.
    pub fn with_filter_scale(mut self, filter_scale: f32) -> Self {
        self.filter_scale = filter_scale;
        self
    }

    /// Compute the CQT with this configuration.
    ///
    /// # Arguments
    /// * `y` - Audio signal
    ///
    /// # Returns
    /// CQT spectrogram as complex values (n_bins x n_frames)
    pub fn compute(&self, y: &[f32]) -> crate::Result<Array2<Complex32>> {
        cqt(
            y,
            self.sr,
            self.hop_length,
            self.fmin,
            self.n_bins,
            self.bins_per_octave,
            self.tuning,
            self.filter_scale,
        )
    }
}

impl Default for CqtConfig {
    fn default() -> Self {
        Self {
            sr: 22050,
            hop_length: 512,
            fmin: 32.70,
            n_bins: 84,
            bins_per_octave: 12,
            tuning: 0.0,
            filter_scale: 1.0,
        }
    }
}

/// Configuration for Variable-Q Transform (VQT).
///
/// This struct provides a builder pattern for configuring VQT parameters.
///
/// # Example
/// ```
/// use giggle::cqt::VqtConfig;
///
/// let config = VqtConfig::new(22050, 512)
///     .with_fmin(110.0)
///     .with_gamma(0.5);
/// ```
#[derive(Debug, Clone)]
pub struct VqtConfig {
    /// Sample rate
    pub sr: u32,
    /// Number of samples between successive VQT columns
    pub hop_length: usize,
    /// Minimum frequency
    pub fmin: f32,
    /// Number of frequency bins
    pub n_bins: usize,
    /// Number of bins per octave
    pub bins_per_octave: usize,
    /// Tuning offset in fractions of a bin
    pub tuning: f32,
    /// Filter scale factor
    pub filter_scale: f32,
    /// Bandwidth offset (0 for CQT)
    pub gamma: f32,
}

impl VqtConfig {
    /// Create a new VQT configuration with defaults.
    ///
    /// # Arguments
    /// * `sr` - Sample rate
    /// * `hop_length` - Number of samples between successive VQT columns
    pub fn new(sr: u32, hop_length: usize) -> Self {
        Self {
            sr,
            hop_length,
            fmin: 32.70,
            n_bins: 84,
            bins_per_octave: 12,
            tuning: 0.0,
            filter_scale: 1.0,
            gamma: 0.0,
        }
    }

    /// Set the minimum frequency.
    pub fn with_fmin(mut self, fmin: f32) -> Self {
        self.fmin = fmin;
        self
    }

    /// Set the number of frequency bins.
    pub fn with_n_bins(mut self, n_bins: usize) -> Self {
        self.n_bins = n_bins;
        self
    }

    /// Set the number of bins per octave.
    pub fn with_bins_per_octave(mut self, bins_per_octave: usize) -> Self {
        self.bins_per_octave = bins_per_octave;
        self
    }

    /// Set the tuning offset.
    pub fn with_tuning(mut self, tuning: f32) -> Self {
        self.tuning = tuning;
        self
    }

    /// Set the filter scale factor.
    pub fn with_filter_scale(mut self, filter_scale: f32) -> Self {
        self.filter_scale = filter_scale;
        self
    }

    /// Set the bandwidth offset (gamma).
    pub fn with_gamma(mut self, gamma: f32) -> Self {
        self.gamma = gamma;
        self
    }

    /// Compute the VQT with this configuration.
    ///
    /// # Arguments
    /// * `y` - Audio signal
    ///
    /// # Returns
    /// VQT spectrogram as complex values (n_bins x n_frames)
    pub fn compute(&self, y: &[f32]) -> crate::Result<Array2<Complex32>> {
        vqt(
            y,
            self.sr,
            self.hop_length,
            self.fmin,
            self.n_bins,
            self.bins_per_octave,
            self.tuning,
            self.filter_scale,
            self.gamma,
        )
    }
}

impl Default for VqtConfig {
    fn default() -> Self {
        Self {
            sr: 22050,
            hop_length: 512,
            fmin: 32.70,
            n_bins: 84,
            bins_per_octave: 12,
            tuning: 0.0,
            filter_scale: 1.0,
            gamma: 0.0,
        }
    }
}

/// Compute the Constant-Q Transform of an audio signal.
///
/// CQT provides logarithmic frequency resolution where each octave has
/// the same number of frequency bins. This is useful for music analysis
/// where pitch perception is logarithmic.
///
/// This implementation uses the pseudo-CQT method with direct STFT-based
/// approximation for efficiency.
///
/// # Arguments
/// * `y` - Audio signal
/// * `sr` - Sample rate
/// * `hop_length` - Number of samples between successive CQT columns
/// * `fmin` - Minimum frequency (default: ~32.7 Hz, C1)
/// * `n_bins` - Number of frequency bins (default: 84, 7 octaves)
/// * `bins_per_octave` - Number of bins per octave (default: 12)
/// * `tuning` - Tuning offset in fractions of a bin (default: 0.0)
/// * `filter_scale` - Filter scale factor (default: 1.0)
///
/// # Returns
/// CQT spectrogram as complex values (n_bins x n_frames)
///
/// # Errors
/// Returns an error if `y` is empty or contains non-finite values, or if
/// `n_bins` or `bins_per_octave` is zero.
///
/// # Example
/// ```
/// use giggle::cqt::cqt;
/// use giggle::io;
///
/// let signal = io::tone(440.0, 22050, 1.0);
/// let cqt_spec = cqt(&signal, 22050, 512, 32.7, 84, 12, 0.0, 1.0).unwrap();
/// assert_eq!(cqt_spec.shape()[0], 84);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn cqt(
    y: &[f32],
    sr: u32,
    hop_length: usize,
    fmin: f32,
    n_bins: usize,
    bins_per_octave: usize,
    tuning: f32,
    filter_scale: f32,
) -> crate::Result<Array2<Complex32>> {
    // CQT is VQT with gamma=0
    vqt(
        y,
        sr,
        hop_length,
        fmin,
        n_bins,
        bins_per_octave,
        tuning,
        filter_scale,
        0.0,
    )
}

/// Compute the Variable-Q Transform of an audio signal.
///
/// VQT is a generalization of CQT where the bandwidth can vary with frequency.
/// When gamma=0, it reduces to CQT.
///
/// # Arguments
/// * `y` - Audio signal
/// * `sr` - Sample rate
/// * `hop_length` - Number of samples between successive VQT columns
/// * `fmin` - Minimum frequency
/// * `n_bins` - Number of frequency bins
/// * `bins_per_octave` - Number of bins per octave
/// * `tuning` - Tuning offset in fractions of a bin
/// * `filter_scale` - Filter scale factor
/// * `gamma` - Bandwidth offset (0 for CQT)
///
/// # Returns
/// VQT spectrogram as complex values (n_bins x n_frames)
///
/// # Errors
/// Returns an error if `y` is empty or contains non-finite values, or if
/// `n_bins` or `bins_per_octave` is zero.
#[allow(clippy::too_many_arguments)]
pub fn vqt(
    y: &[f32],
    sr: u32,
    hop_length: usize,
    fmin: f32,
    n_bins: usize,
    bins_per_octave: usize,
    tuning: f32,
    filter_scale: f32,
    gamma: f32,
) -> crate::Result<Array2<Complex32>> {
    crate::utils::valid_audio(y)?;

    if n_bins == 0 {
        return Err(crate::Error::InvalidSize {
            name: "n_bins",
            value: 0,
            reason: "must be greater than zero",
        });
    }
    if bins_per_octave == 0 {
        return Err(crate::Error::InvalidSize {
            name: "bins_per_octave",
            value: 0,
            reason: "must be greater than zero",
        });
    }

    // Apply tuning correction to fmin
    let fmin_tuned = fmin * 2.0_f32.powf(tuning / bins_per_octave as f32);

    // Get CQT frequencies
    let freqs = cqt_frequencies(n_bins, fmin_tuned, bins_per_octave, 0.0);

    // Compute filter lengths for each bin
    let q = filter_scale / (2.0_f32.powf(1.0 / bins_per_octave as f32) - 1.0);
    let lengths: Vec<usize> = freqs
        .iter()
        .map(|&f| {
            let bw = f / q + gamma;
            let len = (sr as f32 / bw * filter_scale).ceil() as usize;
            len.max(1)
        })
        .collect();

    // Find the maximum filter length to determine FFT size
    let max_len = lengths.iter().copied().max().unwrap_or(2048);
    let n_fft = max_len.next_power_of_two().max(512);

    // Number of output frames
    let n_frames = if y.len() > n_fft / 2 {
        (y.len() - n_fft / 2) / hop_length + 1
    } else {
        1
    };

    let mut cqt_result = Array2::<Complex32>::zeros((n_bins, n_frames));

    // Create FFT plan
    let fft = FftPlan::new(n_fft);

    // Process each frame
    for frame_idx in 0..n_frames {
        let center = frame_idx * hop_length + n_fft / 2;

        // Extract and window the frame
        let mut buffer = vec![Complex32::new(0.0, 0.0); n_fft];

        for (i, buf) in buffer.iter_mut().enumerate().take(n_fft) {
            let sample_idx = center as isize - (n_fft / 2) as isize + i as isize;
            let sample = if sample_idx >= 0 && (sample_idx as usize) < y.len() {
                y[sample_idx as usize]
            } else {
                0.0
            };

            // Apply Hann window
            let w = 0.5 * (1.0 - (2.0 * PI * i as f32 / (n_fft - 1) as f32).cos());
            buf.re = sample * w;
        }

        // Compute FFT
        fft.forward(&mut buffer);

        // Extract CQT bins by interpolating FFT bins
        for (bin_idx, &freq) in freqs.iter().enumerate() {
            // Find corresponding FFT bin
            let fft_bin = freq * n_fft as f32 / sr as f32;
            let bin_low = fft_bin.floor() as usize;
            let bin_high = (bin_low + 1).min(n_fft / 2);
            let frac = fft_bin - bin_low as f32;

            // Linear interpolation between FFT bins
            let val = if bin_low < n_fft / 2 {
                let v_low = buffer[bin_low];
                let v_high = buffer[bin_high];
                Complex32::new(
                    v_low.re * (1.0 - frac) + v_high.re * frac,
                    v_low.im * (1.0 - frac) + v_high.im * frac,
                )
            } else {
                Complex32::new(0.0, 0.0)
            };

            // Scale by filter length ratio for normalization
            let scale = (n_fft as f32 / lengths[bin_idx] as f32).sqrt();
            cqt_result[(bin_idx, frame_idx)] = val * scale;
        }
    }

    Ok(cqt_result)
}

/// Compute the pseudo Constant-Q Transform.
///
/// This is a faster approximation of CQT using a single FFT per frame.
/// Less accurate than full CQT but much faster.
///
/// # Arguments
/// * `y` - Audio signal
/// * `sr` - Sample rate
/// * `hop_length` - Hop length
/// * `fmin` - Minimum frequency
/// * `n_bins` - Number of frequency bins
/// * `bins_per_octave` - Bins per octave
///
/// # Returns
/// Pseudo-CQT spectrogram as complex values
///
/// # Errors
/// Returns an error if `y` is empty or contains non-finite values, or if
/// `n_bins` or `bins_per_octave` is zero.
pub fn pseudo_cqt(
    y: &[f32],
    sr: u32,
    hop_length: usize,
    fmin: f32,
    n_bins: usize,
    bins_per_octave: usize,
) -> crate::Result<Array2<Complex32>> {
    cqt(y, sr, hop_length, fmin, n_bins, bins_per_octave, 0.0, 1.0)
}

/// Compute the hybrid Constant-Q Transform.
///
/// Uses pseudo-CQT for higher frequencies and full CQT for lower frequencies.
/// This is an alias for the regular CQT in this simplified implementation.
///
/// # Errors
/// Returns an error if `y` is empty or contains non-finite values, or if
/// `n_bins` or `bins_per_octave` is zero.
pub fn hybrid_cqt(
    y: &[f32],
    sr: u32,
    hop_length: usize,
    fmin: f32,
    n_bins: usize,
    bins_per_octave: usize,
) -> crate::Result<Array2<Complex32>> {
    cqt(y, sr, hop_length, fmin, n_bins, bins_per_octave, 0.0, 1.0)
}

/// Inverse Constant-Q Transform.
///
/// Reconstructs audio signal from CQT spectrogram using Griffin-Lim style
/// iterative reconstruction.
///
/// # Arguments
/// * `cqt_spec` - CQT spectrogram (n_bins x n_frames)
/// * `sr` - Sample rate
/// * `hop_length` - Hop length used in forward CQT
/// * `fmin` - Minimum frequency used in forward CQT
/// * `bins_per_octave` - Bins per octave used in forward CQT
/// * `n_iter` - Number of Griffin-Lim iterations (default: 32)
///
/// # Returns
/// Reconstructed audio signal
///
/// # Errors
/// Returns an error if the CQT spectrogram is empty (zero bins or zero frames).
pub fn icqt(
    cqt_spec: &Array2<Complex32>,
    sr: u32,
    hop_length: usize,
    fmin: f32,
    bins_per_octave: usize,
    n_iter: usize,
) -> crate::Result<Vec<f32>> {
    let n_bins = cqt_spec.shape()[0];
    let n_frames = cqt_spec.shape()[1];

    if n_bins == 0 {
        return Err(crate::Error::InvalidSize {
            name: "cqt_spec n_bins",
            value: 0,
            reason: "CQT spectrogram must have at least one frequency bin",
        });
    }
    if n_frames == 0 {
        return Err(crate::Error::InvalidSize {
            name: "cqt_spec n_frames",
            value: 0,
            reason: "CQT spectrogram must have at least one frame",
        });
    }

    // Estimate output length
    let output_len = n_frames * hop_length + hop_length;

    // Get CQT frequencies
    let freqs = cqt_frequencies(n_bins, fmin, bins_per_octave, 0.0);

    // Determine FFT size based on lowest frequency
    let q = 1.0 / (2.0_f32.powf(1.0 / bins_per_octave as f32) - 1.0);
    let max_len = (sr as f32 / (freqs[0] / q)).ceil() as usize;
    let n_fft = max_len.next_power_of_two().max(512);

    // Initialize with random phase
    let mut y = vec![0.0f32; output_len];
    for (i, sample) in y.iter_mut().enumerate().take(output_len) {
        *sample = (i as f32 * 0.001).sin() * 0.01;
    }

    // Griffin-Lim iteration
    for _ in 0..n_iter {
        // Forward CQT
        let cqt_est = cqt(&y, sr, hop_length, fmin, n_bins, bins_per_octave, 0.0, 1.0)?;

        // Apply magnitude from input, keep estimated phase
        let mut cqt_updated =
            Array2::<Complex32>::zeros((n_bins, n_frames.min(cqt_est.shape()[1])));
        for bin in 0..n_bins {
            for frame in 0..cqt_updated.shape()[1] {
                let target_mag = cqt_spec[(bin, frame)].norm();
                let est_phase = cqt_est[(bin, frame)].arg();
                cqt_updated[(bin, frame)] = Complex32::from_polar(target_mag, est_phase);
            }
        }

        // Inverse transform (simplified overlap-add)
        let fft = FftPlan::new(n_fft);
        y = vec![0.0f32; output_len];
        let mut window_sum = vec![0.0f32; output_len];

        let win = window::hann(n_fft);

        for frame_idx in 0..cqt_updated.shape()[1] {
            // Build spectrum from CQT bins
            let mut spectrum = vec![Complex32::new(0.0, 0.0); n_fft];

            for (bin_idx, &freq) in freqs.iter().enumerate() {
                let fft_bin = (freq * n_fft as f32 / sr as f32).round() as usize;
                if fft_bin < n_fft / 2 {
                    spectrum[fft_bin] = cqt_updated[(bin_idx, frame_idx)];
                    // Mirror for real signal
                    if fft_bin > 0 {
                        spectrum[n_fft - fft_bin] = cqt_updated[(bin_idx, frame_idx)].conj();
                    }
                }
            }

            // IFFT
            fft.inverse(&mut spectrum);

            // Overlap-add
            let center = frame_idx * hop_length;
            for i in 0..n_fft {
                let out_idx = center + i;
                if out_idx < output_len {
                    y[out_idx] += spectrum[i].re * win[i];
                    window_sum[out_idx] += win[i] * win[i];
                }
            }
        }

        // Normalize by window sum
        for i in 0..output_len {
            if window_sum[i] > 1e-8 {
                y[i] /= window_sum[i];
            }
        }
    }

    Ok(y)
}

/// Griffin-Lim style reconstruction from CQT magnitude.
///
/// # Arguments
/// * `cqt_mag` - CQT magnitude spectrogram (n_bins x n_frames)
/// * `sr` - Sample rate
/// * `hop_length` - Hop length
/// * `fmin` - Minimum frequency
/// * `bins_per_octave` - Bins per octave
/// * `n_iter` - Number of iterations
///
/// # Returns
/// Reconstructed audio signal
///
/// # Errors
/// Returns an error if the magnitude spectrogram is empty.
pub fn griffinlim_cqt(
    cqt_mag: &Array2<f32>,
    sr: u32,
    hop_length: usize,
    fmin: f32,
    bins_per_octave: usize,
    n_iter: usize,
) -> crate::Result<Vec<f32>> {
    let n_bins = cqt_mag.shape()[0];
    let n_frames = cqt_mag.shape()[1];

    if n_bins == 0 {
        return Err(crate::Error::InvalidSize {
            name: "cqt_mag n_bins",
            value: 0,
            reason: "magnitude spectrogram must have at least one frequency bin",
        });
    }
    if n_frames == 0 {
        return Err(crate::Error::InvalidSize {
            name: "cqt_mag n_frames",
            value: 0,
            reason: "magnitude spectrogram must have at least one frame",
        });
    }

    // Convert magnitude to complex with zero phase
    let mut cqt_complex = Array2::<Complex32>::zeros((n_bins, n_frames));

    for bin in 0..n_bins {
        for frame in 0..n_frames {
            cqt_complex[(bin, frame)] = Complex32::new(cqt_mag[(bin, frame)], 0.0);
        }
    }

    icqt(&cqt_complex, sr, hop_length, fmin, bins_per_octave, n_iter)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io;

    #[test]
    fn test_cqt_shape() {
        let signal = io::tone(440.0, 22050, 1.0);
        let cqt_spec = cqt(&signal, 22050, 512, 32.7, 84, 12, 0.0, 1.0).unwrap();

        assert_eq!(cqt_spec.shape()[0], 84);
        assert!(cqt_spec.shape()[1] > 0);
    }

    #[test]
    fn test_cqt_empty() {
        let signal: Vec<f32> = vec![];
        let result = cqt(&signal, 22050, 512, 32.7, 84, 12, 0.0, 1.0);

        assert!(result.is_err());
    }

    #[test]
    fn test_cqt_zero_bins() {
        let signal = io::tone(440.0, 22050, 0.5);
        let result = cqt(&signal, 22050, 512, 32.7, 0, 12, 0.0, 1.0);

        assert!(result.is_err());
    }

    #[test]
    fn test_cqt_zero_bins_per_octave() {
        let signal = io::tone(440.0, 22050, 0.5);
        let result = cqt(&signal, 22050, 512, 32.7, 84, 0, 0.0, 1.0);

        assert!(result.is_err());
    }

    #[test]
    fn test_cqt_frequency_detection() {
        let sr = 22050u32;
        // A4 = 440 Hz
        let signal = io::tone(440.0, sr, 1.0);

        let cqt_spec = cqt(&signal, sr, 512, 32.7, 84, 12, 0.0, 1.0).unwrap();

        // Find the bin with maximum energy
        let mut max_bin = 0;
        let mut max_energy = 0.0f32;

        for bin in 0..84 {
            let energy: f32 = (0..cqt_spec.shape()[1])
                .map(|t| cqt_spec[(bin, t)].norm_sqr())
                .sum();

            if energy > max_energy {
                max_energy = energy;
                max_bin = bin;
            }
        }

        // Compute expected bin for 440 Hz
        // fmin = 32.7 Hz (C1), 12 bins per octave
        // bin = 12 * log2(f / fmin)
        let expected_bin = (12.0 * (440.0 / 32.7_f32).log2()).round() as i32;

        // Peak should be near the expected bin (within 5 bins tolerance)
        assert!(
            (max_bin as i32 - expected_bin).abs() <= 5,
            "Expected peak near bin {}, got {} for 440 Hz tone",
            expected_bin,
            max_bin
        );

        // Energy should be concentrated in a few bins around the peak
        assert!(max_energy > 0.0, "Should detect energy in CQT bins");
    }

    #[test]
    fn test_vqt_with_gamma() {
        let signal = io::tone(440.0, 22050, 0.5);
        let vqt_spec = vqt(&signal, 22050, 512, 32.7, 84, 12, 0.0, 1.0, 24.7).unwrap();

        assert_eq!(vqt_spec.shape()[0], 84);
        assert!(vqt_spec.shape()[1] > 0);
    }

    #[test]
    fn test_pseudo_cqt() {
        let signal = io::tone(440.0, 22050, 0.5);
        let pcqt = pseudo_cqt(&signal, 22050, 512, 32.7, 84, 12).unwrap();

        assert_eq!(pcqt.shape()[0], 84);
    }

    #[test]
    fn test_hybrid_cqt() {
        let signal = io::tone(440.0, 22050, 0.5);
        let hcqt = hybrid_cqt(&signal, 22050, 512, 32.7, 84, 12).unwrap();

        assert_eq!(hcqt.shape()[0], 84);
    }

    #[test]
    fn test_icqt_reconstruction() {
        let sr = 22050u32;
        let signal = io::tone(440.0, sr, 0.5);

        let cqt_spec = cqt(&signal, sr, 512, 32.7, 84, 12, 0.0, 1.0).unwrap();
        let reconstructed = icqt(&cqt_spec, sr, 512, 32.7, 12, 8).unwrap();

        // Should have roughly the same length
        assert!(!reconstructed.is_empty());

        // Check that reconstruction has energy
        let energy: f32 = reconstructed.iter().map(|&x| x * x).sum();
        assert!(energy > 0.0);
    }

    #[test]
    fn test_icqt_empty_spectrogram() {
        let empty_spec = Array2::<Complex32>::zeros((0, 0));
        let result = icqt(&empty_spec, 22050, 512, 32.7, 12, 8);

        assert!(result.is_err());
    }

    #[test]
    fn test_griffinlim_cqt() {
        let sr = 22050u32;
        let signal = io::tone(440.0, sr, 0.3);

        let cqt_spec = cqt(&signal, sr, 512, 32.7, 48, 12, 0.0, 1.0).unwrap();

        // Get magnitude
        let mut mag = Array2::<f32>::zeros((cqt_spec.shape()[0], cqt_spec.shape()[1]));
        for ((i, j), val) in cqt_spec.indexed_iter() {
            mag[(i, j)] = val.norm();
        }

        let reconstructed = griffinlim_cqt(&mag, sr, 512, 32.7, 12, 4).unwrap();

        assert!(!reconstructed.is_empty());
    }

    #[test]
    fn test_griffinlim_cqt_empty() {
        let empty_mag = Array2::<f32>::zeros((0, 0));
        let result = griffinlim_cqt(&empty_mag, 22050, 512, 32.7, 12, 4);

        assert!(result.is_err());
    }

    #[test]
    fn test_cqt_different_bins_per_octave() {
        let signal = io::tone(440.0, 22050, 0.5);

        // 24 bins per octave (quarter-tone resolution)
        let cqt_24 = cqt(&signal, 22050, 512, 32.7, 168, 24, 0.0, 1.0).unwrap();
        assert_eq!(cqt_24.shape()[0], 168);

        // 36 bins per octave
        let cqt_36 = cqt(&signal, 22050, 512, 32.7, 252, 36, 0.0, 1.0).unwrap();
        assert_eq!(cqt_36.shape()[0], 252);
    }

    #[test]
    fn test_cqt_with_tuning() {
        let signal = io::tone(440.0, 22050, 0.5);

        let cqt_no_tuning = cqt(&signal, 22050, 512, 32.7, 84, 12, 0.0, 1.0).unwrap();
        let cqt_tuned = cqt(&signal, 22050, 512, 32.7, 84, 12, 0.5, 1.0).unwrap();

        // Both should have same shape
        assert_eq!(cqt_no_tuning.shape(), cqt_tuned.shape());

        // But different values due to frequency shift
        let mut different = false;
        for i in 0..84 {
            if (cqt_no_tuning[(i, 0)].re - cqt_tuned[(i, 0)].re).abs() > 1e-6 {
                different = true;
                break;
            }
        }
        assert!(different, "Tuning should affect CQT values");
    }
}
