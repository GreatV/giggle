use super::{hz_to_mel, mel_to_hz};

/// Get FFT bin frequencies.
///
/// # Arguments
/// * `sr` - Sample rate in Hz
/// * `n_fft` - FFT size
///
/// # Returns
/// Vector of frequency values (Hz) for each FFT bin
///
/// # Example
/// ```
/// use giggle::convert::fft_frequencies;
///
/// let freqs = fft_frequencies(22050, 2048);
/// assert_eq!(freqs.len(), 1025); // n_fft/2 + 1
/// assert_eq!(freqs[0], 0.0);
/// ```
pub fn fft_frequencies(sr: u32, n_fft: usize) -> Vec<f32> {
    let n_bins = n_fft / 2 + 1;
    (0..n_bins)
        .map(|i| i as f32 * sr as f32 / n_fft as f32)
        .collect()
}

/// Get mel-spaced frequencies.
///
/// # Arguments
/// * `n_mels` - Number of mel bins
/// * `fmin` - Minimum frequency (Hz)
/// * `fmax` - Maximum frequency (Hz)
/// * `htk` - Use HTK formula (default: false)
///
/// # Returns
/// Vector of frequency values (Hz) for mel bin centers
///
/// # Example
/// ```
/// use giggle::convert::mel_frequencies;
///
/// let freqs = mel_frequencies(128, 0.0, 8000.0, false);
/// assert_eq!(freqs.len(), 128);
/// assert!(freqs[0] >= 0.0);
/// // Last frequency should be close to fmax
/// assert!((freqs[127] - 8000.0).abs() < 1.0);
/// ```
pub fn mel_frequencies(n_mels: usize, fmin: f32, fmax: f32, htk: bool) -> Vec<f32> {
    let mel_min = hz_to_mel(&[fmin], htk)[0];
    let mel_max = hz_to_mel(&[fmax], htk)[0];

    let mels: Vec<f32> = (0..n_mels)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels - 1).max(1) as f32)
        .collect();

    mel_to_hz(&mels, htk)
}

/// Get tempo frequencies for tempogram analysis.
///
/// # Arguments
/// * `sr` - Sample rate
/// * `hop_length` - Hop length
/// * `win_length` - Window length in frames
///
/// # Returns
/// Tempo values in BPM for each tempogram bin
///
/// # Example
/// ```
/// use giggle::convert::tempo_frequencies;
///
/// let tempos = tempo_frequencies(22050, 512, 384);
/// assert!(tempos.len() > 0);
/// ```
pub fn tempo_frequencies(sr: u32, hop_length: usize, win_length: usize) -> Vec<f32> {
    let n_bins = win_length / 2 + 1;
    (0..n_bins)
        .map(|i| {
            let freq_hz = i as f32 * sr as f32 / (hop_length * win_length) as f32;
            freq_hz * 60.0 // Convert to BPM
        })
        .collect()
}

/// Compute the center frequencies of Constant-Q bins.
///
/// # Arguments
/// * `n_bins` - Number of CQ bins
/// * `fmin` - Minimum frequency in Hz
/// * `bins_per_octave` - Number of bins per octave (default: 12)
/// * `tuning` - Deviation from A440 tuning in fractional bins (default: 0.0)
///
/// # Returns
/// Center frequency for each CQ bin
///
/// # Example
/// ```
/// use giggle::convert::cqt_frequencies;
///
/// // Get CQ frequencies for 24 notes starting at C2 (~65 Hz)
/// let freqs = cqt_frequencies(24, 65.406, 12, 0.0);
/// assert_eq!(freqs.len(), 24);
/// assert!((freqs[0] - 65.406).abs() < 0.01);
/// assert!((freqs[12] - 130.813).abs() < 0.1); // One octave up
/// ```
pub fn cqt_frequencies(n_bins: usize, fmin: f32, bins_per_octave: usize, tuning: f32) -> Vec<f32> {
    let correction = 2.0_f32.powf(tuning / bins_per_octave as f32);

    (0..n_bins)
        .map(|i| correction * fmin * 2.0_f32.powf(i as f32 / bins_per_octave as f32))
        .collect()
}

/// Compute Fourier tempo frequencies.
///
/// # Arguments
/// * `sr` - Sample rate
/// * `hop_length` - Hop length
/// * `win_length` - Window length (in frames)
///
/// # Returns
/// Tempo values in BPM for each Fourier tempogram bin
///
/// # Example
/// ```
/// use giggle::convert::fourier_tempo_frequencies;
///
/// let tempos = fourier_tempo_frequencies(22050, 512, 384);
/// assert!(tempos.len() > 0);
/// ```
pub fn fourier_tempo_frequencies(sr: u32, hop_length: usize, win_length: usize) -> Vec<f32> {
    if win_length == 0 {
        return Vec::new();
    }

    let n_bins = win_length / 2 + 1;
    (0..n_bins)
        .map(|i| {
            let freq_hz = i as f32 * sr as f32 / (hop_length * win_length) as f32;
            freq_hz * 60.0 // Convert to BPM
        })
        .collect()
}

/// Convert A4 reference frequency to tuning offset.
///
/// # Arguments
/// * `a4` - Reference frequency for A4 in Hz (typically near 440)
///
/// # Returns
/// Tuning offset in fractional bins (relative to A440)
///
/// # Example
/// ```
/// use giggle::convert::a4_to_tuning;
///
/// assert!((a4_to_tuning(440.0)).abs() < 0.001);
/// assert!((a4_to_tuning(442.0) - 0.0787).abs() < 0.01);
/// ```
pub fn a4_to_tuning(a4: f32) -> f32 {
    12.0 * (a4 / 440.0).log2()
}

/// Convert tuning offset to A4 reference frequency.
///
/// # Arguments
/// * `tuning` - Tuning offset in fractional bins
///
/// # Returns
/// A4 reference frequency in Hz
///
/// # Example
/// ```
/// use giggle::convert::tuning_to_a4;
///
/// assert!((tuning_to_a4(0.0) - 440.0).abs() < 0.001);
/// ```
pub fn tuning_to_a4(tuning: f32) -> f32 {
    440.0 * 2.0_f32.powf(tuning / 12.0)
}
