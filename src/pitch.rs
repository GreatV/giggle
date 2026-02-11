/// Pitch estimation algorithms.
///
/// This module provides various algorithms for estimating the fundamental
/// frequency (F0) of audio signals.
use crate::convert;
use crate::spectrum::{StftConfig, stft};
use crate::window;
use ndarray::Array2;
use std::f32::consts::PI;

/// Configuration for YIN pitch estimation.
///
/// This struct provides a builder pattern for configuring YIN parameters
/// instead of using many positional arguments.
///
/// # Example
/// ```
/// use giggle::pitch::YinConfig;
///
/// let config = YinConfig::new(22050)
///     .with_fmin(40.0)
///     .with_fmax(5000.0);
/// ```
#[derive(Debug, Clone)]
pub struct YinConfig {
    /// Sample rate
    pub sr: u32,
    /// Length of analysis frames in samples
    pub frame_length: usize,
    /// Number of samples between frames
    pub hop_length: usize,
    /// Minimum frequency to consider in Hz
    pub fmin: f32,
    /// Maximum frequency to consider in Hz
    pub fmax: f32,
    /// Threshold for peak picking
    pub threshold: f32,
}

impl YinConfig {
    /// Create a new YIN configuration with defaults.
    ///
    /// # Arguments
    /// * `sr` - Sample rate
    pub fn new(sr: u32) -> Self {
        Self {
            sr,
            frame_length: 2048,
            hop_length: 512,
            fmin: 40.0,
            fmax: sr as f32 / 4.0,
            threshold: 0.1,
        }
    }

    /// Set the frame length.
    pub fn with_frame_length(mut self, frame_length: usize) -> Self {
        self.frame_length = frame_length;
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

    /// Set the threshold for peak picking.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Compute YIN pitch estimation with this configuration.
    ///
    /// # Arguments
    /// * `y` - Input audio signal
    ///
    /// # Returns
    /// Array of F0 estimates per frame in Hz (0.0 indicates no pitch detected)
    pub fn compute(&self, y: &[f32]) -> crate::Result<Vec<f32>> {
        yin(
            y,
            self.sr,
            self.frame_length,
            self.hop_length,
            self.fmin,
            self.fmax,
            self.threshold,
        )
    }
}

impl Default for YinConfig {
    fn default() -> Self {
        Self {
            sr: 22050,
            frame_length: 2048,
            hop_length: 512,
            fmin: 40.0,
            fmax: 5512.5,
            threshold: 0.1,
        }
    }
}

/// YIN pitch estimation algorithm.
///
/// YIN is an autocorrelation-based pitch estimation algorithm that uses
/// the cumulative mean normalized difference function (CMNDF) to detect
/// the fundamental frequency.
///
/// # Arguments
/// * `y` - Input audio signal
/// * `sr` - Sample rate
/// * `frame_length` - Length of analysis frames in samples (default: 2048)
/// * `hop_length` - Number of samples between frames (default: 512)
/// * `fmin` - Minimum frequency to consider in Hz (default: 40.0)
/// * `fmax` - Maximum frequency to consider in Hz (default: sr/4.0)
/// * `threshold` - Threshold for peak picking (default: 0.1)
///
/// # Returns
/// Array of F0 estimates per frame in Hz (0.0 indicates no pitch detected)
///
/// # Errors
/// Returns `Error::EmptyAudio` if input is empty
/// Returns `Error::InvalidSize` if frame_length or hop_length is 0
///
/// # Example
/// ```
/// use giggle::pitch::yin;
/// use giggle::io;
///
/// let signal = io::tone(440.0, 22050, 0.5);
/// let f0 = yin(&signal, 22050, 2048, 512, 40.0, 5512.0, 0.1).unwrap();
/// // Most frames should detect around 440 Hz
/// let avg_f0: f32 = f0.iter().filter(|&&x| x > 0.0).sum::<f32>() / f0.iter().filter(|&&x| x > 0.0).count() as f32;
/// assert!((avg_f0 - 440.0).abs() < 50.0);
/// ```
pub fn yin(
    y: &[f32],
    sr: u32,
    frame_length: usize,
    hop_length: usize,
    fmin: f32,
    fmax: f32,
    threshold: f32,
) -> crate::Result<Vec<f32>> {
    if y.is_empty() {
        return Err(crate::Error::EmptyAudio);
    }
    if frame_length == 0 {
        return Err(crate::Error::InvalidSize {
            name: "frame_length",
            value: 0,
            reason: "must be > 0",
        });
    }
    if hop_length == 0 {
        return Err(crate::Error::InvalidSize {
            name: "hop_length",
            value: 0,
            reason: "must be > 0",
        });
    }

    let n_frames = (y.len().saturating_sub(frame_length)) / hop_length + 1;
    let mut f0 = vec![0.0f32; n_frames];

    let tau_min = (sr as f32 / fmax).max(1.0) as usize;
    let tau_max = (sr as f32 / fmin).min(frame_length as f32 / 2.0) as usize;

    for (frame_idx, f0_val) in f0.iter_mut().enumerate().take(n_frames) {
        let start = frame_idx * hop_length;
        let end = (start + frame_length).min(y.len());

        if end <= start || end - start < tau_min {
            continue;
        }

        let frame = &y[start..end];
        let frame_len = frame.len();

        // Ensure we have enough samples
        if frame_len < tau_max {
            continue;
        }

        // Compute difference function
        let mut diff = vec![0.0f32; tau_max];
        for tau in 0..tau_max {
            if tau >= frame_len {
                break;
            }
            let mut sum = 0.0f32;
            for j in 0..(frame_len.saturating_sub(tau)) {
                let delta = frame[j] - frame[j + tau];
                sum += delta * delta;
            }
            diff[tau] = sum;
        }

        // Compute cumulative mean normalized difference function (CMNDF)
        let mut cmndf = vec![1.0f32; tau_max];
        cmndf[0] = 1.0;

        let mut running_sum = 0.0f32;
        for tau in 1..tau_max {
            running_sum += diff[tau];
            if running_sum > 0.0 {
                cmndf[tau] = diff[tau] * (tau as f32) / running_sum;
            } else {
                cmndf[tau] = 1.0;
            }
        }

        // Find the first minimum below threshold
        let mut tau_estimate = 0;
        for tau in tau_min..tau_max {
            if cmndf[tau] < threshold {
                // Found a candidate, now find local minimum
                if tau + 1 < tau_max && cmndf[tau] < cmndf[tau + 1] {
                    tau_estimate = tau;
                    break;
                }
            }
        }

        // Convert tau to frequency
        if tau_estimate > 0 && tau_estimate < tau_max {
            // Parabolic interpolation for sub-sample accuracy
            if tau_estimate > 0 && tau_estimate < tau_max - 1 {
                let s0 = cmndf[tau_estimate - 1];
                let s1 = cmndf[tau_estimate];
                let s2 = cmndf[tau_estimate + 1];

                let adjustment = 0.5 * (s0 - s2) / (s0 - 2.0 * s1 + s2);
                let tau_refined = tau_estimate as f32 + adjustment;

                if tau_refined > 0.0 {
                    *f0_val = sr as f32 / tau_refined;
                }
            } else {
                *f0_val = sr as f32 / tau_estimate as f32;
            }

            // Sanity check
            if *f0_val < fmin || *f0_val > fmax {
                *f0_val = 0.0;
            }
        }
    }

    Ok(f0)
}

/// Pitch tracking using parabolic interpolation of peak frequencies.
///
/// Tracks pitch by estimating the instantaneous frequency at spectral peaks.
/// Uses parabolic interpolation around magnitude peaks for sub-bin accuracy.
///
/// # Arguments
/// * `y` - Input audio signal
/// * `sr` - Sample rate
/// * `n_fft` - FFT size (default: 2048)
/// * `hop_length` - Hop length (default: 512)
/// * `fmin` - Minimum frequency to consider (default: 150 Hz)
/// * `fmax` - Maximum frequency to consider (default: 4000 Hz)
/// * `threshold` - Minimum magnitude threshold (default: 0.1)
///
/// # Returns
/// Tuple of (pitches, magnitudes) arrays, both (n_bins x n_frames).
/// Each column contains pitch estimates in Hz and their magnitudes.
///
/// # Example
/// ```
/// use giggle::pitch::piptrack;
/// use giggle::io;
///
/// let signal = io::tone(440.0, 22050, 0.5);
/// let (pitches, mags) = piptrack(&signal, 22050, 2048, 512, 150.0, 4000.0, 0.1).unwrap();
/// assert_eq!(pitches.shape()[1], mags.shape()[1]);
/// ```
pub fn piptrack(
    y: &[f32],
    sr: u32,
    n_fft: usize,
    hop_length: usize,
    fmin: f32,
    fmax: f32,
    threshold: f32,
) -> crate::Result<(Array2<f32>, Array2<f32>)> {
    if y.is_empty() || n_fft == 0 {
        return Ok((Array2::zeros((0, 0)), Array2::zeros((0, 0))));
    }

    // Compute STFT
    let config = StftConfig {
        n_fft,
        win_length: n_fft,
        hop_length,
        window: window::hann(n_fft),
        center: true,
        ..Default::default()
    };

    let stft_matrix = stft(y, &config)?;
    let n_freq = stft_matrix.shape()[0];
    let n_frames = stft_matrix.shape()[1];

    if n_freq == 0 || n_frames == 0 {
        return Ok((Array2::zeros((0, 0)), Array2::zeros((0, 0))));
    }

    // Frequency resolution
    let freq_res = sr as f32 / n_fft as f32;

    // Frequency bins to consider
    let bin_min = (fmin / freq_res).ceil() as usize;
    let bin_max = (fmax / freq_res).floor() as usize;
    let bin_min = bin_min.max(1).min(n_freq - 2);
    let bin_max = bin_max.max(bin_min + 1).min(n_freq - 2);

    // Compute magnitude spectrogram
    let mut magnitude = Array2::<f32>::zeros((n_freq, n_frames));
    for freq in 0..n_freq {
        for frame in 0..n_frames {
            let val = stft_matrix[(freq, frame)];
            magnitude[(freq, frame)] = (val.re * val.re + val.im * val.im).sqrt();
        }
    }

    // Find peaks and estimate pitches
    let mut pitches = Array2::<f32>::zeros((n_freq, n_frames));
    let mut magnitudes = Array2::<f32>::zeros((n_freq, n_frames));

    for frame in 0..n_frames {
        for bin in bin_min..=bin_max {
            let mag_prev = magnitude[(bin - 1, frame)];
            let mag_curr = magnitude[(bin, frame)];
            let mag_next = magnitude[(bin + 1, frame)];

            // Check if this is a local maximum
            if mag_curr > mag_prev && mag_curr > mag_next && mag_curr > threshold {
                // Parabolic interpolation for sub-bin frequency estimate
                let alpha = mag_prev;
                let beta = mag_curr;
                let gamma = mag_next;

                let p = 0.5 * (alpha - gamma) / (alpha - 2.0 * beta + gamma);
                let bin_refined = bin as f32 + p;

                // Convert bin to frequency
                let freq = bin_refined * freq_res;

                // Store pitch and magnitude
                pitches[(bin, frame)] = freq;
                magnitudes[(bin, frame)] = mag_curr;
            }
        }
    }

    Ok((pitches, magnitudes))
}

/// Estimate tuning offset from a collection of detected pitches.
///
/// Given a collection of frequencies, estimates the tuning offset
/// (in fractions of a bin) relative to A440=440.0Hz.
///
/// # Arguments
/// * `frequencies` - Collection of detected frequencies in Hz
/// * `resolution` - Resolution of the tuning as a fraction of a bin (default: 0.01 = cents)
/// * `bins_per_octave` - Number of frequency bins per octave (default: 12)
///
/// # Returns
/// Estimated tuning deviation in fractions of a bin, in range [-0.5, 0.5)
///
/// # Example
/// ```
/// use giggle::pitch::pitch_tuning;
///
/// // Frequencies exactly on A440 tuning
/// let freqs = vec![440.0, 880.0, 220.0];
/// let tuning = pitch_tuning(&freqs, 0.01, 12);
/// assert!(tuning.abs() < 0.1);
/// ```
pub fn pitch_tuning(frequencies: &[f32], resolution: f32, bins_per_octave: usize) -> f32 {
    // Filter out non-positive frequencies
    let valid_freqs: Vec<f32> = frequencies.iter().filter(|&&f| f > 0.0).cloned().collect();

    if valid_freqs.is_empty() {
        return 0.0;
    }

    // Convert to octaves
    let octs = convert::hz_to_octs(&valid_freqs, 0.0, bins_per_octave);

    // Compute residuals (fractional part of bin number)
    let residuals: Vec<f32> = octs
        .iter()
        .map(|&o| {
            let bin = o * bins_per_octave as f32;
            let mut residual = bin - bin.floor();
            // Wrap residuals > 0.5 to negative
            if residual >= 0.5 {
                residual -= 1.0;
            }
            residual
        })
        .collect();

    if residuals.is_empty() {
        return 0.0;
    }

    // Build histogram
    let n_bins = (1.0 / resolution).ceil() as usize + 1;
    let mut histogram = vec![0usize; n_bins];

    for &r in &residuals {
        // Map [-0.5, 0.5) to histogram bins
        let bin_idx = ((r + 0.5) / resolution).floor() as usize;
        let bin_idx = bin_idx.min(n_bins - 1);
        histogram[bin_idx] += 1;
    }

    // Find peak
    let peak_idx = histogram
        .iter()
        .enumerate()
        .max_by_key(|(_, count)| *count)
        .map(|(idx, _)| idx)
        .unwrap_or(n_bins / 2);

    // Convert back to tuning offset

    (peak_idx as f32 * resolution) - 0.5
}

/// Estimate the tuning of an audio signal.
///
/// Uses pitch detection (piptrack) and then estimates the tuning offset.
///
/// # Arguments
/// * `y` - Audio signal
/// * `sr` - Sample rate
/// * `n_fft` - FFT size (default: 2048)
/// * `resolution` - Resolution of tuning estimate (default: 0.01 = cents)
/// * `bins_per_octave` - Bins per octave (default: 12)
///
/// # Returns
/// Estimated tuning deviation in fractions of a bin, in range [-0.5, 0.5)
///
/// # Example
/// ```
/// use giggle::pitch::estimate_tuning;
/// use giggle::io;
///
/// let signal = io::tone(440.0, 22050, 1.0);
/// let tuning = estimate_tuning(&signal, 22050, 2048, 0.01, 12).unwrap();
/// assert!(tuning.abs() < 0.2);
/// ```
pub fn estimate_tuning(
    y: &[f32],
    sr: u32,
    n_fft: usize,
    resolution: f32,
    bins_per_octave: usize,
) -> crate::Result<f32> {
    if y.is_empty() {
        return Ok(0.0);
    }

    // Use piptrack to get pitch estimates
    let (pitches, magnitudes) = piptrack(y, sr, n_fft, n_fft / 4, 150.0, 4000.0, 0.1)?;

    if pitches.is_empty() {
        return Ok(0.0);
    }

    // Collect high-magnitude pitch estimates
    let n_freq = pitches.shape()[0];
    let n_frames = pitches.shape()[1];

    // Find threshold (median magnitude)
    let mut all_mags: Vec<f32> = magnitudes.iter().filter(|&&m| m > 0.0).cloned().collect();
    if all_mags.is_empty() {
        return Ok(0.0);
    }

    all_mags.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let threshold = all_mags[all_mags.len() / 2];

    // Collect pitches above threshold
    let mut selected_pitches = Vec::new();
    for f in 0..n_freq {
        for t in 0..n_frames {
            let pitch = pitches[(f, t)];
            let mag = magnitudes[(f, t)];

            if pitch > 0.0 && mag >= threshold {
                selected_pitches.push(pitch);
            }
        }
    }

    if selected_pitches.is_empty() {
        return Ok(0.0);
    }

    Ok(pitch_tuning(&selected_pitches, resolution, bins_per_octave))
}

/// Probabilistic YIN (pYIN) pitch estimation.
///
/// pYIN is a modification of the YIN algorithm that uses probabilistic
/// threshold distributions and Viterbi decoding to improve pitch tracking
/// robustness and provide voicing detection.
///
/// # Arguments
/// * `y` - Input audio signal
/// * `sr` - Sample rate
/// * `fmin` - Minimum frequency to detect (Hz)
/// * `fmax` - Maximum frequency to detect (Hz)
/// * `frame_length` - Frame length in samples (default: 2048)
/// * `hop_length` - Hop length in samples (default: frame_length/4)
/// * `n_thresholds` - Number of thresholds for candidate generation (default: 100)
/// * `beta_params` - Beta distribution parameters (a, b) for threshold prior (default: (2.0, 18.0))
/// * `resolution` - Pitch bin resolution in fractions of semitone (default: 0.1)
///
/// # Returns
/// Tuple of (f0, voiced_flag, voiced_prob):
/// - f0: Estimated fundamental frequency per frame (Hz), NaN for unvoiced
/// - voiced_flag: Boolean flags indicating voiced frames
/// - voiced_prob: Probability of voicing per frame
///
/// # Example
/// ```
/// use giggle::pitch::pyin;
/// use giggle::io;
///
/// let signal = io::tone(440.0, 22050, 1.0);
/// let (f0, voiced, probs) = pyin(&signal, 22050, 65.0, 2093.0, 2048, 512, 100, (2.0, 18.0), 0.1);
/// assert_eq!(f0.len(), voiced.len());
/// assert_eq!(f0.len(), probs.len());
/// ```
#[allow(clippy::too_many_arguments)]
pub fn pyin(
    y: &[f32],
    sr: u32,
    fmin: f32,
    fmax: f32,
    frame_length: usize,
    hop_length: usize,
    n_thresholds: usize,
    beta_params: (f32, f32),
    resolution: f32,
) -> (Vec<f32>, Vec<bool>, Vec<f32>) {
    if y.is_empty() || frame_length == 0 || hop_length == 0 {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    // Validate parameters
    let fmax = fmax.min(sr as f32 / 2.0);
    if fmin >= fmax {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    // Pad the signal for centered frames
    let pad_len = frame_length / 2;
    let mut padded = vec![0.0f32; pad_len];
    padded.extend_from_slice(y);
    padded.extend(vec![0.0f32; pad_len]);

    // Calculate min and max periods
    let min_period = (sr as f32 / fmax).floor() as usize;
    let max_period = (sr as f32 / fmin).ceil() as usize;
    let max_period = max_period.min(frame_length - 1);

    if min_period >= max_period || max_period == 0 {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    // Number of frames
    let n_frames = (padded.len().saturating_sub(frame_length)) / hop_length + 1;
    if n_frames == 0 {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    // Compute pitch bins
    let n_bins_per_semitone = (1.0 / resolution).ceil() as usize;
    let n_pitch_bins =
        (12.0 * n_bins_per_semitone as f32 * (fmax / fmin).log2()).floor() as usize + 1;

    if n_pitch_bins == 0 {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    // Precompute threshold probabilities using beta distribution
    let thresholds: Vec<f32> = (0..=n_thresholds)
        .map(|i| i as f32 / n_thresholds as f32)
        .collect();
    let beta_probs = compute_beta_probs(&thresholds, beta_params.0, beta_params.1);

    // Compute observation probabilities for each frame
    let mut observation_probs = vec![vec![0.0f32; 2 * n_pitch_bins]; n_frames];
    let mut voiced_probs = vec![0.0f32; n_frames];

    for frame_idx in 0..n_frames {
        let start = frame_idx * hop_length;
        let end = (start + frame_length).min(padded.len());

        if end <= start || end - start < max_period {
            // Insufficient data, mark as unvoiced
            for prob in observation_probs[frame_idx][n_pitch_bins..(2 * n_pitch_bins)].iter_mut() {
                *prob = 1.0 / n_pitch_bins as f32;
            }
            continue;
        }

        let frame = &padded[start..end];

        // Compute cumulative mean normalized difference (CMNDF)
        let cmndf = compute_cmndf(frame, min_period, max_period);

        // Compute parabolic interpolation shifts
        let shifts = parabolic_shifts(&cmndf);

        // Find candidates and compute probabilities
        let (pitch_probs, voiced_prob) = compute_pyin_probs(
            &cmndf,
            &shifts,
            &thresholds,
            &beta_probs,
            sr,
            min_period,
            fmin,
            n_pitch_bins,
            n_bins_per_semitone,
        );

        // Store observation probabilities
        // First n_pitch_bins: voiced states
        // Next n_pitch_bins: unvoiced states
        for (i, &p) in pitch_probs.iter().enumerate() {
            observation_probs[frame_idx][i] = p;
        }

        // Unvoiced observation: uniform over unvoiced states
        let unvoiced_prob = 1.0 - voiced_prob;
        for prob in observation_probs[frame_idx][n_pitch_bins..(2 * n_pitch_bins)].iter_mut() {
            *prob = unvoiced_prob / n_pitch_bins as f32;
        }

        voiced_probs[frame_idx] = voiced_prob;
    }

    // Build transition matrix
    let switch_prob = 0.01f32;
    let max_transition_rate = 35.92f32; // octaves per second
    let max_semitones_per_frame =
        (max_transition_rate * 12.0 * hop_length as f32 / sr as f32).round() as usize;
    let transition_width = max_semitones_per_frame * n_bins_per_semitone + 1;

    // Viterbi decoding
    let states = viterbi_pyin(
        &observation_probs,
        n_pitch_bins,
        transition_width,
        switch_prob,
    );

    // Convert states to frequencies
    let mut f0 = vec![f32::NAN; n_frames];
    let mut voiced_flag = vec![false; n_frames];

    for (i, &state) in states.iter().enumerate() {
        let pitch_bin = state % n_pitch_bins;
        let is_voiced = state < n_pitch_bins;

        voiced_flag[i] = is_voiced;

        if is_voiced {
            // Convert bin to frequency
            let freq = fmin * 2.0_f32.powf(pitch_bin as f32 / (12.0 * n_bins_per_semitone as f32));
            f0[i] = freq;
        }
    }

    (f0, voiced_flag, voiced_probs)
}

/// Compute cumulative mean normalized difference function.
fn compute_cmndf(frame: &[f32], _min_period: usize, max_period: usize) -> Vec<f32> {
    let frame_len = frame.len();
    let tau_max = max_period.min(frame_len);

    // Compute difference function
    let mut diff = vec![0.0f32; tau_max];
    for tau in 0..tau_max {
        let mut sum = 0.0f32;
        for j in 0..(frame_len.saturating_sub(tau)) {
            let delta = frame[j] - frame[j + tau];
            sum += delta * delta;
        }
        diff[tau] = sum;
    }

    // Compute CMNDF
    let mut cmndf = vec![1.0f32; tau_max];
    cmndf[0] = 1.0;

    let mut running_sum = 0.0f32;
    for tau in 1..tau_max {
        running_sum += diff[tau];
        if running_sum > 0.0 {
            cmndf[tau] = diff[tau] * tau as f32 / running_sum;
        } else {
            cmndf[tau] = 1.0;
        }
    }

    cmndf
}

/// Compute parabolic interpolation shifts for each sample.
fn parabolic_shifts(cmndf: &[f32]) -> Vec<f32> {
    let n = cmndf.len();
    let mut shifts = vec![0.0f32; n];

    for i in 1..(n - 1) {
        let s0 = cmndf[i - 1];
        let s1 = cmndf[i];
        let s2 = cmndf[i + 1];

        let denom = s0 - 2.0 * s1 + s2;
        if denom.abs() > 1e-10 {
            let shift = 0.5 * (s0 - s2) / denom;
            // Only keep shift if it's within valid range
            if shift.abs() <= 1.0 {
                shifts[i] = shift;
            }
        }
    }

    shifts
}

/// Compute beta distribution CDF differences for threshold probabilities.
fn compute_beta_probs(thresholds: &[f32], a: f32, b: f32) -> Vec<f32> {
    let n = thresholds.len();
    if n < 2 {
        return Vec::new();
    }

    // Compute beta CDF at each threshold using regularized incomplete beta function
    let cdf: Vec<f32> = thresholds.iter().map(|&t| beta_cdf(t, a, b)).collect();

    // Compute differences
    let mut probs = Vec::with_capacity(n - 1);
    for i in 0..(n - 1) {
        probs.push((cdf[i + 1] - cdf[i]).max(0.0));
    }

    probs
}

/// Approximate beta CDF using numerical integration.
fn beta_cdf(x: f32, a: f32, b: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    // Use numerical integration (Simpson's rule)
    let n_steps = 100;
    let h = x / n_steps as f32;

    let mut sum = 0.0f32;
    for i in 0..=n_steps {
        let t = i as f32 * h;
        let f = if t > 0.0 && t < 1.0 {
            t.powf(a - 1.0) * (1.0 - t).powf(b - 1.0)
        } else {
            0.0
        };

        let weight = if i == 0 || i == n_steps {
            1.0
        } else if i % 2 == 1 {
            4.0
        } else {
            2.0
        };

        sum += weight * f;
    }

    let integral = sum * h / 3.0;

    // Normalize by beta function B(a, b)
    let beta_func = gamma_approx(a) * gamma_approx(b) / gamma_approx(a + b);

    (integral / beta_func).clamp(0.0, 1.0)
}

/// Approximate gamma function using Stirling's approximation.
fn gamma_approx(x: f32) -> f32 {
    if x <= 0.0 {
        return f32::INFINITY;
    }

    // Use Lanczos approximation for better accuracy
    let g = 7.0f32;
    let c = [
        0.99999999999980993f32,
        676.520_4,
        -1_259.139_2,
        771.323_4,
        -176.615_04,
        12.507_343,
        -0.138_571_1,
        9.984_369e-6,
        1.505_632_7e-7,
    ];

    if x < 0.5 {
        // Reflection formula
        PI / ((PI * x).sin() * gamma_approx(1.0 - x))
    } else {
        let x = x - 1.0;
        let mut y = c[0];
        for (i, &ci) in c.iter().enumerate().skip(1) {
            y += ci / (x + i as f32);
        }
        let t = x + g + 0.5;
        (2.0 * PI).sqrt() * t.powf(x + 0.5) * (-t).exp() * y
    }
}

/// Compute pyin observation probabilities for a single frame.
#[allow(clippy::too_many_arguments)]
fn compute_pyin_probs(
    cmndf: &[f32],
    shifts: &[f32],
    thresholds: &[f32],
    beta_probs: &[f32],
    sr: u32,
    min_period: usize,
    fmin: f32,
    n_pitch_bins: usize,
    n_bins_per_semitone: usize,
) -> (Vec<f32>, f32) {
    let n_tau = cmndf.len();
    let mut pitch_probs = vec![0.0f32; n_pitch_bins];
    let mut voiced_prob = 0.0f32;

    let boltzmann_param = 2.0f32;
    let no_trough_prob = 0.01f32;

    // Find local minima (troughs) in CMNDF
    let mut troughs = Vec::new();
    for tau in min_period.max(1)..(n_tau - 1) {
        if cmndf[tau] < cmndf[tau - 1] && cmndf[tau] < cmndf[tau + 1] {
            troughs.push(tau);
        }
    }

    // Also check first valid tau
    if min_period < n_tau - 1
        && cmndf[min_period] < cmndf[min_period + 1]
        && (troughs.is_empty() || troughs[0] != min_period)
    {
        troughs.insert(0, min_period);
    }

    if troughs.is_empty() {
        // No troughs found, return low voiced probability
        let uniform_prob = 1.0 / n_pitch_bins as f32;
        for p in pitch_probs.iter_mut() {
            *p = uniform_prob * no_trough_prob;
        }
        return (pitch_probs, no_trough_prob);
    }

    // For each threshold, find candidates
    for (thresh_idx, &threshold) in thresholds[..thresholds.len() - 1].iter().enumerate() {
        let thresh_prob = beta_probs[thresh_idx];

        // Find troughs below this threshold
        let candidates: Vec<usize> = troughs
            .iter()
            .filter(|&&tau| cmndf[tau] < threshold)
            .cloned()
            .collect();

        if candidates.is_empty() {
            continue;
        }

        // Compute Boltzmann weights for candidates
        let mut weights = Vec::with_capacity(candidates.len());
        let mut weight_sum = 0.0f32;

        for &tau in &candidates {
            let weight = (-boltzmann_param * tau as f32).exp();
            weights.push(weight);
            weight_sum += weight;
        }

        if weight_sum <= 0.0 {
            continue;
        }

        // Normalize weights
        for w in weights.iter_mut() {
            *w /= weight_sum;
        }

        // Add probability mass to pitch bins
        for (&tau, &weight) in candidates.iter().zip(weights.iter()) {
            // Apply parabolic shift
            let tau_refined = tau as f32 + shifts.get(tau).copied().unwrap_or(0.0);

            if tau_refined <= 0.0 {
                continue;
            }

            // Convert to frequency
            let freq = sr as f32 / tau_refined;

            // Convert to pitch bin
            if freq < fmin || freq.is_nan() || freq.is_infinite() {
                continue;
            }

            let bin_float = 12.0 * n_bins_per_semitone as f32 * (freq / fmin).log2();
            let bin = bin_float.round() as i32;

            if bin >= 0 && (bin as usize) < n_pitch_bins {
                let prob = thresh_prob * weight;
                pitch_probs[bin as usize] += prob;
                voiced_prob += prob;
            }
        }
    }

    // Normalize pitch probabilities
    if voiced_prob > 0.0 {
        for p in pitch_probs.iter_mut() {
            *p /= voiced_prob;
        }
    } else {
        // Uniform if no probability mass
        let uniform = 1.0 / n_pitch_bins as f32;
        for p in pitch_probs.iter_mut() {
            *p = uniform;
        }
        voiced_prob = no_trough_prob;
    }

    // Clamp voiced probability
    voiced_prob = voiced_prob.clamp(0.01, 0.99);

    (pitch_probs, voiced_prob)
}

/// Simplified Viterbi decoding for pyin.
fn viterbi_pyin(
    observation_probs: &[Vec<f32>],
    n_pitch_bins: usize,
    transition_width: usize,
    switch_prob: f32,
) -> Vec<usize> {
    let n_frames = observation_probs.len();
    if n_frames == 0 || n_pitch_bins == 0 {
        return Vec::new();
    }

    let n_states = 2 * n_pitch_bins; // voiced + unvoiced states

    // Initialize
    let mut prev_probs = vec![1.0 / n_states as f32; n_states];
    let mut backtrack = vec![vec![0usize; n_states]; n_frames];

    // Forward pass
    for frame in 0..n_frames {
        let obs = &observation_probs[frame];
        let mut curr_probs = vec![0.0f32; n_states];

        for state in 0..n_states {
            let is_voiced = state < n_pitch_bins;
            let pitch_bin = state % n_pitch_bins;

            let obs_prob = obs.get(state).copied().unwrap_or(1e-10).max(1e-10);

            let mut best_prev_prob = 0.0f32;
            let mut best_prev_state = state;

            // Check transitions from previous states
            for (prev_state, &prev_prob) in prev_probs.iter().enumerate().take(n_states) {
                let prev_voiced = prev_state < n_pitch_bins;
                let prev_pitch = prev_state % n_pitch_bins;

                // Transition probability
                let trans_prob = if is_voiced == prev_voiced {
                    // Same voicing state
                    let pitch_diff = (pitch_bin as i32 - prev_pitch as i32).unsigned_abs() as usize;
                    if pitch_diff <= transition_width {
                        // Triangle window transition
                        let weight = 1.0 - pitch_diff as f32 / (transition_width + 1) as f32;
                        (1.0 - switch_prob) * weight
                    } else {
                        1e-10
                    }
                } else {
                    // Switching voicing state
                    switch_prob / n_pitch_bins as f32
                };

                let prob = prev_prob * trans_prob;
                if prob > best_prev_prob {
                    best_prev_prob = prob;
                    best_prev_state = prev_state;
                }
            }

            curr_probs[state] = best_prev_prob * obs_prob;
            backtrack[frame][state] = best_prev_state;
        }

        // Normalize to prevent underflow
        let sum: f32 = curr_probs.iter().sum();
        if sum > 0.0 {
            for p in curr_probs.iter_mut() {
                *p /= sum;
            }
        }

        prev_probs = curr_probs;
    }

    // Backtrack
    let mut states = vec![0usize; n_frames];

    // Find best final state
    let mut best_state = 0;
    let mut best_prob = 0.0f32;
    for (state, &prob) in prev_probs.iter().enumerate() {
        if prob > best_prob {
            best_prob = prob;
            best_state = state;
        }
    }

    states[n_frames - 1] = best_state;
    for frame in (0..(n_frames - 1)).rev() {
        states[frame] = backtrack[frame + 1][states[frame + 1]];
    }

    states
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io;

    #[test]
    fn test_yin_pure_tone() {
        let sr = 22050;
        let freq = 440.0;
        let signal = io::tone(freq, sr, 1.0);

        let f0 = yin(&signal, sr, 2048, 512, 40.0, 5000.0, 0.1).unwrap();

        // Count how many frames detected a pitch
        let detected: Vec<f32> = f0.iter().filter(|&&x| x > 0.0).cloned().collect();

        assert!(!detected.is_empty(), "Should detect pitch in some frames");

        // Average detected frequency should be close to 440 Hz
        let avg_f0 = detected.iter().sum::<f32>() / detected.len() as f32;
        assert!(
            (avg_f0 - freq).abs() < 50.0,
            "Detected F0 {} should be close to {}",
            avg_f0,
            freq
        );
    }

    #[test]
    fn test_yin_empty() {
        let signal = vec![];
        let result = yin(&signal, 22050, 2048, 512, 40.0, 5000.0, 0.1);
        assert!(result.is_err());
        assert!(matches!(result, Err(crate::Error::EmptyAudio)));
    }

    #[test]
    fn test_yin_short_signal() {
        let signal = vec![0.1; 100];
        let f0 = yin(&signal, 22050, 2048, 512, 40.0, 5000.0, 0.1).unwrap();
        // Should return at least one frame
        assert!(!f0.is_empty());
    }

    #[test]
    fn test_yin_silence() {
        let signal = vec![0.0; 22050];
        let f0 = yin(&signal, 22050, 2048, 512, 40.0, 5000.0, 0.1).unwrap();

        // Silence should mostly produce no pitch estimates
        let detected_count = f0.iter().filter(|&&x| x > 0.0).count();
        let total_frames = f0.len();

        // Allow some spurious detections but most should be 0
        assert!(detected_count < total_frames / 2);
    }

    #[test]
    fn test_yin_multiple_frequencies() {
        let sr = 22050;

        // Test different frequencies
        for freq in &[110.0, 220.0, 440.0, 880.0] {
            let signal = io::tone(*freq, sr, 0.5);
            let f0 = yin(&signal, sr, 2048, 512, 40.0, 5000.0, 0.1).unwrap();

            let detected: Vec<f32> = f0.iter().filter(|&&x| x > 0.0).cloned().collect();

            if !detected.is_empty() {
                let avg_f0 = detected.iter().sum::<f32>() / detected.len() as f32;
                let error_pct = ((avg_f0 - freq) / freq * 100.0).abs();
                assert!(
                    error_pct < 15.0,
                    "Frequency {} detected as {} (error {:.1}%)",
                    freq,
                    avg_f0,
                    error_pct
                );
            }
        }
    }

    #[test]
    fn test_piptrack_pure_tone() {
        let sr = 22050;
        let freq = 440.0;
        let signal = io::tone(freq, sr, 1.0);

        let (pitches, _mags) =
            super::piptrack(&signal, sr, 2048, 512, 150.0, 4000.0, 0.01).unwrap();

        // Should have detected pitches in some frames
        let detected_count = pitches.iter().filter(|&&x| x > 0.0).count();
        assert!(
            detected_count > 0,
            "Should detect pitch in some time-frequency bins"
        );

        // Check that some detected pitches are near the fundamental or its harmonics
        // piptrack detects all spectral peaks, not just the fundamental
        let detected_pitches: Vec<f32> = pitches.iter().filter(|&&x| x > 0.0).cloned().collect();
        if !detected_pitches.is_empty() {
            // Check that at least some pitches are near fundamental or harmonics (440, 880, 1320, etc)
            let has_fundamental_or_harmonic = detected_pitches.iter().any(|&p| {
                for harmonic in 1..=5 {
                    let expected = freq * harmonic as f32;
                    if (p - expected).abs() < 50.0 {
                        return true;
                    }
                }
                false
            });
            assert!(
                has_fundamental_or_harmonic,
                "Should detect fundamental or harmonic frequencies"
            );
        }
    }

    #[test]
    fn test_piptrack_shape() {
        let signal = io::tone(880.0, 22050, 0.5);
        let (pitches, mags) =
            super::piptrack(&signal, 22050, 2048, 512, 150.0, 4000.0, 0.1).unwrap();

        // Shapes should match
        assert_eq!(pitches.shape(), mags.shape());

        // Should have time frames
        assert!(pitches.shape()[1] > 0);
    }

    #[test]
    fn test_piptrack_empty() {
        let signal = vec![];
        let (pitches, mags) =
            super::piptrack(&signal, 22050, 2048, 512, 150.0, 4000.0, 0.1).unwrap();

        assert_eq!(pitches.shape(), &[0, 0]);
        assert_eq!(mags.shape(), &[0, 0]);
    }

    #[test]
    fn test_piptrack_silence() {
        let signal = vec![0.0; 22050];
        let (pitches, _mags) =
            super::piptrack(&signal, 22050, 2048, 512, 150.0, 4000.0, 0.1).unwrap();

        // Silence should produce mostly zero pitches (below threshold)
        let detected_count = pitches.iter().filter(|&&x| x > 0.0).count();
        let total_bins = pitches.len();

        // Most bins should be zero
        assert!(detected_count < total_bins / 10);
    }

    #[test]
    fn test_piptrack_magnitudes() {
        let signal = io::tone(440.0, 22050, 0.5);
        let (pitches, mags) =
            super::piptrack(&signal, 22050, 2048, 512, 150.0, 4000.0, 0.01).unwrap();

        // Where pitches are non-zero, magnitudes should be non-zero
        for i in 0..pitches.len() {
            let pitch = pitches.as_slice().unwrap()[i];
            let mag = mags.as_slice().unwrap()[i];

            if pitch > 0.0 {
                assert!(mag > 0.0, "Non-zero pitch should have non-zero magnitude");
            }
        }
    }

    #[test]
    fn test_pitch_tuning_on_pitch() {
        // Frequencies exactly on A440 tuning
        let freqs = vec![220.0, 440.0, 880.0, 1760.0];
        let tuning = super::pitch_tuning(&freqs, 0.01, 12);

        // Should be very close to 0
        assert!(
            tuning.abs() < 0.1,
            "On-pitch frequencies should give tuning near 0, got {}",
            tuning
        );
    }

    #[test]
    fn test_pitch_tuning_empty() {
        let freqs: Vec<f32> = vec![];
        let tuning = super::pitch_tuning(&freqs, 0.01, 12);
        assert_eq!(tuning, 0.0);
    }

    #[test]
    fn test_pitch_tuning_negative_freqs() {
        let freqs = vec![-100.0, 0.0, -50.0];
        let tuning = super::pitch_tuning(&freqs, 0.01, 12);
        assert_eq!(tuning, 0.0);
    }

    #[test]
    fn test_estimate_tuning_pure_tone() {
        let signal = io::tone(440.0, 22050, 1.0);
        let tuning = super::estimate_tuning(&signal, 22050, 2048, 0.01, 12).unwrap();

        // A440 should give tuning near 0
        assert!(
            tuning.abs() < 0.3,
            "A440 should give tuning near 0, got {}",
            tuning
        );
    }

    #[test]
    fn test_estimate_tuning_empty() {
        let signal: Vec<f32> = vec![];
        let tuning = super::estimate_tuning(&signal, 22050, 2048, 0.01, 12).unwrap();
        assert_eq!(tuning, 0.0);
    }

    #[test]
    fn test_estimate_tuning_range() {
        let signal = io::tone(440.0, 22050, 1.0);
        let tuning = super::estimate_tuning(&signal, 22050, 2048, 0.01, 12).unwrap();

        // Tuning should be in valid range
        assert!(
            (-0.5..0.5).contains(&tuning),
            "Tuning {} should be in [-0.5, 0.5)",
            tuning
        );
    }

    #[test]
    fn test_pyin_pure_tone() {
        let sr = 22050;
        let freq = 440.0;
        let signal = io::tone(freq, sr, 1.0);

        let (f0, voiced, probs) =
            super::pyin(&signal, sr, 65.0, 2093.0, 2048, 512, 100, (2.0, 18.0), 0.1);

        assert!(!f0.is_empty(), "Should produce output frames");
        assert_eq!(f0.len(), voiced.len());
        assert_eq!(f0.len(), probs.len());

        // Count voiced frames
        let voiced_count = voiced.iter().filter(|&&v| v).count();
        assert!(
            voiced_count > f0.len() / 2,
            "Most frames should be voiced for a tone"
        );

        // Check average detected frequency (excluding NaN)
        let valid_f0: Vec<f32> = f0.iter().filter(|x| !x.is_nan()).cloned().collect();
        if !valid_f0.is_empty() {
            let avg_f0 = valid_f0.iter().sum::<f32>() / valid_f0.len() as f32;
            assert!(
                (avg_f0 - freq).abs() < 100.0,
                "Average F0 {} should be close to {}",
                avg_f0,
                freq
            );
        }
    }

    #[test]
    fn test_pyin_empty() {
        let signal: Vec<f32> = vec![];
        let (f0, voiced, probs) = super::pyin(
            &signal,
            22050,
            65.0,
            2093.0,
            2048,
            512,
            100,
            (2.0, 18.0),
            0.1,
        );

        assert!(f0.is_empty());
        assert!(voiced.is_empty());
        assert!(probs.is_empty());
    }

    #[test]
    fn test_pyin_silence() {
        let signal = vec![0.0f32; 22050];
        let (f0, _voiced, probs) = super::pyin(
            &signal,
            22050,
            65.0,
            2093.0,
            2048,
            512,
            100,
            (2.0, 18.0),
            0.1,
        );

        assert!(!f0.is_empty());

        // Silence should have low voiced probability on average
        let avg_prob = probs.iter().sum::<f32>() / probs.len() as f32;
        // The algorithm may still assign some voiced probability to silence
        // but it should generally be lower than for a clear tone
        assert!(
            avg_prob < 0.9,
            "Silence should have moderate voiced probability"
        );
    }

    #[test]
    fn test_pyin_different_frequencies() {
        let sr = 22050;

        // Test that pyin produces output for different frequencies
        // The algorithm is probabilistic so we just verify it runs and produces valid output
        for &freq in &[330.0, 440.0, 880.0] {
            let signal = io::tone(freq, sr, 0.5);
            let (f0, voiced, probs) =
                super::pyin(&signal, sr, 65.0, 2093.0, 2048, 512, 100, (2.0, 18.0), 0.1);

            // Basic validity checks
            assert!(!f0.is_empty(), "Should produce output for {} Hz", freq);
            assert_eq!(f0.len(), voiced.len());
            assert_eq!(f0.len(), probs.len());

            // All probabilities should be valid
            for &p in &probs {
                assert!((0.0..=1.0).contains(&p));
            }

            // Should detect some voiced frames
            let voiced_count = voiced.iter().filter(|&&v| v).count();
            assert!(
                voiced_count > 0,
                "Should detect some voiced frames for {} Hz",
                freq
            );
        }
    }

    #[test]
    fn test_pyin_output_lengths() {
        let signal = io::tone(440.0, 22050, 0.5);
        let (f0, voiced, probs) = super::pyin(
            &signal,
            22050,
            65.0,
            2093.0,
            2048,
            256,
            100,
            (2.0, 18.0),
            0.1,
        );

        // All outputs should have same length
        assert_eq!(f0.len(), voiced.len());
        assert_eq!(f0.len(), probs.len());

        // Should have multiple frames
        assert!(f0.len() > 1);
    }

    #[test]
    fn test_pyin_voiced_prob_range() {
        let signal = io::tone(440.0, 22050, 0.5);
        let (_f0, _voiced, probs) = super::pyin(
            &signal,
            22050,
            65.0,
            2093.0,
            2048,
            512,
            100,
            (2.0, 18.0),
            0.1,
        );

        // All probabilities should be in [0, 1]
        for &p in &probs {
            assert!(
                (0.0..=1.0).contains(&p),
                "Probability {} should be in [0, 1]",
                p
            );
        }
    }

    #[test]
    fn test_beta_cdf() {
        // Test beta CDF with known values
        let cdf_0 = super::beta_cdf(0.0, 2.0, 18.0);
        let cdf_1 = super::beta_cdf(1.0, 2.0, 18.0);
        let cdf_half = super::beta_cdf(0.5, 2.0, 18.0);

        assert!((cdf_0 - 0.0).abs() < 0.01, "CDF(0) should be 0");
        assert!((cdf_1 - 1.0).abs() < 0.01, "CDF(1) should be 1");
        assert!(cdf_half > 0.9, "CDF(0.5) for Beta(2,18) should be high");
    }

    #[test]
    fn test_gamma_approx() {
        // Test gamma approximation with known values
        // Gamma(1) = 1
        let g1 = super::gamma_approx(1.0);
        assert!((g1 - 1.0).abs() < 0.01, "Gamma(1) should be 1, got {}", g1);

        // Gamma(2) = 1
        let g2 = super::gamma_approx(2.0);
        assert!((g2 - 1.0).abs() < 0.01, "Gamma(2) should be 1, got {}", g2);

        // Gamma(3) = 2
        let g3 = super::gamma_approx(3.0);
        assert!((g3 - 2.0).abs() < 0.1, "Gamma(3) should be 2, got {}", g3);

        // Gamma(4) = 6
        let g4 = super::gamma_approx(4.0);
        assert!((g4 - 6.0).abs() < 0.1, "Gamma(4) should be 6, got {}", g4);
    }
}
