use crate::fft::FftPlan;
use crate::window;
use ndarray::Array2;
use num_complex::Complex32;

#[derive(Debug, Clone)]
pub struct StftConfig {
    pub n_fft: usize,
    pub hop_length: usize,
    pub win_length: usize,
    pub center: bool,
    pub window: Vec<f32>,
    pub pad_mode: PadMode,
}

#[derive(Debug, Clone, Copy)]
pub enum PadMode {
    Constant,
    Reflect,
}

impl Default for StftConfig {
    fn default() -> Self {
        let n_fft = 2048;
        let win_length = 2048;
        Self {
            n_fft,
            hop_length: n_fft / 4,
            win_length,
            center: true,
            window: window::hann(win_length),
            pad_mode: PadMode::Constant,
        }
    }
}

fn pad_window(window: &[f32], n_fft: usize) -> Vec<f32> {
    if window.len() == n_fft {
        return window.to_vec();
    }
    let mut padded = vec![0.0f32; n_fft];
    let start = (n_fft - window.len()) / 2;
    padded[start..start + window.len()].copy_from_slice(window);
    padded
}

fn reflect_index(mut idx: isize, len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    if len == 1 {
        return 0;
    }
    let last = len as isize - 1;
    while idx < 0 || idx > last {
        if idx < 0 {
            idx = -idx;
        }
        if idx > last {
            idx = 2 * last - idx;
        }
    }
    idx as usize
}

fn pad_center(y: &[f32], n_fft: usize, center: bool, pad_mode: PadMode) -> Vec<f32> {
    if !center {
        return y.to_vec();
    }
    let pad = n_fft / 2;
    let mut out = vec![0.0f32; y.len() + 2 * pad];
    if y.is_empty() {
        return out;
    }
    match pad_mode {
        PadMode::Constant => {
            out[pad..pad + y.len()].copy_from_slice(y);
        }
        PadMode::Reflect => {
            for (i, out_val) in out.iter_mut().enumerate() {
                let src_idx = i as isize - pad as isize;
                let idx = reflect_index(src_idx, y.len());
                *out_val = y[idx];
            }
        }
    }
    out
}

#[inline]
fn compute_frame(
    frame: usize,
    padded: &[f32],
    window: &[f32],
    fft: &FftPlan,
    hop_length: usize,
    n_fft: usize,
    n_freq: usize,
) -> Vec<Complex32> {
    let start = frame * hop_length;
    let mut buffer = vec![Complex32::new(0.0, 0.0); n_fft];
    for i in 0..n_fft {
        let sample = padded.get(start + i).copied().unwrap_or(0.0);
        buffer[i].re = sample * window[i];
    }
    fft.forward(&mut buffer);
    buffer.truncate(n_freq);
    buffer
}

/// Compute the Short-Time Fourier Transform (STFT).
///
/// # Arguments
/// * `y` - Input audio signal
/// * `config` - STFT configuration (FFT size, hop length, window, etc.)
///
/// # Returns
/// Complex STFT matrix of shape (n_freq, n_frames) where n_freq = n_fft/2 + 1
///
/// # Errors
/// Returns an error if the audio is invalid or if n_fft/hop_length is zero.
pub fn stft(y: &[f32], config: &StftConfig) -> crate::Result<Array2<Complex32>> {
    crate::utils::valid_audio(y)?;
    if config.n_fft == 0 {
        return Err(crate::Error::InvalidSize {
            name: "n_fft",
            value: 0,
            reason: "must be > 0",
        });
    }
    if config.hop_length == 0 {
        return Err(crate::Error::InvalidSize {
            name: "hop_length",
            value: 0,
            reason: "must be > 0",
        });
    }

    let window = pad_window(&config.window, config.n_fft);
    let padded = pad_center(y, config.n_fft, config.center, config.pad_mode);
    let n_frames = if padded.len() < config.n_fft {
        0
    } else {
        (padded.len() - config.n_fft) / config.hop_length + 1
    };

    let n_freq = config.n_fft / 2 + 1;
    let fft = FftPlan::new(config.n_fft);

    let frame_results: Vec<Vec<Complex32>> = {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            (0..n_frames)
                .into_par_iter()
                .map(|frame| {
                    compute_frame(
                        frame,
                        &padded,
                        &window,
                        &fft,
                        config.hop_length,
                        config.n_fft,
                        n_freq,
                    )
                })
                .collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            (0..n_frames)
                .map(|frame| {
                    compute_frame(
                        frame,
                        &padded,
                        &window,
                        &fft,
                        config.hop_length,
                        config.n_fft,
                        n_freq,
                    )
                })
                .collect()
        }
    };

    let mut stft_matrix = Array2::<Complex32>::zeros((n_freq, n_frames));
    for (frame, result) in frame_results.iter().enumerate() {
        for (f, &val) in result.iter().enumerate() {
            stft_matrix[(f, frame)] = val;
        }
    }

    Ok(stft_matrix)
}

/// Compute the Inverse Short-Time Fourier Transform (ISTFT).
///
/// Reconstructs a time-domain signal from its STFT representation using
/// overlap-add synthesis.
///
/// # Arguments
/// * `stft_matrix` - Complex STFT matrix (n_freq x n_frames)
/// * `config` - STFT configuration (must match the forward STFT)
/// * `length` - Optional output signal length (truncates if provided)
///
/// # Returns
/// Reconstructed time-domain signal
///
/// # Errors
/// Returns an error if the STFT matrix is empty.
pub fn istft(
    stft_matrix: &Array2<Complex32>,
    config: &StftConfig,
    length: Option<usize>,
) -> crate::Result<Vec<f32>> {
    let n_freq = stft_matrix.shape().first().copied().unwrap_or(0);
    let n_frames = stft_matrix.shape().get(1).copied().unwrap_or(0);
    if n_freq == 0 || n_frames == 0 {
        return Err(crate::Error::InvalidSize {
            name: "stft_matrix",
            value: 0,
            reason: "STFT matrix must be non-empty",
        });
    }

    let n_fft = (n_freq - 1) * 2;
    let window = pad_window(&config.window, n_fft);

    let mut y = vec![0.0f32; n_frames * config.hop_length + n_fft];
    let mut window_sums = vec![0.0f32; y.len()];
    let fft = FftPlan::new(n_fft);

    for frame in 0..n_frames {
        let start = frame * config.hop_length;
        let mut buffer = vec![Complex32::new(0.0, 0.0); n_fft];

        for f in 0..n_freq {
            buffer[f] = stft_matrix[(f, frame)];
        }
        for f in 1..(n_freq - 1) {
            buffer[n_fft - f] = stft_matrix[(f, frame)].conj();
        }

        fft.inverse(&mut buffer);

        for i in 0..n_fft {
            let w = window[i];
            let sample = buffer[i].re * w;
            let idx = start + i;
            y[idx] += sample;
            window_sums[idx] += w * w;
        }
    }

    for i in 0..y.len() {
        if window_sums[i] > 1e-8 {
            y[i] /= window_sums[i];
        }
    }

    let mut out = if config.center {
        let pad = n_fft / 2;
        if y.len() > 2 * pad {
            y[pad..y.len() - pad].to_vec()
        } else {
            y
        }
    } else {
        y
    };

    if let Some(len) = length {
        out.truncate(len);
    }
    Ok(out)
}

/// Separate a complex STFT matrix into magnitude and phase.
///
/// # Arguments
/// * `stft_matrix` - Complex STFT matrix
///
/// # Returns
/// Tuple of (magnitude, phase) where magnitude is real-valued and
/// phase is unit-magnitude complex values.
pub fn magphase(stft_matrix: &Array2<Complex32>) -> (Array2<f32>, Array2<Complex32>) {
    let shape = stft_matrix.raw_dim();
    let mut magnitude = Array2::<f32>::zeros(shape);
    let mut phase = Array2::<Complex32>::zeros(shape);

    for ((idx, val), mag) in stft_matrix.indexed_iter().zip(magnitude.iter_mut()) {
        let v = *val;
        let m = v.norm();
        *mag = m;
        if m > 0.0 {
            phase[idx] = v / m;
        } else {
            phase[idx] = Complex32::new(0.0, 0.0);
        }
    }

    (magnitude, phase)
}

/// Convert power spectrogram to dB scale.
/// S_db = 10 * log10(S / ref)
pub fn power_to_db(
    power: &Array2<f32>,
    ref_power: f32,
    amin: f32,
    top_db: Option<f32>,
) -> Array2<f32> {
    let shape = power.raw_dim();
    let mut db = Array2::<f32>::zeros(shape);
    let log_ref = 10.0 * ref_power.max(amin).log10();

    for (idx, &p) in power.indexed_iter() {
        let log_spec = 10.0 * p.max(amin).log10();
        db[idx] = log_spec - log_ref;
    }

    if let Some(top) = top_db {
        let max_db = db.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let threshold = max_db - top;
        db.mapv_inplace(|v| v.max(threshold));
    }

    db
}

/// Convert amplitude spectrogram to dB scale.
/// S_db = 20 * log10(S / ref)
pub fn amplitude_to_db(
    amplitude: &Array2<f32>,
    ref_amplitude: f32,
    amin: f32,
    top_db: Option<f32>,
) -> Array2<f32> {
    let shape = amplitude.raw_dim();
    let mut db = Array2::<f32>::zeros(shape);
    let log_ref = 20.0 * ref_amplitude.max(amin).log10();

    for (idx, &a) in amplitude.indexed_iter() {
        let log_spec = 20.0 * a.max(amin).log10();
        db[idx] = log_spec - log_ref;
    }

    if let Some(top) = top_db {
        let max_db = db.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let threshold = max_db - top;
        db.mapv_inplace(|v| v.max(threshold));
    }

    db
}

/// Convert dB scale to power.
/// S = ref * 10^(S_db / 10)
pub fn db_to_power(db: &Array2<f32>, ref_power: f32) -> Array2<f32> {
    db.mapv(|v| ref_power * 10.0f32.powf(v / 10.0))
}

/// Convert dB scale to amplitude.
/// S = ref * 10^(S_db / 20)
pub fn db_to_amplitude(db: &Array2<f32>, ref_amplitude: f32) -> Array2<f32> {
    db.mapv(|v| ref_amplitude * 10.0f32.powf(v / 20.0))
}

/// Per-Channel Energy Normalization (PCEN).
///
/// PCEN applies automatic gain control (AGC) filtering followed by
/// dynamic range compression. It's designed to be robust to loudness
/// variations and background noise.
///
/// # Arguments
/// * `spectrogram` - Input magnitude spectrogram (n_freq × n_frames)
/// * `sr` - Sample rate
/// * `hop_length` - STFT hop length
/// * `gain` - Gain normalization exponent (typical: 0.8)
/// * `bias` - Bias term to prevent division by zero (typical: 2.0)
/// * `power` - Compression exponent (typical: 0.25)
/// * `time_constant` - AGC time constant in seconds (typical: 0.4)
/// * `eps` - Small constant for numerical stability (typical: 1e-6)
///
/// # Returns
/// PCEN-normalized spectrogram with same shape as input
///
/// # Formula
/// PCEN(t,f) = ((S(t,f) / (ε + M(t,f))^gain + δ)^power - δ^power)
/// where M is the AGC filter output
///
/// Reference: Wang et al. "Trainable Frontend For Robust and Far-Field Keyword Spotting" (2017)
#[allow(clippy::too_many_arguments)]
pub fn pcen(
    spectrogram: &Array2<f32>,
    sr: u32,
    hop_length: usize,
    gain: f32,
    bias: f32,
    power: f32,
    time_constant: f32,
    eps: f32,
) -> crate::Result<Array2<f32>> {
    let shape = spectrogram.shape();
    let (n_freq, n_frames) = (shape[0], shape[1]);

    if n_frames == 0 {
        return Err(crate::Error::InvalidSize {
            name: "spectrogram",
            value: 0,
            reason: "spectrogram must have at least one frame",
        });
    }

    // Compute AGC filter coefficient
    // α = exp(-1 / (time_constant * sr / hop_length))
    let frames_per_second = sr as f32 / hop_length as f32;
    let alpha = (-1.0 / (time_constant * frames_per_second)).exp();

    // Initialize AGC filter state (smoothed energy)
    let mut smoothed = Array2::<f32>::zeros((n_freq, n_frames));

    // Apply AGC filter (one-pole IIR lowpass per frequency bin)
    for freq in 0..n_freq {
        let mut state = spectrogram[(freq, 0)];
        smoothed[(freq, 0)] = state;

        for frame in 1..n_frames {
            state = alpha * state + (1.0 - alpha) * spectrogram[(freq, frame)];
            smoothed[(freq, frame)] = state;
        }
    }

    // Apply PCEN formula
    let mut result = Array2::<f32>::zeros((n_freq, n_frames));
    for freq in 0..n_freq {
        for frame in 0..n_frames {
            let s = spectrogram[(freq, frame)];
            let m = smoothed[(freq, frame)];

            // Adaptive gain control
            let normalized = s / (eps + m).powf(gain);

            // Dynamic range compression with bias
            let compressed = (normalized + bias).powf(power) - bias.powf(power);

            result[(freq, frame)] = compressed;
        }
    }

    Ok(result)
}

/// Frequency weighting curves for perceptual modeling.
///
/// These curves model human hearing sensitivity at different loudness levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightingType {
    /// A-weighting: Models human ear at ~40 phon (most common)
    A,
    /// B-weighting: Intermediate level, rarely used
    B,
    /// C-weighting: For loud sounds, nearly flat
    C,
    /// D-weighting: For aircraft noise
    D,
}

/// Compute perceptual frequency weighting in dB.
///
/// Returns frequency weighting values according to international standards
/// (IEC 61672). These curves model human hearing sensitivity and are widely
/// used in sound level meters and audio analysis.
///
/// # Arguments
/// * `frequencies` - Frequency values in Hz
/// * `weighting` - Weighting curve type (A, B, C, or D)
///
/// # Returns
/// Weighting values in dB (apply by adding to magnitude spectrum in dB)
///
/// # Example
/// ```
/// use giggle::spectrum::{perceptual_weighting, WeightingType};
///
/// let freqs = vec![100.0, 1000.0, 10000.0];
/// let weights = perceptual_weighting(&freqs, WeightingType::A);
/// assert_eq!(weights.len(), 3);
/// ```
pub fn perceptual_weighting(frequencies: &[f32], weighting: WeightingType) -> Vec<f32> {
    frequencies
        .iter()
        .map(|&f| {
            if f <= 0.0 {
                return f32::NEG_INFINITY;
            }

            match weighting {
                WeightingType::A => a_weighting(f),
                WeightingType::B => b_weighting(f),
                WeightingType::C => c_weighting(f),
                WeightingType::D => d_weighting(f),
            }
        })
        .collect()
}

/// A-weighting curve (IEC 61672).
fn a_weighting(f: f32) -> f32 {
    let f2 = f * f;
    let c1 = 12194.0f32.powi(2);
    let c2 = 20.6f32.powi(2);
    let c3 = 107.7f32.powi(2);
    let c4 = 737.9f32.powi(2);

    let numerator = c1 * f2 * f2;
    let denominator = (f2 + c2) * ((f2 + c3) * (f2 + c4)).sqrt() * (f2 + c1);

    let r_a = numerator / denominator;
    20.0 * r_a.log10() + 2.0
}

/// B-weighting curve (IEC 61672).
fn b_weighting(f: f32) -> f32 {
    let f2 = f * f;
    let c1 = 12194.0f32.powi(2);
    let c2 = 20.6f32.powi(2);
    let c3 = 158.5f32.powi(2);

    let numerator = c1 * f2 * f;
    let denominator = (f2 + c2) * (f2 + c3).sqrt() * (f2 + c1);

    let r_b = numerator / denominator;
    20.0 * r_b.log10() + 0.17
}

/// C-weighting curve (IEC 61672).
fn c_weighting(f: f32) -> f32 {
    let f2 = f * f;
    let c1 = 12194.0f32.powi(2);
    let c2 = 20.6f32.powi(2);

    let numerator = c1 * f2;
    let denominator = (f2 + c2) * (f2 + c1);

    let r_c = numerator / denominator;
    20.0 * r_c.log10() + 0.06
}

/// D-weighting curve (IEC 61672).
fn d_weighting(f: f32) -> f32 {
    let f2 = f * f;
    let h_f = ((1_037_918.5 - f2).powi(2) + 1_080_768.1 * f2)
        / ((9837328.0 - f2).powi(2) + 11723776.0 * f2);

    let r_d = (f / 6.896_689e-5) * (h_f / ((f2 + 79919.29) * (f2 + 1345600.0))).sqrt();

    20.0 * r_d.log10()
}

/// Compute reassigned spectrogram with improved time-frequency resolution.
///
/// Time-frequency reassignment moves each STFT point to its instantaneous
/// frequency and group delay, resulting in sharper spectral representation.
/// This is particularly useful for resolving closely-spaced components.
///
/// # Arguments
/// * `y` - Input audio signal
/// * `config` - STFT configuration
///
/// # Returns
/// Tuple of (reassigned_magnitude, time_reassignment, freq_reassignment)
/// - reassigned_magnitude: Reassigned magnitude spectrogram (n_freq x n_frames)
/// - time_reassignment: Time reassignment in frames (n_freq x n_frames)
/// - freq_reassignment: Frequency reassignment in bins (n_freq x n_frames)
///
/// # Example
/// ```
/// use giggle::spectrum::{reassigned_spectrogram, StftConfig};
/// use giggle::io;
///
/// let signal = io::tone(440.0, 22050, 0.5);
/// let config = StftConfig::default();
/// let (mag, time_r, freq_r) = reassigned_spectrogram(&signal, &config).unwrap();
/// assert_eq!(mag.shape(), time_r.shape());
/// assert_eq!(mag.shape(), freq_r.shape());
/// ```
pub fn reassigned_spectrogram(
    y: &[f32],
    config: &StftConfig,
) -> crate::Result<(Array2<f32>, Array2<f32>, Array2<f32>)> {
    crate::utils::valid_audio(y)?;
    if config.n_fft == 0 {
        return Err(crate::Error::InvalidSize {
            name: "n_fft",
            value: 0,
            reason: "must be > 0",
        });
    }
    if config.hop_length == 0 {
        return Err(crate::Error::InvalidSize {
            name: "hop_length",
            value: 0,
            reason: "must be > 0",
        });
    }

    // Compute regular STFT
    let stft_matrix = stft(y, config)?;
    let n_freq = stft_matrix.shape()[0];
    let n_frames = stft_matrix.shape()[1];

    // Create time-weighted window for group delay estimation
    let time_weighted_window: Vec<f32> = config
        .window
        .iter()
        .enumerate()
        .take(config.win_length)
        .map(|(i, &w)| w * i as f32)
        .collect();

    // Create derivative window for instantaneous frequency estimation
    let mut derivative_window = vec![0.0f32; config.win_length];
    for (i, dw) in derivative_window
        .iter_mut()
        .enumerate()
        .take(config.win_length - 1)
        .skip(1)
    {
        *dw = (config.window[i + 1] - config.window[i - 1]) / 2.0;
    }

    // Compute STFT with time-weighted window
    let mut time_config = config.clone();
    time_config.window = time_weighted_window;
    let stft_time = stft(y, &time_config)?;

    // Compute STFT with derivative window
    let mut deriv_config = config.clone();
    deriv_config.window = derivative_window;
    let stft_deriv = stft(y, &deriv_config)?;

    // Compute magnitude spectrogram and reassignment coordinates
    let mut magnitude = Array2::<f32>::zeros((n_freq, n_frames));
    let mut time_reassign = Array2::<f32>::zeros((n_freq, n_frames));
    let mut freq_reassign = Array2::<f32>::zeros((n_freq, n_frames));

    for freq in 0..n_freq {
        for frame in 0..n_frames {
            let s = stft_matrix[(freq, frame)];
            let s_time = stft_time[(freq, frame)];
            let s_deriv = stft_deriv[(freq, frame)];

            // Magnitude
            let mag = s.norm();
            magnitude[(freq, frame)] = mag;

            if mag > 1e-10 {
                // Group delay (time reassignment)
                let time_shift = (s_time.re * s.re + s_time.im * s.im) / (mag * mag);
                let time_reass = frame as f32 - time_shift / config.hop_length as f32;

                // Clamp to valid frame range
                time_reassign[(freq, frame)] = time_reass.clamp(0.0, (n_frames - 1) as f32);

                // Instantaneous frequency (frequency reassignment)
                let freq_shift = (s_deriv.im * s.re - s_deriv.re * s.im) / (mag * mag);
                let omega = 2.0 * std::f32::consts::PI * freq as f32 / config.n_fft as f32;
                let freq_reass = if omega.abs() > 1e-10 {
                    freq as f32 + freq_shift / omega
                } else {
                    freq as f32
                };

                // Clamp to valid frequency range
                freq_reassign[(freq, frame)] = freq_reass.clamp(0.0, (n_freq - 1) as f32);
            } else {
                // No reassignment for very small magnitudes
                time_reassign[(freq, frame)] = frame as f32;
                freq_reassign[(freq, frame)] = freq as f32;
            }
        }
    }

    Ok((magnitude, time_reassign, freq_reassign))
}

/// Griffin-Lim algorithm for phase reconstruction from magnitude spectrogram.
///
/// This iterative algorithm estimates the phase from a magnitude spectrogram
/// using repeated STFT/ISTFT operations.
///
/// # Arguments
/// * `magnitude` - Magnitude spectrogram (n_freq x n_frames)
/// * `config` - STFT configuration
/// * `n_iter` - Number of iterations (default: 32)
/// * `length` - Optional output length to match input signal
/// * `momentum` - Momentum coefficient for fast Griffin-Lim (0.0-1.0, default: 0.0)
///
/// # Returns
/// Reconstructed time-domain signal
pub fn griffinlim(
    magnitude: &Array2<f32>,
    config: &StftConfig,
    n_iter: usize,
    length: Option<usize>,
    momentum: f32,
) -> crate::Result<Vec<f32>> {
    let shape = magnitude.shape();
    let n_freq = shape[0];
    let n_frames = shape[1];

    if n_freq == 0 || n_frames == 0 {
        return Err(crate::Error::InvalidSize {
            name: "magnitude",
            value: 0,
            reason: "magnitude spectrogram must be non-empty",
        });
    }

    // Initialize with random phase (uniform on unit circle)
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut angles = Array2::<f32>::zeros((n_freq, n_frames));
    for angle in angles.iter_mut() {
        *angle = rng.gen_range(-std::f32::consts::PI..std::f32::consts::PI);
    }

    // Combine magnitude with initial angles to get complex STFT
    let mut stft_matrix = Array2::<Complex32>::zeros((n_freq, n_frames));
    for ((i, j), val) in stft_matrix.indexed_iter_mut() {
        let mag = magnitude[(i, j)];
        let angle = angles[(i, j)];
        *val = Complex32::new(mag * angle.cos(), mag * angle.sin());
    }

    // Store previous reconstruction for momentum
    let mut prev_reconstruction: Option<Vec<f32>> = None;

    // Iterative phase refinement
    for _ in 0..n_iter {
        // ISTFT to time domain
        let mut reconstruction = istft(&stft_matrix, config, length)?;

        // Apply momentum (fast Griffin-Lim)
        if momentum > 0.0
            && let Some(prev) = &prev_reconstruction
        {
            let min_len = reconstruction.len().min(prev.len());
            for i in 0..min_len {
                reconstruction[i] += momentum * (reconstruction[i] - prev[i]);
            }
        }

        prev_reconstruction = Some(reconstruction.clone());

        // STFT back to frequency domain
        let new_stft = stft(&reconstruction, config)?;

        // Extract phase from new STFT, keep original magnitude
        for ((i, j), val) in stft_matrix.indexed_iter_mut() {
            if i < new_stft.shape()[0] && j < new_stft.shape()[1] {
                let new_val = new_stft[(i, j)];
                let phase_val = if new_val.norm() > 1e-10 {
                    new_val / new_val.norm()
                } else {
                    Complex32::new(1.0, 0.0)
                };
                *val = magnitude[(i, j)] * phase_val;
            }
        }
    }

    // Final ISTFT
    istft(&stft_matrix, config, length)
}

/// Fast Mellin Transform (FMT) / Scale Transform.
///
/// The Mellin transform of a signal is performed by interpolating on an exponential
/// time axis, applying a polynomial window, and then taking the discrete Fourier transform.
///
/// When beta=0.5, it is known as the scale transform, which is useful for audio analysis
/// because its magnitude is invariant to scaling of the domain (time stretching/compression).
///
/// # Arguments
/// * `y` - Input signal (must have at least 3 samples)
/// * `t_min` - Minimum time spacing in samples (default: 0.5)
/// * `n_fmt` - Number of FMT bins (None for auto)
/// * `beta` - Mellin parameter (0.5 for scale transform)
/// * `over_sample` - Over-sampling factor (default: 1.0)
///
/// # Returns
/// Scale transform of the input signal (complex values)
///
/// # Example
/// ```
/// use giggle::spectrum::fmt;
///
/// let signal: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.01).sin()).collect();
/// let scale = fmt(&signal, 0.5, None, 0.5, 1.0).unwrap();
/// assert!(scale.len() > 0);
/// ```
pub fn fmt(
    y: &[f32],
    t_min: f32,
    n_fmt: Option<usize>,
    beta: f32,
    over_sample: f32,
) -> crate::Result<Vec<Complex32>> {
    let n = y.len();

    if n < 3 {
        return Err(crate::Error::InvalidSize {
            name: "y",
            value: n,
            reason: "must have at least 3 samples",
        });
    }
    if t_min <= 0.0 {
        return Err(crate::Error::InvalidParameter {
            name: "t_min",
            value: t_min.to_string(),
            reason: "must be > 0".to_string(),
        });
    }

    // Determine n_fmt if not provided
    let log_base = (n as f32 - 1.0).ln() - (n as f32 - 2.0).ln();
    let n_fmt = n_fmt.unwrap_or_else(|| {
        let bins = over_sample * ((n as f32 - 1.0).ln() - t_min.ln()) / log_base;
        bins.ceil() as usize
    });

    if n_fmt < 3 {
        return Err(crate::Error::InvalidSize {
            name: "n_fmt",
            value: n_fmt,
            reason: "computed n_fmt must be >= 3",
        });
    }

    let base = log_base.exp();

    // Original grid: signal covers [0, 1)
    let x: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();

    // Build exponential sampling grid
    let mut x_exp = Vec::with_capacity(n_fmt);
    let start = (t_min.ln() - (n as f32).ln()) / log_base;

    for i in 0..n_fmt {
        let exp_val = base.powf(start + i as f32 * (0.0 - start) / n_fmt as f32);
        x_exp.push(exp_val.clamp(t_min / n as f32, x[n - 1]));
    }

    // Linear interpolation to resample signal
    let mut y_res = Vec::with_capacity(n_fmt);
    for &x_val in &x_exp {
        // Find the surrounding samples for interpolation
        let x_scaled = x_val * n as f32;
        let idx_low = (x_scaled.floor() as usize).min(n - 2);
        let idx_high = idx_low + 1;
        let frac = x_scaled - idx_low as f32;

        let interp_val = y[idx_low] * (1.0 - frac) + y[idx_high] * frac;
        y_res.push(interp_val);
    }

    // Apply window (x^beta) and normalization
    let norm_factor = (n as f32).sqrt() / n_fmt as f32;
    for (i, val) in y_res.iter_mut().enumerate() {
        let window = x_exp[i].powf(beta) * norm_factor;
        *val *= window;
    }

    // Compute real FFT
    let fft_size = n_fmt.next_power_of_two();
    let fft = crate::fft::FftPlan::new(fft_size);

    let mut buffer = vec![Complex32::new(0.0, 0.0); fft_size];
    for i in 0..n_fmt {
        buffer[i].re = y_res[i];
    }

    fft.forward(&mut buffer);

    // Return positive frequency components
    buffer.truncate(fft_size / 2 + 1);
    Ok(buffer)
}

/// Biquad filter coefficients for IIR filtering.
#[derive(Clone, Debug)]
pub struct BiquadCoeffs {
    /// Numerator coefficients [b0, b1, b2]
    pub b: [f32; 3],
    /// Denominator coefficients [a0, a1, a2] (a0 is typically 1.0)
    pub a: [f32; 3],
}

impl BiquadCoeffs {
    /// Design a bandpass biquad filter using the bilinear transform.
    ///
    /// # Arguments
    /// * `center_freq` - Center frequency in Hz
    /// * `bandwidth` - Bandwidth in Hz
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    /// Biquad coefficients for the bandpass filter
    pub fn bandpass(center_freq: f32, bandwidth: f32, sample_rate: f32) -> Self {
        use std::f32::consts::PI;

        // Prewarped frequencies
        let omega0 = 2.0 * PI * center_freq / sample_rate;
        let cos_omega0 = omega0.cos();
        let sin_omega0 = omega0.sin();

        // Q factor from bandwidth
        let q = center_freq / bandwidth.max(1.0);
        let alpha = sin_omega0 / (2.0 * q);

        // Bandpass filter coefficients (constant 0 dB peak gain)
        let b0 = alpha;
        let b1 = 0.0;
        let b2 = -alpha;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_omega0;
        let a2 = 1.0 - alpha;

        // Normalize by a0
        BiquadCoeffs {
            b: [b0 / a0, b1 / a0, b2 / a0],
            a: [1.0, a1 / a0, a2 / a0],
        }
    }

    /// Apply the biquad filter to a signal (direct form II transposed).
    pub fn filter(&self, input: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0f32; input.len()];
        let mut z1 = 0.0f32;
        let mut z2 = 0.0f32;

        for (i, &x) in input.iter().enumerate() {
            let y = self.b[0] * x + z1;
            z1 = self.b[1] * x - self.a[1] * y + z2;
            z2 = self.b[2] * x - self.a[2] * y;
            output[i] = y;
        }

        output
    }

    /// Apply zero-phase filtering (forward-backward filtering).
    ///
    /// This applies the filter forward, then backward, resulting in
    /// zero phase distortion but doubled filter order.
    pub fn filtfilt(&self, input: &[f32]) -> Vec<f32> {
        if input.is_empty() {
            return Vec::new();
        }

        // Pad the signal to reduce edge effects
        let pad_len = 3 * 3.max(input.len() / 10).min(100);
        let mut padded = Vec::with_capacity(input.len() + 2 * pad_len);

        // Reflect padding at start
        for i in (1..=pad_len).rev() {
            let idx = i.min(input.len() - 1);
            padded.push(2.0 * input[0] - input[idx]);
        }
        padded.extend_from_slice(input);
        // Reflect padding at end
        for i in 1..=pad_len {
            let idx = (input.len() - 1).saturating_sub(i);
            padded.push(2.0 * input[input.len() - 1] - input[idx]);
        }

        // Forward filter
        let forward = self.filter(&padded);

        // Reverse
        let reversed: Vec<f32> = forward.into_iter().rev().collect();

        // Backward filter
        let backward = self.filter(&reversed);

        // Reverse again and remove padding
        backward
            .into_iter()
            .rev()
            .skip(pad_len)
            .take(input.len())
            .collect()
    }
}

/// Second-order sections (SOS) for cascaded biquad filtering.
#[derive(Clone, Debug)]
pub struct SosFilter {
    sections: Vec<BiquadCoeffs>,
}

impl SosFilter {
    /// Create a new SOS filter from a list of biquad sections.
    pub fn new(sections: Vec<BiquadCoeffs>) -> Self {
        SosFilter { sections }
    }

    /// Design a higher-order bandpass filter as cascaded biquads.
    ///
    /// # Arguments
    /// * `center_freq` - Center frequency in Hz
    /// * `bandwidth` - Bandwidth in Hz
    /// * `sample_rate` - Sample rate in Hz
    /// * `order` - Filter order (number of biquad sections)
    pub fn bandpass(center_freq: f32, bandwidth: f32, sample_rate: f32, order: usize) -> Self {
        let sections: Vec<BiquadCoeffs> = (0..order.max(1))
            .map(|_| BiquadCoeffs::bandpass(center_freq, bandwidth, sample_rate))
            .collect();
        SosFilter { sections }
    }

    /// Apply zero-phase filtering using cascaded biquads.
    pub fn filtfilt(&self, input: &[f32]) -> Vec<f32> {
        if self.sections.is_empty() {
            return input.to_vec();
        }

        let mut result = input.to_vec();
        for section in &self.sections {
            result = section.filtfilt(&result);
        }
        result
    }
}

/// Time-frequency representation using IIR filters (IIRT).
///
/// IIRT provides a time-frequency representation using a multirate
/// filterbank of IIR bandpass filters. This is useful for music analysis
/// with pitch-based frequency resolution.
///
/// The default configuration produces 85 filters with MIDI pitches [24, 108]
/// as center frequencies, each filter having a bandwidth of one semitone.
///
/// # Arguments
/// * `y` - Audio time series
/// * `sr` - Sample rate in Hz
/// * `win_length` - Window length for computing short-time mean-square power (default: 2048)
/// * `hop_length` - Hop length between frames (default: win_length/4)
/// * `center` - If true, center frames on samples (default: true)
/// * `fmin` - Minimum frequency in Hz (default: ~32.7 Hz, C1)
/// * `fmax` - Maximum frequency in Hz (default: ~4186 Hz, C8)
/// * `bins_per_octave` - Number of frequency bins per octave (default: 12)
/// * `tuning` - Tuning offset in fractions of a bin (default: 0.0)
/// * `q_factor` - Q factor for filter bandwidth (default: 25.0)
///
/// # Returns
/// Short-time mean-square power spectrogram (n_bins x n_frames)
///
/// # Example
/// ```
/// use giggle::spectrum::iirt;
/// use giggle::io;
///
/// let signal = io::tone(440.0, 22050, 0.5);
/// let spec = iirt(&signal, 22050, 2048, 512, true, 32.7, 4186.0, 12, 0.0, 25.0).unwrap();
/// assert!(spec.shape()[0] > 0);
/// assert!(spec.shape()[1] > 0);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn iirt(
    y: &[f32],
    sr: u32,
    win_length: usize,
    hop_length: usize,
    center: bool,
    fmin: f32,
    fmax: f32,
    bins_per_octave: usize,
    tuning: f32,
    q_factor: f32,
) -> crate::Result<Array2<f32>> {
    crate::utils::valid_audio(y)?;
    if win_length == 0 {
        return Err(crate::Error::InvalidSize {
            name: "win_length",
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

    let sr_f = sr as f32;
    let nyquist = sr_f / 2.0;

    // Clamp frequency range
    let fmin = fmin.max(20.0);
    let fmax = fmax.min(nyquist * 0.95);

    if fmin >= fmax {
        return Err(crate::Error::InvalidFrequencyRange {
            fmin,
            fmax,
            reason: "fmin must be less than fmax (after clamping)".to_string(),
        });
    }

    // Calculate number of bins
    let n_octaves = (fmax / fmin).log2();
    let n_bins = (n_octaves * bins_per_octave as f32).ceil() as usize;

    if n_bins == 0 {
        return Err(crate::Error::InvalidSize {
            name: "n_bins",
            value: 0,
            reason: "frequency range too narrow for any bins",
        });
    }

    // Generate center frequencies with tuning adjustment
    let tuning_factor = 2.0_f32.powf(tuning / bins_per_octave as f32);
    let center_freqs: Vec<f32> = (0..n_bins)
        .map(|i| fmin * tuning_factor * 2.0_f32.powf(i as f32 / bins_per_octave as f32))
        .filter(|&f| f < nyquist)
        .collect();

    let n_bins = center_freqs.len();
    if n_bins == 0 {
        return Err(crate::Error::InvalidSize {
            name: "n_bins",
            value: 0,
            reason: "no center frequencies below Nyquist after tuning",
        });
    }

    // Pad the signal if centering
    let padded = if center {
        let pad = win_length / 2;
        let mut p = vec![0.0f32; pad];
        p.extend_from_slice(y);
        p.extend(vec![0.0f32; pad]);
        p
    } else {
        y.to_vec()
    };

    // Calculate number of frames
    let n_frames = if padded.len() >= win_length {
        (padded.len() - win_length) / hop_length + 1
    } else {
        0
    };

    if n_frames == 0 {
        return Err(crate::Error::InvalidSize {
            name: "n_frames",
            value: 0,
            reason: "signal too short for any frames with given win_length",
        });
    }

    // Determine optimal sample rates for each filter
    // Use multirate approach: lower frequencies use lower sample rates
    let sample_rates: Vec<f32> = center_freqs
        .iter()
        .map(|&f| {
            // Target at least 4x the center frequency, but not more than original sr
            let target_sr = (f * 4.0).max(1000.0).min(sr_f);
            // Round to a "nice" value that divides evenly
            let ratio = (sr_f / target_sr).floor().max(1.0);
            sr_f / ratio
        })
        .collect();

    // Get unique sample rates for resampling
    let mut unique_srs: Vec<f32> = sample_rates.clone();
    unique_srs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    unique_srs.dedup_by(|a, b| (*a - *b).abs() < 0.1);

    // Resample signal to each unique sample rate
    let resampled: std::collections::HashMap<i32, Vec<f32>> = unique_srs
        .iter()
        .map(|&target_sr| {
            let key = (target_sr * 10.0) as i32; // Use scaled int as key
            let resampled_signal = if (target_sr - sr_f).abs() < 0.1 {
                padded.clone()
            } else {
                // Simple decimation (could use rubato for better quality)
                let ratio = (sr_f / target_sr).round() as usize;
                padded.iter().step_by(ratio.max(1)).cloned().collect()
            };
            (key, resampled_signal)
        })
        .collect();

    // Create output array
    let mut bands_power = Array2::<f32>::zeros((n_bins, n_frames));

    // Process each frequency band
    for (bin_idx, (&center_freq, &filter_sr)) in
        center_freqs.iter().zip(sample_rates.iter()).enumerate()
    {
        // Get the resampled signal for this filter's sample rate
        let sr_key = (filter_sr * 10.0) as i32;
        let signal = match resampled.get(&sr_key) {
            Some(s) => s,
            None => continue,
        };

        if signal.is_empty() {
            continue;
        }

        // Design bandpass filter
        let bandwidth = center_freq / q_factor;
        let filter = SosFilter::bandpass(center_freq, bandwidth, filter_sr, 2);

        // Apply zero-phase filter
        let filtered = filter.filtfilt(signal);

        // Calculate scaling factor for this sample rate
        let factor = sr_f / filter_sr;
        let hop_scaled = (hop_length as f32 / factor).round() as usize;
        let win_scaled = (win_length as f32 / factor).round() as usize;

        if hop_scaled == 0 || win_scaled == 0 {
            continue;
        }

        // Compute short-time mean-square power for each frame
        for frame_idx in 0..n_frames {
            let start_orig = frame_idx * hop_length;
            let start_scaled = (start_orig as f32 / factor).round() as usize;

            if start_scaled + win_scaled > filtered.len() {
                continue;
            }

            // Sum of squared samples
            let power: f32 = filtered[start_scaled..start_scaled + win_scaled]
                .iter()
                .map(|&x| x * x)
                .sum();

            // Scale by factor to account for different sample rates
            bands_power[(bin_idx, frame_idx)] = power * factor / win_scaled as f32;
        }
    }

    Ok(bands_power)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_power_db_roundtrip() {
        let power = Array2::from_shape_vec(
            (3, 4),
            vec![
                1.0, 2.0, 0.5, 0.1, 0.01, 0.001, 10.0, 100.0, 0.25, 0.75, 1.5, 5.0,
            ],
        )
        .unwrap();

        let db = power_to_db(&power, 1.0, 1e-10, None);
        let recovered = db_to_power(&db, 1.0);

        for i in 0..3 {
            for j in 0..4 {
                assert_relative_eq!(power[(i, j)], recovered[(i, j)], epsilon = 1e-5);
            }
        }
    }

    #[test]
    fn test_amplitude_db_roundtrip() {
        let amplitude =
            Array2::from_shape_vec((2, 3), vec![1.0, 0.5, 2.0, 0.1, 10.0, 0.01]).unwrap();

        let db = amplitude_to_db(&amplitude, 1.0, 1e-10, None);
        let recovered = db_to_amplitude(&db, 1.0);

        for i in 0..2 {
            for j in 0..3 {
                assert_relative_eq!(amplitude[(i, j)], recovered[(i, j)], epsilon = 1e-5);
            }
        }
    }

    #[test]
    fn test_power_to_db_reference() {
        let power = Array2::from_shape_vec((1, 3), vec![1.0, 10.0, 100.0]).unwrap();
        let db = power_to_db(&power, 1.0, 1e-10, None);

        assert_relative_eq!(db[(0, 0)], 0.0, epsilon = 0.01);
        assert_relative_eq!(db[(0, 1)], 10.0, epsilon = 0.01);
        assert_relative_eq!(db[(0, 2)], 20.0, epsilon = 0.01);
    }

    #[test]
    fn test_amplitude_to_db_reference() {
        let amplitude = Array2::from_shape_vec((1, 3), vec![1.0, 10.0, 100.0]).unwrap();
        let db = amplitude_to_db(&amplitude, 1.0, 1e-10, None);

        assert_relative_eq!(db[(0, 0)], 0.0, epsilon = 0.01);
        assert_relative_eq!(db[(0, 1)], 20.0, epsilon = 0.01);
        assert_relative_eq!(db[(0, 2)], 40.0, epsilon = 0.01);
    }

    #[test]
    fn test_db_top_clipping() {
        let power = Array2::from_shape_vec((1, 5), vec![100.0, 10.0, 1.0, 0.1, 0.01]).unwrap();
        let db = power_to_db(&power, 1.0, 1e-10, Some(30.0));

        let min_db = db.iter().copied().fold(f32::INFINITY, f32::min);
        let max_db = db.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        assert!((max_db - min_db) <= 30.0 + 0.1);
    }

    #[test]
    fn test_griffinlim_basic() {
        use crate::io;

        // Generate a simple tone
        let sr = 22050;
        let duration = 1.0;
        let freq = 440.0;
        let signal = io::tone(freq, sr, duration);

        // Compute STFT
        let config = StftConfig {
            n_fft: 2048,
            hop_length: 512,
            win_length: 2048,
            window: window::hann(2048),
            center: true,
            pad_mode: PadMode::Reflect,
        };

        let stft_matrix = stft(&signal, &config).unwrap();
        let (magnitude, _phase) = magphase(&stft_matrix);

        // Reconstruct using Griffin-Lim
        let reconstructed = griffinlim(&magnitude, &config, 8, Some(signal.len()), 0.0).unwrap();

        // Check length matches
        assert_eq!(reconstructed.len(), signal.len());

        // The reconstruction should be non-zero and have reasonable energy
        let energy: f32 = reconstructed.iter().map(|x| x * x).sum();
        assert!(energy > 0.1, "Reconstructed signal has too little energy");
    }

    #[test]
    fn test_griffinlim_empty() {
        let magnitude = Array2::<f32>::zeros((0, 0));
        let config = StftConfig::default();
        let result = griffinlim(&magnitude, &config, 8, None, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_pcen_basic() {
        // Create a simple spectrogram
        let spec = Array2::from_shape_vec(
            (4, 10),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.5, 1.0, 1.5, 2.0, 2.5, 2.0,
                1.5, 1.0, 0.5, 0.25, 2.0, 4.0, 6.0, 8.0, 10.0, 8.0, 6.0, 4.0, 2.0, 1.0, 0.1, 0.2,
                0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05,
            ],
        )
        .unwrap();

        let sr = 22050;
        let hop_length = 512;
        let gain = 0.8;
        let bias = 2.0;
        let power = 0.25;
        let time_constant = 0.4;
        let eps = 1e-6;

        let result = pcen(&spec, sr, hop_length, gain, bias, power, time_constant, eps).unwrap();

        // Check output shape
        assert_eq!(result.shape(), spec.shape());

        // Check all values are finite
        for val in result.iter() {
            assert!(val.is_finite());
        }

        // PCEN should compress dynamic range
        let spec_max = spec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let spec_min = spec
            .iter()
            .cloned()
            .filter(|&x| x > 0.0)
            .fold(f32::INFINITY, f32::min);
        let spec_ratio = spec_max / spec_min;

        let result_max = result.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let result_min = result
            .iter()
            .cloned()
            .filter(|&x| x > 0.0)
            .fold(f32::INFINITY, f32::min);
        let result_ratio = result_max / result_min;

        // Ratio should be compressed (smaller)
        assert!(result_ratio < spec_ratio);
    }

    #[test]
    fn test_pcen_empty() {
        let spec = Array2::<f32>::zeros((4, 0));
        let result = pcen(&spec, 22050, 512, 0.8, 2.0, 0.25, 0.4, 1e-6);
        assert!(result.is_err());
    }

    #[test]
    fn test_pcen_properties() {
        // Test PCEN with constant input
        let constant_spec = Array2::from_elem((3, 20), 1.0);
        let result = pcen(&constant_spec, 22050, 512, 0.8, 2.0, 0.25, 0.4, 1e-6).unwrap();

        // With constant input, PCEN should stabilize after AGC filter settles
        // Later frames should have similar values
        let last_frame_vals: Vec<f32> = (0..3).map(|i| result[(i, 19)]).collect();

        for val in &last_frame_vals {
            assert!(val.is_finite());
            // Values should be in reasonable range
            assert!(*val >= 0.0 && *val < 10.0);
        }
    }

    #[test]
    fn test_pcen_gain_normalization() {
        // Test that PCEN normalizes gain variations
        let mut loud_spec = Array2::from_elem((2, 10), 10.0);
        let mut quiet_spec = Array2::from_elem((2, 10), 1.0);

        // Add some variation
        for i in 0..10 {
            loud_spec[(0, i)] += (i as f32).sin();
            quiet_spec[(0, i)] += (i as f32).sin() * 0.1;
        }

        let loud_pcen = pcen(&loud_spec, 22050, 512, 0.8, 2.0, 0.25, 0.4, 1e-6).unwrap();
        let quiet_pcen = pcen(&quiet_spec, 22050, 512, 0.8, 2.0, 0.25, 0.4, 1e-6).unwrap();

        // PCEN should reduce the difference between loud and quiet signals
        let loud_mean = loud_spec.mean().unwrap();
        let quiet_mean = quiet_spec.mean().unwrap();
        let input_ratio = loud_mean / quiet_mean;

        let loud_pcen_mean = loud_pcen.mean().unwrap();
        let quiet_pcen_mean = quiet_pcen.mean().unwrap();
        let pcen_ratio = loud_pcen_mean / quiet_pcen_mean;

        // Ratio should be closer to 1 after PCEN
        assert!(pcen_ratio < input_ratio);
        assert!((pcen_ratio - 1.0).abs() < (input_ratio - 1.0).abs());
    }

    #[test]
    fn test_a_weighting_reference() {
        // Test A-weighting at reference frequency (1 kHz should be close to 0 dB)
        let freqs = vec![1000.0];
        let weights = perceptual_weighting(&freqs, WeightingType::A);

        assert_relative_eq!(weights[0], 0.0, epsilon = 0.1);
    }

    #[test]
    fn test_perceptual_weighting_shape() {
        let freqs = vec![100.0, 500.0, 1000.0, 5000.0, 10000.0];
        let weights = perceptual_weighting(&freqs, WeightingType::A);

        assert_eq!(weights.len(), freqs.len());
    }

    #[test]
    fn test_a_weighting_properties() {
        // A-weighting should attenuate low frequencies and have peak near 2-4 kHz
        let freqs = vec![100.0, 1000.0, 3000.0, 10000.0];
        let weights = perceptual_weighting(&freqs, WeightingType::A);

        // 100 Hz should be heavily attenuated (negative dB)
        assert!(weights[0] < -10.0);

        // 3 kHz should have less attenuation than 100 Hz
        assert!(weights[2] > weights[0]);

        // 1 kHz should be close to 0 dB
        assert!(weights[1].abs() < 5.0);
    }

    #[test]
    fn test_c_weighting_flatter() {
        // C-weighting should be flatter than A-weighting
        let freqs = vec![100.0, 1000.0, 10000.0];
        let a_weights = perceptual_weighting(&freqs, WeightingType::A);
        let c_weights = perceptual_weighting(&freqs, WeightingType::C);

        // Range of C-weighting should be smaller
        let a_range = a_weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            - a_weights.iter().cloned().fold(f32::INFINITY, f32::min);
        let c_range = c_weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            - c_weights.iter().cloned().fold(f32::INFINITY, f32::min);

        assert!(c_range < a_range);
    }

    #[test]
    fn test_perceptual_weighting_finite() {
        let freqs = vec![20.0, 100.0, 1000.0, 10000.0, 20000.0];

        for weighting_type in &[
            WeightingType::A,
            WeightingType::B,
            WeightingType::C,
            WeightingType::D,
        ] {
            let weights = perceptual_weighting(&freqs, *weighting_type);
            for &w in &weights {
                assert!(w.is_finite(), "Non-finite weight for {:?}", weighting_type);
            }
        }
    }

    #[test]
    fn test_perceptual_weighting_zero_freq() {
        let freqs = vec![0.0, -100.0];
        let weights = perceptual_weighting(&freqs, WeightingType::A);

        // Zero and negative frequencies should return -infinity
        assert_eq!(weights[0], f32::NEG_INFINITY);
        assert_eq!(weights[1], f32::NEG_INFINITY);
    }

    #[test]
    fn test_all_weighting_curves() {
        let freqs = vec![100.0, 1000.0, 10000.0];

        // Test that all weighting curves produce reasonable values
        let a = perceptual_weighting(&freqs, WeightingType::A);
        let b = perceptual_weighting(&freqs, WeightingType::B);
        let c = perceptual_weighting(&freqs, WeightingType::C);
        let d = perceptual_weighting(&freqs, WeightingType::D);

        // All should have 3 values
        assert_eq!(a.len(), 3);
        assert_eq!(b.len(), 3);
        assert_eq!(c.len(), 3);
        assert_eq!(d.len(), 3);

        // All should be finite
        for weights in &[a, b, c, d] {
            for &w in weights {
                assert!(w.is_finite());
            }
        }
    }

    #[test]
    fn test_reassigned_spectrogram_shape() {
        use crate::io;

        let signal = io::tone(440.0, 22050, 0.5);
        let config = StftConfig::default();

        let (mag, time_r, freq_r) = reassigned_spectrogram(&signal, &config).unwrap();

        // All arrays should have the same shape
        assert_eq!(mag.shape(), time_r.shape());
        assert_eq!(mag.shape(), freq_r.shape());

        // Should have non-zero dimensions
        assert!(mag.shape()[0] > 0);
        assert!(mag.shape()[1] > 0);
    }

    #[test]
    fn test_reassigned_spectrogram_empty() {
        let signal = vec![];
        let config = StftConfig::default();

        let result = reassigned_spectrogram(&signal, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_reassigned_spectrogram_properties() {
        use crate::io;

        let signal = io::tone(880.0, 22050, 0.3);
        let config = StftConfig::default();

        let (mag, time_r, freq_r) = reassigned_spectrogram(&signal, &config).unwrap();

        // All values should be finite
        for &v in mag.iter() {
            assert!(v.is_finite());
            assert!(v >= 0.0); // Magnitude should be non-negative
        }

        for &v in time_r.iter() {
            assert!(v.is_finite());
        }

        for &v in freq_r.iter() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_reassigned_spectrogram_magnitude() {
        use crate::io;

        let signal = io::tone(440.0, 22050, 0.5);
        let config = StftConfig::default();

        // Compute both regular STFT and reassigned
        let stft_matrix = stft(&signal, &config).unwrap();
        let (reassigned_mag, _time_r, _freq_r) = reassigned_spectrogram(&signal, &config).unwrap();

        // Shapes should match
        assert_eq!(reassigned_mag.shape()[0], stft_matrix.shape()[0]);
        assert_eq!(reassigned_mag.shape()[1], stft_matrix.shape()[1]);

        // Energy should be preserved (roughly)
        let stft_energy: f32 = stft_matrix.iter().map(|c| c.norm_sqr()).sum();
        let reassigned_energy: f32 = reassigned_mag.iter().map(|&m| m * m).sum();

        let ratio = (stft_energy / reassigned_energy).log10().abs();
        assert!(ratio < 0.1, "Energy should be preserved");
    }

    #[test]
    fn test_reassigned_spectrogram_bounds() {
        use crate::io;

        let signal = io::tone(440.0, 22050, 0.5);
        let config = StftConfig::default();

        let (mag, time_r, freq_r) = reassigned_spectrogram(&signal, &config).unwrap();

        let n_freq = mag.shape()[0];
        let n_frames = mag.shape()[1];

        // Time reassignment should be within reasonable bounds
        for &t in time_r.iter() {
            if t.is_finite() {
                // Allow some margin beyond frame boundaries
                assert!(t >= -10.0 && t < (n_frames + 10) as f32);
            }
        }

        // Frequency reassignment should be within reasonable bounds
        for &f in freq_r.iter() {
            if f.is_finite() {
                // Allow some margin beyond frequency boundaries
                assert!(f >= -10.0 && f < (n_freq + 10) as f32);
            }
        }
    }

    #[test]
    fn test_fmt_basic() {
        // Create a simple test signal
        let signal: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).sin()).collect();

        let result = fmt(&signal, 1.0, None, 0.5, 1.0).unwrap();

        // Should return non-empty result
        assert!(!result.is_empty());

        // All values should be finite
        for &v in &result {
            assert!(v.re.is_finite());
            assert!(v.im.is_finite());
        }
    }

    #[test]
    fn test_fmt_empty_input() {
        let signal: Vec<f32> = vec![];
        let result = fmt(&signal, 1.0, None, 0.5, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_fmt_short_input() {
        // Less than 3 samples should return error
        let signal = vec![1.0, 2.0];
        let result = fmt(&signal, 1.0, None, 0.5, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_fmt_invalid_t_min() {
        let signal: Vec<f32> = (0..100).map(|i| i as f32).collect();

        // Zero t_min should return error
        let result = fmt(&signal, 0.0, None, 0.5, 1.0);
        assert!(result.is_err());

        // Negative t_min should return error
        let result = fmt(&signal, -1.0, None, 0.5, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_fmt_with_explicit_n_fmt() {
        let signal: Vec<f32> = (0..256).map(|i| (i as f32 * 0.05).sin()).collect();

        let result = fmt(&signal, 1.0, Some(64), 0.5, 1.0).unwrap();

        // Result length should be related to next power of 2 of n_fmt
        assert!(!result.is_empty());
        assert!(result.len() <= 65); // 64.next_power_of_two() / 2 + 1 = 33
    }

    #[test]
    fn test_fmt_different_beta() {
        let signal: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).sin()).collect();

        // Different beta values should produce different results
        let result_beta0 = fmt(&signal, 1.0, Some(32), 0.0, 1.0).unwrap();
        let result_beta1 = fmt(&signal, 1.0, Some(32), 1.0, 1.0).unwrap();

        assert!(!result_beta0.is_empty());
        assert!(!result_beta1.is_empty());

        // Results should differ
        let mut different = false;
        for i in 0..result_beta0.len().min(result_beta1.len()) {
            if (result_beta0[i].re - result_beta1[i].re).abs() > 1e-6 {
                different = true;
                break;
            }
        }
        assert!(
            different,
            "Different beta values should produce different results"
        );
    }

    #[test]
    fn test_fmt_over_sample() {
        let signal: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).sin()).collect();

        // Higher over_sample should produce more bins
        let result_os1 = fmt(&signal, 1.0, None, 0.5, 1.0).unwrap();
        let result_os2 = fmt(&signal, 1.0, None, 0.5, 2.0).unwrap();

        // With higher oversampling, we expect more output bins
        assert!(!result_os1.is_empty());
        assert!(!result_os2.is_empty());
    }

    #[test]
    fn test_fmt_scale_invariance_property() {
        // FMT should be useful for scale-invariant analysis
        // Test that similar patterns at different scales produce related outputs
        let signal1: Vec<f32> = (0..256).map(|i| (i as f32 * 0.05).sin()).collect();
        let signal2: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).sin()).collect();

        let result1 = fmt(&signal1, 1.0, Some(32), 0.5, 1.0).unwrap();
        let result2 = fmt(&signal2, 1.0, Some(32), 0.5, 1.0).unwrap();

        // Both should produce valid outputs
        assert!(!result1.is_empty());
        assert!(!result2.is_empty());

        // Check energy is present in both
        let energy1: f32 = result1.iter().map(|c| c.norm_sqr()).sum();
        let energy2: f32 = result2.iter().map(|c| c.norm_sqr()).sum();

        assert!(energy1 > 0.0);
        assert!(energy2 > 0.0);
    }

    #[test]
    fn test_biquad_bandpass() {
        let coeffs = BiquadCoeffs::bandpass(1000.0, 100.0, 44100.0);

        // Coefficients should be finite
        for &b in &coeffs.b {
            assert!(b.is_finite());
        }
        for &a in &coeffs.a {
            assert!(a.is_finite());
        }

        // a[0] should be 1.0 (normalized)
        assert_relative_eq!(coeffs.a[0], 1.0, epsilon = 0.001);
    }

    #[test]
    fn test_biquad_filter() {
        let coeffs = BiquadCoeffs::bandpass(1000.0, 100.0, 44100.0);

        // Create a test signal with a component at 1000 Hz
        let signal: Vec<f32> = (0..4410)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 44100.0).sin())
            .collect();

        let filtered = coeffs.filter(&signal);

        assert_eq!(filtered.len(), signal.len());

        // Output should have energy
        let energy: f32 = filtered.iter().map(|&x| x * x).sum();
        assert!(energy > 0.0);
    }

    #[test]
    fn test_biquad_filtfilt() {
        let coeffs = BiquadCoeffs::bandpass(1000.0, 200.0, 44100.0);

        // Create a test signal
        let signal: Vec<f32> = (0..4410)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 44100.0).sin())
            .collect();

        let filtered = coeffs.filtfilt(&signal);

        assert_eq!(filtered.len(), signal.len());

        // Output should have energy
        let energy: f32 = filtered.iter().map(|&x| x * x).sum();
        assert!(energy > 0.0);

        // All values should be finite
        for &v in &filtered {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_sos_filter() {
        let sos = SosFilter::bandpass(1000.0, 200.0, 44100.0, 2);

        let signal: Vec<f32> = (0..4410)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 44100.0).sin())
            .collect();

        let filtered = sos.filtfilt(&signal);

        assert_eq!(filtered.len(), signal.len());

        // All values should be finite
        for &v in &filtered {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_iirt_basic() {
        use crate::io;

        let signal = io::tone(440.0, 22050, 0.5);
        let spec = iirt(&signal, 22050, 2048, 512, true, 32.7, 4186.0, 12, 0.0, 25.0).unwrap();

        // Should produce output
        assert!(spec.shape()[0] > 0, "Should have frequency bins");
        assert!(spec.shape()[1] > 0, "Should have time frames");

        // All values should be non-negative (power)
        for &v in spec.iter() {
            assert!(v >= 0.0, "Power should be non-negative");
            assert!(v.is_finite(), "Values should be finite");
        }
    }

    #[test]
    fn test_iirt_empty() {
        let signal: Vec<f32> = vec![];
        let result = iirt(&signal, 22050, 2048, 512, true, 32.7, 4186.0, 12, 0.0, 25.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_iirt_shape() {
        use crate::io;

        let signal = io::tone(440.0, 22050, 1.0);
        let spec = iirt(&signal, 22050, 2048, 512, true, 65.0, 2093.0, 12, 0.0, 25.0).unwrap();

        // Check that we get reasonable number of bins
        // 65 Hz to 2093 Hz is about 5 octaves, so ~60 bins at 12 bins per octave
        assert!(
            spec.shape()[0] >= 50 && spec.shape()[0] <= 70,
            "Expected ~60 bins, got {}",
            spec.shape()[0]
        );

        // Check reasonable number of frames for 1 second at hop=512
        // ~22050 / 512 = ~43 frames
        assert!(
            spec.shape()[1] >= 30 && spec.shape()[1] <= 60,
            "Expected ~43 frames, got {}",
            spec.shape()[1]
        );
    }

    #[test]
    fn test_iirt_frequency_detection() {
        use crate::io;

        let sr = 22050;
        let freq = 440.0; // A4
        let signal = io::tone(freq, sr, 0.5);

        let spec = iirt(&signal, sr, 2048, 512, true, 65.0, 2093.0, 12, 0.0, 25.0).unwrap();

        // Find the bin with maximum total energy
        let n_bins = spec.shape()[0];
        let mut max_bin = 0;
        let mut max_energy = 0.0f32;

        for bin in 0..n_bins {
            let energy: f32 = (0..spec.shape()[1]).map(|t| spec[(bin, t)]).sum();

            if energy > max_energy {
                max_energy = energy;
                max_bin = bin;
            }
        }

        // Expected bin for 440 Hz with fmin=65 Hz, 12 bins per octave
        // bin = 12 * log2(440/65) ≈ 12 * 2.76 ≈ 33
        let expected_bin = (12.0 * (freq / 65.0).log2()).round() as i32;

        // Peak should be within a few bins of expected
        assert!(
            (max_bin as i32 - expected_bin).abs() <= 5,
            "Expected peak near bin {}, got {} for {} Hz",
            expected_bin,
            max_bin,
            freq
        );
    }

    #[test]
    fn test_iirt_different_tuning() {
        use crate::io;

        let signal = io::tone(440.0, 22050, 0.3);

        let spec_no_tuning =
            iirt(&signal, 22050, 2048, 512, true, 65.0, 2093.0, 12, 0.0, 25.0).unwrap();
        let spec_tuned =
            iirt(&signal, 22050, 2048, 512, true, 65.0, 2093.0, 12, 0.5, 25.0).unwrap();

        // Both should have same shape
        assert_eq!(spec_no_tuning.shape(), spec_tuned.shape());

        // But different values due to frequency shift
        let mut different = false;
        for i in 0..spec_no_tuning.shape()[0].min(10) {
            if (spec_no_tuning[(i, 0)] - spec_tuned[(i, 0)]).abs() > 1e-10 {
                different = true;
                break;
            }
        }
        assert!(different, "Tuning should affect IIRT values");
    }

    #[test]
    fn test_iirt_bins_per_octave() {
        use crate::io;

        let signal = io::tone(440.0, 22050, 0.3);

        // 12 bins per octave (semitones)
        let spec_12 = iirt(&signal, 22050, 2048, 512, true, 65.0, 2093.0, 12, 0.0, 25.0).unwrap();

        // 24 bins per octave (quarter-tones)
        let spec_24 = iirt(&signal, 22050, 2048, 512, true, 65.0, 2093.0, 24, 0.0, 25.0).unwrap();

        // 24 bins per octave should have roughly twice as many bins
        assert!(spec_24.shape()[0] > spec_12.shape()[0]);
        assert!(spec_24.shape()[0] <= spec_12.shape()[0] * 3);
    }

    #[test]
    fn test_iirt_center_padding() {
        use crate::io;

        let signal = io::tone(440.0, 22050, 0.3);

        let spec_centered =
            iirt(&signal, 22050, 2048, 512, true, 65.0, 2093.0, 12, 0.0, 25.0).unwrap();
        let spec_not_centered = iirt(
            &signal, 22050, 2048, 512, false, 65.0, 2093.0, 12, 0.0, 25.0,
        )
        .unwrap();

        // Both should produce output
        assert!(spec_centered.shape()[1] > 0);
        assert!(spec_not_centered.shape()[1] > 0);

        // Centered should have same or more frames due to padding
        assert!(spec_centered.shape()[1] >= spec_not_centered.shape()[1]);
    }
}
