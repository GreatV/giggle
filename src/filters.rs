//! Advanced filter functions for audio analysis.
//!
//! This module provides wavelet filterbanks, multi-rate filterbanks,
//! and related filter design utilities.

use crate::convert::{A4_HZ, MIDI_A4};
use crate::utils;
use crate::window::{self, WindowType};
use ndarray::Array2;
use num_complex::Complex32;

/// Convert single MIDI note to Hz
fn midi_to_hz_single(midi: f32) -> f32 {
    A4_HZ * 2.0f32.powf((midi - MIDI_A4) / 12.0)
}

/// Get the equivalent noise bandwidth (ENBW) of a window function.
///
/// The ENBW of a window is defined as the normalized ratio of
/// the sum of squares to the square of sums:
///
/// `enbw = n * sum(window^2) / sum(window)^2`
///
/// Reference: Harris, F. J. "On the use of windows for harmonic analysis
/// with the discrete Fourier transform." Proceedings of the IEEE, 66(1), 51-83. 1978.
///
/// # Arguments
/// * `window` - Window type
/// * `n` - Number of coefficients to use (default: 1000)
///
/// # Returns
/// The equivalent noise bandwidth (in FFT bins)
pub fn window_bandwidth(window: WindowType, n: usize) -> f32 {
    let n = if n == 0 { 1000 } else { n };
    let w = window::get_window(window, n);

    let sum_sq: f32 = w.iter().map(|x| x * x).sum();
    let sum: f32 = w.iter().sum();

    if sum.abs() < 1e-10 {
        return 1.0; // fallback for zero-sum windows
    }

    (n as f32) * sum_sq / (sum * sum)
}

/// Get equivalent noise bandwidth from a string window specification.
pub fn window_bandwidth_from_str(window_name: &str, n: usize) -> Option<f32> {
    WindowType::parse(window_name).map(|wtype| window_bandwidth(wtype, n))
}

/// Generate center frequencies and sample rate pairs for multi-rate filterbank.
///
/// Returns center frequencies and corresponding sample rates for a pitch
/// filterbank covering MIDI notes 24-108 (C1 to C9).
///
/// Reference: Müller, Meinard. "Information Retrieval for Music and Motion."
/// Springer Verlag. 2007.
///
/// # Arguments
/// * `tuning` - Tuning deviation from A440, measured as a fraction of
///   the equally tempered semitone (1/12 of an octave).
///
/// # Returns
/// (center_freqs, sample_rates) tuple where:
/// - center_freqs: Center frequencies of the filter kernels
/// - sample_rates: Sample rate for each filter in the multirate filterbank
pub fn mr_frequencies(tuning: f32) -> (Vec<f32>, Vec<f32>) {
    // MIDI notes 24 to 108 (85 notes total)
    let center_freqs: Vec<f32> = (24..109)
        .map(|midi| midi_to_hz_single((midi as f32) + tuning))
        .collect();

    // Sample rates based on frequency range:
    // MIDI 24-59 (36 notes): 882 Hz
    // MIDI 60-93 (34 notes): 4410 Hz
    // MIDI 94-108 (15 notes): 22050 Hz
    let mut sample_rates = Vec::with_capacity(85);

    // First 36 notes (MIDI 24-59)
    sample_rates.extend(std::iter::repeat_n(882.0, 36));
    // Next 34 notes (MIDI 60-93)
    sample_rates.extend(std::iter::repeat_n(4410.0, 34));
    // Last 15 notes (MIDI 94-108)
    sample_rates.extend(std::iter::repeat_n(22050.0, 15));

    (center_freqs, sample_rates)
}

/// Compute relative bandwidth for a set of frequencies.
///
/// This is a helper function for wavelet basis construction.
fn relative_bandwidth(freqs: &[f32]) -> Vec<f32> {
    if freqs.len() <= 1 {
        return vec![0.0];
    }

    let n = freqs.len();
    let mut bpo = vec![0.0f32; n];
    let logf: Vec<f32> = freqs.iter().map(|f| f.log2()).collect();

    // Reflect at boundaries
    bpo[0] = 1.0 / (logf[1] - logf[0]);
    bpo[n - 1] = 1.0 / (logf[n - 1] - logf[n - 2]);

    // Centered difference for interior points
    for i in 1..(n - 1) {
        bpo[i] = 2.0 / (logf[i + 1] - logf[i - 1]);
    }

    // Compute relative bandwidths
    bpo.iter()
        .map(|&b| {
            let two_pow = 2.0f32.powf(2.0 / b);
            (two_pow - 1.0) / (two_pow + 1.0)
        })
        .collect()
}

/// Return length of each filter in a wavelet basis.
///
/// # Arguments
/// * `freqs` - Center frequencies of the filters (in Hz), must be in ascending order
/// * `sr` - Audio sampling rate
/// * `window` - Window type to use on filters
/// * `filter_scale` - Resolution of filter windows (larger values = longer windows)
/// * `gamma` - Bandwidth offset for variable-Q transforms (None for ERB-based)
/// * `alpha` - Optional pre-computed relative bandwidth parameter
///
/// # Returns
/// (lengths, f_cutoff) tuple where:
/// - lengths: The length of each filter
/// - f_cutoff: The lowest frequency at which all filters' main lobes have decayed by at least 3dB
///
/// # Errors
/// Returns an error if `freqs` is empty or `filter_scale` is not positive.
pub fn wavelet_lengths(
    freqs: &[f32],
    sr: f32,
    window: WindowType,
    filter_scale: f32,
    gamma: Option<f32>,
    alpha: Option<&[f32]>,
) -> crate::Result<(Vec<f32>, f32)> {
    if freqs.is_empty() {
        return Err(crate::Error::InvalidSize {
            name: "freqs",
            value: 0,
            reason: "frequency list must not be empty",
        });
    }
    if filter_scale <= 0.0 {
        return Err(crate::Error::InvalidParameter {
            name: "filter_scale",
            value: filter_scale.to_string(),
            reason: "filter_scale must be strictly positive".to_string(),
        });
    }

    // Compute alpha if not provided
    let alpha_vec: Vec<f32>;
    let alpha_slice = match alpha {
        Some(a) => a,
        None => {
            alpha_vec = relative_bandwidth(freqs);
            &alpha_vec
        }
    };

    // Compute gamma
    let gamma_vals: Vec<f32> = match gamma {
        Some(g) => vec![g; freqs.len()],
        None => {
            // ERB-based gamma: gamma[k] = 24.7 * alpha[k] / 0.108
            alpha_slice.iter().map(|a| a * 24.7 / 0.108).collect()
        }
    };

    // Compute Q factors
    let q_vals: Vec<f32> = alpha_slice.iter().map(|a| filter_scale / a).collect();

    // Get window bandwidth
    let win_bw = window_bandwidth(window, 1000);

    // Compute cutoff frequency
    let mut f_cutoff = 0.0f32;
    for i in 0..freqs.len() {
        let cutoff = freqs[i] * (1.0 + 0.5 * win_bw / q_vals[i]) + 0.5 * gamma_vals[i];
        f_cutoff = f_cutoff.max(cutoff);
    }

    // Compute filter lengths
    let lengths: Vec<f32> = freqs
        .iter()
        .enumerate()
        .map(|(i, &f)| q_vals[i] * sr / (f + gamma_vals[i] / alpha_slice[i]))
        .collect();

    Ok((lengths, f_cutoff))
}

/// Construct a wavelet basis using windowed complex sinusoids.
///
/// This function constructs a wavelet filterbank at a specified set of center frequencies.
///
/// # Arguments
/// * `freqs` - Center frequencies of the filters (in Hz), must be in ascending order
/// * `sr` - Audio sampling rate
/// * `window` - Window type to use on filters
/// * `filter_scale` - Scale of filter windows (smaller values = shorter windows)
/// * `pad_fft` - Center-pad all filters up to the nearest power of 2
/// * `norm` - Normalization type (L1, L2, or Max)
/// * `gamma` - Bandwidth offset for variable-Q transforms
/// * `alpha` - Optional pre-computed relative bandwidth parameter
///
/// # Returns
/// (filters, lengths) tuple where:
/// - filters: 2D array of complex filters (n_bins x max_len)
/// - lengths: Fractional length of each filter in samples
///
/// # Errors
/// Returns an error if `freqs` is empty or filter parameters are invalid.
#[allow(clippy::too_many_arguments)]
pub fn wavelet(
    freqs: &[f32],
    sr: f32,
    window: WindowType,
    filter_scale: f32,
    pad_fft: bool,
    norm: utils::NormType,
    gamma: Option<f32>,
    alpha: Option<&[f32]>,
) -> crate::Result<(Array2<Complex32>, Vec<f32>)> {
    if freqs.is_empty() {
        return Err(crate::Error::InvalidSize {
            name: "freqs",
            value: 0,
            reason: "frequency list must not be empty",
        });
    }

    // Get filter lengths
    let (lengths, _) = wavelet_lengths(freqs, sr, window, filter_scale, gamma, alpha)?;

    // Build filters
    let mut filters: Vec<Vec<Complex32>> = Vec::with_capacity(freqs.len());

    for (&freq, &ilen) in freqs.iter().zip(lengths.iter()) {
        let len_i = ilen.ceil() as usize;
        let half_len = (ilen / 2.0) as i32;

        // Build complex sinusoid
        let mut sig: Vec<Complex32> = Vec::with_capacity(len_i);
        for j in (-half_len)..(half_len + 1) {
            if sig.len() >= len_i {
                break;
            }
            let phase = (j as f32) * 2.0 * std::f32::consts::PI * freq / sr;
            sig.push(Complex32::new(phase.cos(), phase.sin()));
        }

        // Ensure correct length
        while sig.len() < len_i {
            sig.push(Complex32::new(0.0, 0.0));
        }
        sig.truncate(len_i);

        // Apply window
        let win = window::get_window(window, len_i);
        for (s, w) in sig.iter_mut().zip(win.iter()) {
            *s = Complex32::new(s.re * w, s.im * w);
        }

        // Normalize
        let norm_val = match norm {
            utils::NormType::L1 => sig
                .iter()
                .map(|c| (c.re * c.re + c.im * c.im).sqrt())
                .sum::<f32>(),
            utils::NormType::L2 => sig
                .iter()
                .map(|c| c.re * c.re + c.im * c.im)
                .sum::<f32>()
                .sqrt(),
            utils::NormType::Max => sig
                .iter()
                .map(|c| (c.re * c.re + c.im * c.im).sqrt())
                .fold(0.0f32, f32::max),
        };

        if norm_val > 1e-10 {
            for s in sig.iter_mut() {
                *s = Complex32::new(s.re / norm_val, s.im / norm_val);
            }
        }

        filters.push(sig);
    }

    // Determine max length (with optional FFT padding)
    let max_raw_len = lengths.iter().cloned().fold(0.0f32, f32::max);
    let max_len = if pad_fft {
        (max_raw_len.ceil() as usize).next_power_of_two()
    } else {
        max_raw_len.ceil() as usize
    };

    // Pad and stack filters
    let mut result = Array2::<Complex32>::zeros((freqs.len(), max_len));
    for (i, filt) in filters.iter().enumerate() {
        let filt_len = filt.len();
        let pad_start = (max_len - filt_len) / 2;
        for (j, &val) in filt.iter().enumerate() {
            if pad_start + j < max_len {
                result[(i, pad_start + j)] = val;
            }
        }
    }

    Ok((result, lengths))
}

/// IIR filter coefficients representation.
#[derive(Debug, Clone)]
pub enum FilterCoeffs {
    /// Transfer function form (numerator, denominator)
    BA { b: Vec<f32>, a: Vec<f32> },
    /// Second-order sections form
    SOS(Vec<[f32; 6]>),
}

/// Design a simple IIR band-pass filter using a biquad cascade.
///
/// This is a simplified version that uses cascaded biquad filters
/// instead of scipy's iirdesign.
///
/// # Arguments
/// * `center_freq` - Center frequency (Hz)
/// * `bandwidth` - Filter bandwidth (Hz)
/// * `sample_rate` - Sample rate (Hz)
/// * `order` - Filter order (number of biquad stages)
///
/// # Returns
/// Filter coefficients as second-order sections
fn design_bandpass_sos(
    center_freq: f32,
    bandwidth: f32,
    sample_rate: f32,
    order: usize,
) -> Vec<[f32; 6]> {
    // Normalize frequencies
    let w0 = 2.0 * std::f32::consts::PI * center_freq / sample_rate;
    let bw = 2.0 * std::f32::consts::PI * bandwidth / sample_rate;

    // Q factor
    let q = w0 / bw;

    // Design bandpass biquad
    let alpha = w0.sin() / (2.0 * q);
    let cos_w0 = w0.cos();

    // Bandpass coefficients (constant 0 dB peak gain)
    let b0 = alpha;
    let b1 = 0.0;
    let b2 = -alpha;
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * cos_w0;
    let a2 = 1.0 - alpha;

    // Normalize by a0
    let sos = [b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0];

    // Cascade multiple stages for higher order
    vec![sos; order]
}

/// Construct a multi-rate bank of IIR band-pass filters.
///
/// By default, center frequencies are set to the 85 fundamental frequencies
/// of MIDI notes 24-108 (C1 to C9), tuned to A440 equal temperament.
///
/// Reference: Müller, Meinard. "Information Retrieval for Music and Motion."
/// Springer Verlag. 2007.
///
/// # Arguments
/// * `center_freqs` - Center frequencies of the filter kernels (optional)
/// * `tuning` - Tuning deviation from A440 as a fraction of a semitone
/// * `sample_rates` - Sample rates for each filter (optional)
/// * `q` - Q factor (influences filter bandwidth)
/// * `order` - Filter order (number of biquad stages per filter)
///
/// # Returns
/// (filterbank, sample_rates) tuple where:
/// - filterbank: List of filter coefficients (as SOS)
/// - sample_rates: Sample rate for each filter
pub fn semitone_filterbank(
    center_freqs: Option<&[f32]>,
    tuning: f32,
    sample_rates: Option<&[f32]>,
    q: f32,
    order: usize,
) -> crate::Result<(Vec<FilterCoeffs>, Vec<f32>)> {
    // Get frequencies and sample rates
    let (freqs, srs) = match (center_freqs, sample_rates) {
        (Some(f), Some(s)) => (f.to_vec(), s.to_vec()),
        _ => mr_frequencies(tuning),
    };

    if freqs.len() != srs.len() {
        return Err(crate::Error::ShapeMismatch {
            expected: format!("center_freqs.len() == sample_rates.len() ({})", freqs.len()),
            got: format!("sample_rates.len() == {}", srs.len()),
        });
    }

    // Build filterbank
    let mut filterbank = Vec::with_capacity(freqs.len());

    for (&freq, &sr) in freqs.iter().zip(srs.iter()) {
        let bandwidth = freq / q;
        let sos = design_bandpass_sos(freq, bandwidth, sr, order);
        filterbank.push(FilterCoeffs::SOS(sos));
    }

    Ok((filterbank, srs))
}

/// Apply a single SOS filter section to a signal.
fn apply_sos_section(x: &[f32], sos: &[f32; 6]) -> Vec<f32> {
    let b0 = sos[0];
    let b1 = sos[1];
    let b2 = sos[2];
    let a1 = sos[4];
    let a2 = sos[5];

    let mut y = vec![0.0f32; x.len()];
    let mut x1 = 0.0f32;
    let mut x2 = 0.0f32;
    let mut y1 = 0.0f32;
    let mut y2 = 0.0f32;

    for (i, &xi) in x.iter().enumerate() {
        y[i] = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
        x2 = x1;
        x1 = xi;
        y2 = y1;
        y1 = y[i];
    }

    y
}

/// Apply SOS filter (forward only).
pub fn sosfilt(sos: &[[f32; 6]], x: &[f32]) -> Vec<f32> {
    let mut y = x.to_vec();
    for section in sos {
        y = apply_sos_section(&y, section);
    }
    y
}

/// Apply SOS filter forward and backward (zero-phase filtering).
pub fn sosfiltfilt(sos: &[[f32; 6]], x: &[f32]) -> Vec<f32> {
    // Forward pass
    let y_forward = sosfilt(sos, x);

    // Reverse
    let mut y_rev: Vec<f32> = y_forward.into_iter().rev().collect();

    // Backward pass
    y_rev = sosfilt(sos, &y_rev);

    // Reverse again
    y_rev.into_iter().rev().collect()
}

/// Compute constant-Q filter lengths.
///
/// Returns the length of each filter in a constant-Q basis.
///
/// # Arguments
/// * `sr` - Audio sampling rate
/// * `fmin` - Minimum frequency bin (default: C1 ~= 32.70 Hz)
/// * `n_bins` - Number of frequencies (default: 84, i.e., 7 octaves)
/// * `bins_per_octave` - Number of bins per octave (default: 12)
/// * `window` - Window type to use on filters
/// * `filter_scale` - Resolution of filter windows (default: 1.0)
/// * `gamma` - Bandwidth offset for variable-Q transforms (default: 0.0)
///
/// # Returns
/// Filter lengths for each frequency bin
///
/// # Errors
/// Returns an error if parameters are invalid or if maximum frequency exceeds Nyquist
///
/// # Notes
/// This is the legacy API compatible with librosa. For new code, consider using
/// [`wavelet_lengths`] instead.
pub fn constant_q_lengths(
    sr: f32,
    fmin: Option<f32>,
    n_bins: usize,
    bins_per_octave: usize,
    window: WindowType,
    filter_scale: f32,
    gamma: f32,
) -> crate::Result<Vec<f32>> {
    if sr <= 0.0 {
        return Err(crate::Error::InvalidParameter {
            name: "sr",
            value: sr.to_string(),
            reason: "sample rate must be positive".to_string(),
        });
    }

    let fmin = fmin.unwrap_or(32.7); // C1 ~= 32.70 Hz

    if fmin <= 0.0 {
        return Err(crate::Error::InvalidParameter {
            name: "fmin",
            value: fmin.to_string(),
            reason: "fmin must be strictly positive".to_string(),
        });
    }

    if bins_per_octave == 0 {
        return Err(crate::Error::InvalidSize {
            name: "bins_per_octave",
            value: 0,
            reason: "bins_per_octave must be positive",
        });
    }

    if filter_scale <= 0.0 {
        return Err(crate::Error::InvalidParameter {
            name: "filter_scale",
            value: filter_scale.to_string(),
            reason: "filter_scale must be positive".to_string(),
        });
    }

    if n_bins == 0 {
        return Err(crate::Error::InvalidSize {
            name: "n_bins",
            value: 0,
            reason: "n_bins must be a positive integer",
        });
    }

    // Compute the frequencies
    let ratio = 2.0f32.powf(1.0 / bins_per_octave as f32);
    let freqs: Vec<f32> = (0..n_bins).map(|i| fmin * ratio.powi(i as i32)).collect();

    // Compute alpha (relative bandwidth)
    let two_pow = 2.0f32.powf(2.0 / bins_per_octave as f32);
    let alpha = (two_pow - 1.0) / (two_pow + 1.0);

    // Q factor
    let q = filter_scale / alpha;

    // Check Nyquist
    let win_bw = window_bandwidth(window, 1000);
    let max_freq = freqs.iter().cloned().fold(0.0f32, f32::max);
    if max_freq * (1.0 + 0.5 * win_bw / q) > sr / 2.0 {
        return Err(crate::Error::InvalidFrequencyRange {
            fmin,
            fmax: max_freq,
            reason: format!(
                "maximum filter frequency {:.2} would exceed Nyquist {}",
                max_freq,
                sr / 2.0
            ),
        });
    }

    // Convert frequencies to filter lengths
    let lengths: Vec<f32> = freqs
        .iter()
        .map(|&f| q * sr / (f + gamma / alpha))
        .collect();

    Ok(lengths)
}

/// Construct a constant-Q basis.
///
/// This function constructs a filter bank similar to Morlet wavelets,
/// where complex exponentials are windowed to different lengths
/// such that the number of cycles remains fixed for all frequencies.
///
/// # Arguments
/// * `sr` - Audio sampling rate
/// * `fmin` - Minimum frequency bin (default: C1 ~= 32.70 Hz)
/// * `n_bins` - Number of frequencies (default: 84)
/// * `bins_per_octave` - Number of bins per octave (default: 12)
/// * `window` - Window type to use on filters
/// * `filter_scale` - Scale of filter windows (default: 1.0)
/// * `pad_fft` - Center-pad all filters up to the nearest power of 2 (default: true)
/// * `norm` - Normalization type (default: L1)
/// * `gamma` - Bandwidth offset for variable-Q transforms (default: 0.0)
///
/// # Returns
/// (filters, lengths) tuple where:
/// - filters: 2D array of complex filters (n_bins x max_len)
/// - lengths: The (fractional) length of each filter
///
/// # Errors
/// Returns an error if parameters are invalid
///
/// # Notes
/// This is the legacy API compatible with librosa. For new code, consider using
/// [`wavelet`] instead.
#[allow(clippy::too_many_arguments)]
pub fn constant_q(
    sr: f32,
    fmin: Option<f32>,
    n_bins: usize,
    bins_per_octave: usize,
    window: WindowType,
    filter_scale: f32,
    pad_fft: bool,
    norm: utils::NormType,
    gamma: f32,
) -> crate::Result<(Array2<Complex32>, Vec<f32>)> {
    if sr <= 0.0 {
        return Err(crate::Error::InvalidParameter {
            name: "sr",
            value: sr.to_string(),
            reason: "sample rate must be positive".to_string(),
        });
    }

    let fmin = fmin.unwrap_or(32.7); // C1 ~= 32.70 Hz

    // Get filter lengths
    let lengths = constant_q_lengths(
        sr,
        Some(fmin),
        n_bins,
        bins_per_octave,
        window,
        filter_scale,
        gamma,
    )?;

    // Compute frequencies
    let ratio = 2.0f32.powf(1.0 / bins_per_octave as f32);
    let freqs: Vec<f32> = (0..n_bins).map(|i| fmin * ratio.powi(i as i32)).collect();

    // Build the filters
    let mut filters: Vec<Vec<Complex32>> = Vec::with_capacity(n_bins);

    for (&ilen, &freq) in lengths.iter().zip(freqs.iter()) {
        let len_i = ilen.ceil() as usize;
        let half_len = (ilen / 2.0) as i32;

        // Build the filter: complex sinusoid
        let mut sig: Vec<Complex32> = Vec::with_capacity(len_i);
        for j in (-half_len)..(half_len + 1) {
            if sig.len() >= len_i {
                break;
            }
            let phase = (j as f32) * 2.0 * std::f32::consts::PI * freq / sr;
            sig.push(utils::phasor(phase));
        }

        // Ensure correct length
        while sig.len() < len_i {
            sig.push(Complex32::new(0.0, 0.0));
        }
        sig.truncate(len_i);

        // Apply window
        let win = window::get_window(window, len_i);
        for (s, w) in sig.iter_mut().zip(win.iter()) {
            *s = Complex32::new(s.re * w, s.im * w);
        }

        // Normalize
        let norm_val = match norm {
            utils::NormType::L1 => sig.iter().map(|c| c.norm()).sum::<f32>(),
            utils::NormType::L2 => sig.iter().map(|c| c.norm_sqr()).sum::<f32>().sqrt(),
            utils::NormType::Max => sig.iter().map(|c| c.norm()).fold(0.0f32, f32::max),
        };

        if norm_val > 1e-10 {
            for s in sig.iter_mut() {
                *s = Complex32::new(s.re / norm_val, s.im / norm_val);
            }
        }

        filters.push(sig);
    }

    // Determine max length
    let max_raw_len = lengths.iter().cloned().fold(0.0f32, f32::max);
    let max_len = if pad_fft {
        (max_raw_len.ceil() as usize).next_power_of_two()
    } else {
        max_raw_len.ceil() as usize
    };

    // Pad and stack filters
    let mut result = Array2::<Complex32>::zeros((n_bins, max_len));
    for (i, filt) in filters.iter().enumerate() {
        let filt_len = filt.len();
        let pad_start = (max_len - filt_len) / 2;
        for (j, &val) in filt.iter().enumerate() {
            if pad_start + j < max_len {
                result[(i, pad_start + j)] = val;
            }
        }
    }

    Ok((result, lengths))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_window_bandwidth() {
        // Hann window should have ENBW around 1.5
        let bw = window_bandwidth(WindowType::Hann, 1000);
        assert!(bw > 1.4 && bw < 1.6, "Hann ENBW: {}", bw);

        // Hamming window should have ENBW around 1.36
        let bw = window_bandwidth(WindowType::Hamming, 1000);
        assert!(bw > 1.3 && bw < 1.5, "Hamming ENBW: {}", bw);
    }

    #[test]
    fn test_mr_frequencies() {
        let (freqs, srs) = mr_frequencies(0.0);

        // Should have 85 frequencies (MIDI 24-108)
        assert_eq!(freqs.len(), 85);
        assert_eq!(srs.len(), 85);

        // First frequency should be C1 (MIDI 24) ~= 32.7 Hz
        assert_relative_eq!(freqs[0], 32.7, epsilon = 0.1);

        // Last frequency should be C9 (MIDI 108) ~= 4186 Hz
        assert_relative_eq!(freqs[84], 4186.0, epsilon = 1.0);

        // Check sample rates
        assert_eq!(srs[0], 882.0); // First 36
        assert_eq!(srs[35], 882.0);
        assert_eq!(srs[36], 4410.0); // Next 34
        assert_eq!(srs[69], 4410.0);
        assert_eq!(srs[70], 22050.0); // Last 15
        assert_eq!(srs[84], 22050.0);
    }

    #[test]
    fn test_wavelet_lengths() {
        // Create some CQT-like frequencies
        let fmin = 32.7; // C1
        let n_bins = 12;
        let freqs: Vec<f32> = (0..n_bins)
            .map(|i| fmin * 2.0f32.powf(i as f32 / 12.0))
            .collect();

        let (lengths, f_cutoff) =
            wavelet_lengths(&freqs, 22050.0, WindowType::Hann, 1.0, Some(0.0), None).unwrap();

        assert_eq!(lengths.len(), n_bins);

        // Lengths should decrease as frequency increases
        for i in 1..lengths.len() {
            assert!(
                lengths[i] < lengths[i - 1],
                "Length {} ({}) should be less than {} ({})",
                i,
                lengths[i],
                i - 1,
                lengths[i - 1]
            );
        }

        // Cutoff should be above the highest frequency
        assert!(f_cutoff > freqs[n_bins - 1]);
    }

    #[test]
    fn test_wavelet() {
        let fmin = 65.4; // C2
        let n_bins = 12;
        let freqs: Vec<f32> = (0..n_bins)
            .map(|i| fmin * 2.0f32.powf(i as f32 / 12.0))
            .collect();

        let (filters, lengths) = wavelet(
            &freqs,
            22050.0,
            WindowType::Hann,
            1.0,
            true,
            utils::NormType::L2,
            Some(0.0),
            None,
        )
        .unwrap();

        assert_eq!(filters.nrows(), n_bins);
        assert_eq!(lengths.len(), n_bins);

        // Filter length should be power of 2 (pad_fft=true)
        let ncols = filters.ncols();
        assert!(ncols.is_power_of_two());

        // Check normalization (L2 norm should be ~1)
        for i in 0..n_bins {
            let norm: f32 = filters
                .row(i)
                .iter()
                .map(|c| c.re * c.re + c.im * c.im)
                .sum::<f32>()
                .sqrt();
            assert_relative_eq!(norm, 1.0, epsilon = 0.1);
        }
    }

    #[test]
    fn test_semitone_filterbank() {
        let (filterbank, srs) = semitone_filterbank(None, 0.0, None, 25.0, 2).unwrap();

        // Should have 85 filters
        assert_eq!(filterbank.len(), 85);
        assert_eq!(srs.len(), 85);

        // Check that all filters are SOS format
        for filt in &filterbank {
            match filt {
                FilterCoeffs::SOS(sos) => {
                    assert_eq!(sos.len(), 2); // order=2
                }
                _ => panic!("Expected SOS format"),
            }
        }
    }

    #[test]
    fn test_sosfilt() {
        // Simple low-pass biquad
        let sos = [[0.1, 0.2, 0.1, 1.0, -0.5, 0.1]];
        let x: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();

        let y = sosfilt(&sos, &x);
        assert_eq!(y.len(), x.len());
    }

    #[test]
    fn test_sosfiltfilt() {
        // Zero-phase filtering should not introduce phase shift
        let sos = [[0.1, 0.2, 0.1, 1.0, -0.5, 0.1]];
        let x: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();

        let y = sosfiltfilt(&sos, &x);
        assert_eq!(y.len(), x.len());
    }

    #[test]
    fn test_constant_q_lengths() {
        // Default parameters: 84 bins, 12 bins per octave, starting at C1
        let lengths = constant_q_lengths(
            22050.0,
            None, // fmin defaults to C1 ~= 32.7 Hz
            84,
            12,
            WindowType::Hann,
            1.0,
            0.0,
        )
        .unwrap();

        // Should have 84 lengths
        assert_eq!(lengths.len(), 84);

        // Lengths should decrease as frequency increases
        for i in 1..lengths.len() {
            assert!(
                lengths[i] < lengths[i - 1],
                "Length {} ({}) should be less than {} ({})",
                i,
                lengths[i],
                i - 1,
                lengths[i - 1]
            );
        }

        // First length should be large (low frequency)
        assert!(
            lengths[0] > 100.0,
            "First length should be large: {}",
            lengths[0]
        );

        // Last length should be smaller than first (high frequency vs low frequency)
        assert!(
            lengths[83] < lengths[0],
            "Last length should be smaller than first: {} vs {}",
            lengths[83],
            lengths[0]
        );
    }

    #[test]
    fn test_constant_q_lengths_error_cases() {
        // Invalid sample rate
        assert!(constant_q_lengths(0.0, None, 84, 12, WindowType::Hann, 1.0, 0.0).is_err());

        // Invalid fmin
        assert!(
            constant_q_lengths(22050.0, Some(0.0), 84, 12, WindowType::Hann, 1.0, 0.0).is_err()
        );

        // Invalid bins_per_octave
        assert!(constant_q_lengths(22050.0, None, 84, 0, WindowType::Hann, 1.0, 0.0).is_err());

        // Invalid filter_scale
        assert!(constant_q_lengths(22050.0, None, 84, 12, WindowType::Hann, 0.0, 0.0).is_err());

        // Invalid n_bins
        assert!(constant_q_lengths(22050.0, None, 0, 12, WindowType::Hann, 1.0, 0.0).is_err());

        // Frequency exceeding Nyquist
        assert!(constant_q_lengths(100.0, Some(60.0), 84, 12, WindowType::Hann, 1.0, 0.0).is_err());
    }

    #[test]
    fn test_constant_q() {
        // Default parameters
        let (filters, lengths) = constant_q(
            22050.0,
            None,
            84,
            12,
            WindowType::Hann,
            1.0,
            true, // pad_fft
            utils::NormType::L1,
            0.0,
        )
        .unwrap();

        // Should have 84 filters
        assert_eq!(filters.nrows(), 84);
        assert_eq!(lengths.len(), 84);

        // Filter length should be power of 2 (pad_fft=true)
        let ncols = filters.ncols();
        assert!(
            ncols.is_power_of_two(),
            "Filter length should be power of 2: {}",
            ncols
        );

        // Check normalization (L1 norm should be ~1)
        for i in 0..84 {
            let norm: f32 = filters.row(i).iter().map(|c| c.norm()).sum();
            assert_relative_eq!(norm, 1.0, epsilon = 0.1);
        }
    }

    #[test]
    fn test_constant_q_no_pad() {
        // Without FFT padding
        let (filters, lengths) = constant_q(
            22050.0,
            None,
            12, // fewer bins for simpler test
            12,
            WindowType::Hann,
            1.0,
            false, // no pad_fft
            utils::NormType::L2,
            0.0,
        )
        .unwrap();

        assert_eq!(filters.nrows(), 12);

        // Without padding, max length should just be ceil(max_length)
        let max_len = lengths.iter().cloned().fold(0.0f32, f32::max).ceil() as usize;
        assert_eq!(filters.ncols(), max_len);
    }

    #[test]
    fn test_constant_q_with_gamma() {
        // Variable-Q transform (gamma > 0)
        let (filters, lengths) = constant_q(
            22050.0,
            None,
            24,
            12,
            WindowType::Hann,
            1.0,
            true,
            utils::NormType::L1,
            10.0, // gamma > 0 for variable-Q
        )
        .unwrap();

        assert_eq!(filters.nrows(), 24);
        assert_eq!(lengths.len(), 24);

        // Check normalization
        for i in 0..24 {
            let norm: f32 = filters.row(i).iter().map(|c| c.norm()).sum();
            assert_relative_eq!(norm, 1.0, epsilon = 0.1);
        }
    }
}
