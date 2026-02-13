/// Harmonic calculations for frequency representations.
///
/// This module provides functions for computing harmonic energy,
/// salience, and f0-based harmonic extraction.
use ndarray::{Array2, Array3};

/// Interpolation kind for harmonic estimation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InterpKind {
    /// Linear interpolation between adjacent points
    Linear,
    /// Nearest neighbor (no interpolation)
    Nearest,
}

/// Perform 1D linear interpolation.
///
/// Given sorted x coordinates and corresponding y values, interpolate
/// at the target x positions.
fn interp1d(x: &[f32], y: &[f32], x_new: &[f32], fill_value: f32, kind: InterpKind) -> Vec<f32> {
    if x.is_empty() || y.is_empty() || x.len() != y.len() {
        return vec![fill_value; x_new.len()];
    }

    let n = x.len();
    x_new
        .iter()
        .map(|&xn| {
            // Handle out of bounds
            if xn < x[0] || xn > x[n - 1] {
                return fill_value;
            }

            // Binary search to find the interval
            let mut lo = 0;
            let mut hi = n - 1;
            while lo < hi - 1 {
                let mid = (lo + hi) / 2;
                if x[mid] <= xn {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }

            match kind {
                InterpKind::Linear => {
                    // Linear interpolation
                    if (x[hi] - x[lo]).abs() < 1e-10 {
                        y[lo]
                    } else {
                        let t = (xn - x[lo]) / (x[hi] - x[lo]);
                        y[lo] * (1.0 - t) + y[hi] * t
                    }
                }
                InterpKind::Nearest => {
                    // Nearest neighbor
                    if (xn - x[lo]).abs() <= (x[hi] - xn).abs() {
                        y[lo]
                    } else {
                        y[hi]
                    }
                }
            }
        })
        .collect()
}

/// Compute the energy at harmonics of a time-frequency representation.
///
/// Given a frequency-based energy representation (e.g., spectrogram),
/// this function computes the energy at the chosen harmonics of the
/// frequency axis. The resulting harmonic array can be used as input
/// to a salience computation.
///
/// # Arguments
/// * `x` - Input time-frequency representation (n_freq x n_frames)
/// * `freqs` - Frequency values corresponding to x's rows
/// * `harmonics` - Harmonics to compute (e.g., [1, 2, 3] for fundamental + overtones)
/// * `kind` - Interpolation type
/// * `fill_value` - Value to fill when extrapolating beyond frequency range
///
/// # Returns
/// Harmonic array (n_harmonics x n_freq x n_frames) where `result[h]`
/// contains the energy at `harmonics[h] * freq` for each frequency
///
/// # Example
/// ```
/// use giggle::harmonic::{interp_harmonics, InterpKind};
/// use giggle::spectrum::{stft, StftConfig};
/// use giggle::convert::fft_frequencies;
/// use giggle::io;
///
/// let signal = io::tone(440.0, 22050, 0.5);
/// let stft_result = stft(&signal, &StftConfig::default()).unwrap();
///
/// // Get magnitude
/// let n_freq = stft_result.shape()[0];
/// let n_frames = stft_result.shape()[1];
/// let mut mag = ndarray::Array2::<f32>::zeros((n_freq, n_frames));
/// for f in 0..n_freq {
///     for t in 0..n_frames {
///         mag[(f, t)] = stft_result[(f, t)].norm();
///     }
/// }
///
/// let freqs = fft_frequencies(22050, 2048);
/// let harmonics = vec![1.0, 2.0, 3.0];
/// let harm = interp_harmonics(&mag, &freqs, &harmonics, InterpKind::Linear, 0.0);
/// assert_eq!(harm.shape()[0], 3); // 3 harmonics
/// ```
pub fn interp_harmonics(
    x: &Array2<f32>,
    freqs: &[f32],
    harmonics: &[f32],
    kind: InterpKind,
    fill_value: f32,
) -> Array3<f32> {
    let n_freq = x.shape()[0];
    let n_frames = x.shape()[1];
    let n_harmonics = harmonics.len();

    if n_freq == 0 || n_frames == 0 || n_harmonics == 0 || freqs.len() != n_freq {
        return Array3::zeros((n_harmonics, n_freq, n_frames));
    }

    let mut result = Array3::<f32>::zeros((n_harmonics, n_freq, n_frames));

    // For each frame, interpolate at harmonic frequencies
    for frame in 0..n_frames {
        // Extract column as y values for interpolation
        let y: Vec<f32> = (0..n_freq).map(|f| x[(f, frame)]).collect();

        // For each harmonic
        for (h_idx, &h) in harmonics.iter().enumerate() {
            // Compute target frequencies: h * freqs
            let target_freqs: Vec<f32> = freqs.iter().map(|&f| h * f).collect();

            // Interpolate
            let interp_vals = interp1d(freqs, &y, &target_freqs, fill_value, kind);

            // Store result
            for (f_idx, &val) in interp_vals.iter().enumerate() {
                result[(h_idx, f_idx, frame)] = val;
            }
        }
    }

    result
}

/// Find local maxima along the frequency axis.
fn find_local_maxima(x: &Array2<f32>) -> Vec<Vec<usize>> {
    let n_freq = x.shape()[0];
    let n_frames = x.shape()[1];

    let mut peaks = Vec::with_capacity(n_frames);

    for frame in 0..n_frames {
        let mut frame_peaks = Vec::new();

        for f in 1..n_freq.saturating_sub(1) {
            let val = x[(f, frame)];
            if val > x[(f - 1, frame)] && val > x[(f + 1, frame)] {
                frame_peaks.push(f);
            }
        }

        peaks.push(frame_peaks);
    }

    peaks
}

/// Harmonic salience function.
///
/// Computes a weighted sum of harmonic energies at each frequency.
/// This is useful for enhancing pitched content and identifying
/// harmonic structures in audio.
///
/// # Arguments
/// * `s` - Input magnitude spectrogram (n_freq x n_frames)
/// * `freqs` - Frequency values for each row
/// * `harmonics` - Harmonics to include (e.g., [1, 2, 3, 4])
/// * `weights` - Weight for each harmonic (None for uniform)
/// * `filter_peaks` - If true, only return salience at local maxima
/// * `fill_value` - Value for non-peak positions (if filter_peaks=true)
/// * `kind` - Interpolation kind
///
/// # Returns
/// Salience spectrogram with same shape as input
///
/// # Example
/// ```
/// use giggle::harmonic::{salience, InterpKind};
/// use giggle::spectrum::{stft, StftConfig};
/// use giggle::convert::fft_frequencies;
/// use giggle::io;
///
/// let signal = io::tone(440.0, 22050, 0.5);
/// let stft_result = stft(&signal, &StftConfig::default()).unwrap();
///
/// let n_freq = stft_result.shape()[0];
/// let n_frames = stft_result.shape()[1];
/// let mut mag = ndarray::Array2::<f32>::zeros((n_freq, n_frames));
/// for f in 0..n_freq {
///     for t in 0..n_frames {
///         mag[(f, t)] = stft_result[(f, t)].norm();
///     }
/// }
///
/// let freqs = fft_frequencies(22050, 2048);
/// let harmonics = vec![1.0, 2.0, 3.0, 4.0];
/// let weights = vec![1.0, 0.5, 0.33, 0.25];
/// let sal = salience(&mag, &freqs, &harmonics, Some(&weights), false, 0.0, InterpKind::Linear);
/// assert_eq!(sal.shape(), mag.shape());
/// ```
pub fn salience(
    s: &Array2<f32>,
    freqs: &[f32],
    harmonics: &[f32],
    weights: Option<&[f32]>,
    filter_peaks: bool,
    fill_value: f32,
    kind: InterpKind,
) -> Array2<f32> {
    let n_freq = s.shape()[0];
    let n_frames = s.shape()[1];
    let n_harmonics = harmonics.len();

    if n_freq == 0 || n_frames == 0 || n_harmonics == 0 {
        return Array2::zeros((n_freq, n_frames));
    }

    // Get harmonic energies
    let s_harm = interp_harmonics(s, freqs, harmonics, kind, 0.0);

    // Compute weights
    let w: Vec<f32> = match weights {
        Some(wts) if wts.len() == n_harmonics => wts.to_vec(),
        _ => vec![1.0; n_harmonics],
    };
    let w_sum: f32 = w.iter().sum();

    // Compute weighted average across harmonics
    let mut s_sal = Array2::<f32>::zeros((n_freq, n_frames));

    for frame in 0..n_frames {
        for freq in 0..n_freq {
            let mut weighted_sum = 0.0f32;
            for (h_idx, &wt) in w.iter().enumerate() {
                weighted_sum += s_harm[(h_idx, freq, frame)] * wt;
            }
            s_sal[(freq, frame)] = if w_sum > 1e-10 {
                weighted_sum / w_sum
            } else {
                weighted_sum
            };
        }
    }

    // Optionally filter to peaks only
    if filter_peaks {
        let peaks = find_local_maxima(s);
        let mut s_out = Array2::from_elem((n_freq, n_frames), fill_value);

        for (frame, frame_peaks) in peaks.iter().enumerate() {
            for &peak_idx in frame_peaks {
                s_out[(peak_idx, frame)] = s_sal[(peak_idx, frame)];
            }
        }

        s_out
    } else {
        s_sal
    }
}

/// Compute the energy at selected harmonics of a time-varying fundamental frequency.
///
/// This function reduces a frequency × time representation to a harmonic × time
/// representation, effectively normalizing for the fundamental frequency.
/// Useful for representing timbre (when f0 is pitch) or rhythm (when f0 is tempo).
///
/// # Arguments
/// * `x` - Input magnitude spectrogram (n_freq x n_frames)
/// * `f0` - Fundamental frequency for each frame (n_frames)
/// * `freqs` - Frequency values for each row of x
/// * `harmonics` - Harmonics to compute (e.g., [1, 2, 3, ...])
/// * `kind` - Interpolation type
/// * `fill_value` - Value for out-of-range or NaN frequencies
///
/// # Returns
/// Harmonic representation (n_harmonics x n_frames)
///
/// # Example
/// ```
/// use giggle::harmonic::{f0_harmonics, InterpKind};
/// use giggle::spectrum::{stft, StftConfig};
/// use giggle::convert::fft_frequencies;
/// use giggle::io;
///
/// let signal = io::tone(440.0, 22050, 0.5);
/// let stft_result = stft(&signal, &StftConfig::default()).unwrap();
///
/// let n_freq = stft_result.shape()[0];
/// let n_frames = stft_result.shape()[1];
/// let mut mag = ndarray::Array2::<f32>::zeros((n_freq, n_frames));
/// for f in 0..n_freq {
///     for t in 0..n_frames {
///         mag[(f, t)] = stft_result[(f, t)].norm();
///     }
/// }
///
/// let freqs = fft_frequencies(22050, 2048);
/// // Assume constant f0 of 440 Hz
/// let f0: Vec<f32> = vec![440.0; n_frames];
/// let harmonics: Vec<f32> = (1..=12).map(|i| i as f32).collect();
/// let f0_harm = f0_harmonics(&mag, &f0, &freqs, &harmonics, InterpKind::Linear, 0.0).unwrap();
/// assert_eq!(f0_harm.shape()[0], 12); // 12 harmonics
/// assert_eq!(f0_harm.shape()[1], n_frames);
/// ```
pub fn f0_harmonics(
    x: &Array2<f32>,
    f0: &[f32],
    freqs: &[f32],
    harmonics: &[f32],
    kind: InterpKind,
    fill_value: f32,
) -> crate::Result<Array2<f32>> {
    let n_freq = x.shape()[0];
    let n_frames = x.shape()[1];
    let n_harmonics = harmonics.len();

    if n_freq == 0 || n_frames == 0 || n_harmonics == 0 {
        return Ok(Array2::zeros((n_harmonics, n_frames)));
    }

    if freqs.len() != n_freq {
        return Err(crate::Error::ShapeMismatch {
            expected: format!("freqs.len() == {} (n_freq)", n_freq),
            got: format!("freqs.len() == {}", freqs.len()),
        });
    }

    if f0.len() != n_frames {
        return Err(crate::Error::ShapeMismatch {
            expected: format!("f0.len() == {} (n_frames)", n_frames),
            got: format!("f0.len() == {}", f0.len()),
        });
    }

    let mut result = Array2::<f32>::zeros((n_harmonics, n_frames));

    // For each frame
    for frame in 0..n_frames {
        let f0_val = f0[frame];

        // Skip if f0 is NaN or invalid
        if !f0_val.is_finite() || f0_val <= 0.0 {
            for h_idx in 0..n_harmonics {
                result[(h_idx, frame)] = fill_value;
            }
            continue;
        }

        // Extract column as y values for interpolation
        let y: Vec<f32> = (0..n_freq).map(|f| x[(f, frame)]).collect();

        // Compute target frequencies: harmonics[h] * f0
        let target_freqs: Vec<f32> = harmonics.iter().map(|&h| h * f0_val).collect();

        // Interpolate
        let interp_vals = interp1d(freqs, &y, &target_freqs, fill_value, kind);

        // Store result
        for (h_idx, &val) in interp_vals.iter().enumerate() {
            result[(h_idx, frame)] = if val.is_finite() { val } else { fill_value };
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interp1d_linear() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 2.0, 4.0, 6.0, 8.0];
        let x_new = vec![0.5, 1.5, 2.5, 3.5];

        let result = interp1d(&x, &y, &x_new, -1.0, InterpKind::Linear);

        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - 3.0).abs() < 1e-6);
        assert!((result[2] - 5.0).abs() < 1e-6);
        assert!((result[3] - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_interp1d_out_of_bounds() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![10.0, 20.0, 30.0];
        let x_new = vec![0.0, 4.0]; // Out of bounds

        let result = interp1d(&x, &y, &x_new, -999.0, InterpKind::Linear);

        assert_eq!(result[0], -999.0);
        assert_eq!(result[1], -999.0);
    }

    #[test]
    fn test_interp1d_nearest() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 10.0, 20.0];
        let x_new = vec![0.3, 0.7, 1.4, 1.6];

        let result = interp1d(&x, &y, &x_new, -1.0, InterpKind::Nearest);

        assert_eq!(result[0], 0.0); // Closer to 0
        assert_eq!(result[1], 10.0); // Closer to 1
        assert_eq!(result[2], 10.0); // Closer to 1
        assert_eq!(result[3], 20.0); // Closer to 2
    }

    #[test]
    fn test_interp_harmonics_shape() {
        use crate::convert::fft_frequencies;
        use crate::io;
        use crate::spectrum::{StftConfig, stft};

        let signal = io::tone(440.0, 22050, 0.3);
        let stft_result = stft(&signal, &StftConfig::default()).unwrap();

        let n_freq = stft_result.shape()[0];
        let n_frames = stft_result.shape()[1];
        let mut mag = Array2::<f32>::zeros((n_freq, n_frames));
        for f in 0..n_freq {
            for t in 0..n_frames {
                mag[(f, t)] = stft_result[(f, t)].norm();
            }
        }

        let freqs = fft_frequencies(22050, 2048);
        let harmonics = vec![1.0, 2.0, 3.0];
        let harm = interp_harmonics(&mag, &freqs, &harmonics, InterpKind::Linear, 0.0);

        assert_eq!(harm.shape()[0], 3);
        assert_eq!(harm.shape()[1], n_freq);
        assert_eq!(harm.shape()[2], n_frames);
    }

    #[test]
    fn test_interp_harmonics_fundamental() {
        use crate::convert::fft_frequencies;
        use crate::io;
        use crate::spectrum::{StftConfig, stft};

        let signal = io::tone(440.0, 22050, 0.3);
        let stft_result = stft(&signal, &StftConfig::default()).unwrap();

        let n_freq = stft_result.shape()[0];
        let n_frames = stft_result.shape()[1];
        let mut mag = Array2::<f32>::zeros((n_freq, n_frames));
        for f in 0..n_freq {
            for t in 0..n_frames {
                mag[(f, t)] = stft_result[(f, t)].norm();
            }
        }

        let freqs = fft_frequencies(22050, 2048);
        let harmonics = vec![1.0]; // Just fundamental
        let harm = interp_harmonics(&mag, &freqs, &harmonics, InterpKind::Linear, 0.0);

        // First harmonic (h=1) should match original
        for f in 0..n_freq {
            for t in 0..n_frames {
                assert!(
                    (harm[(0, f, t)] - mag[(f, t)]).abs() < 1e-4,
                    "h=1 should match original"
                );
            }
        }
    }

    #[test]
    fn test_interp_harmonics_empty() {
        let x = Array2::<f32>::zeros((0, 0));
        let freqs: Vec<f32> = vec![];
        let harmonics = vec![1.0, 2.0];

        let harm = interp_harmonics(&x, &freqs, &harmonics, InterpKind::Linear, 0.0);

        assert_eq!(harm.shape(), &[2, 0, 0]);
    }

    #[test]
    fn test_salience_shape() {
        use crate::convert::fft_frequencies;
        use crate::io;
        use crate::spectrum::{StftConfig, stft};

        let signal = io::tone(440.0, 22050, 0.3);
        let stft_result = stft(&signal, &StftConfig::default()).unwrap();

        let n_freq = stft_result.shape()[0];
        let n_frames = stft_result.shape()[1];
        let mut mag = Array2::<f32>::zeros((n_freq, n_frames));
        for f in 0..n_freq {
            for t in 0..n_frames {
                mag[(f, t)] = stft_result[(f, t)].norm();
            }
        }

        let freqs = fft_frequencies(22050, 2048);
        let harmonics = vec![1.0, 2.0, 3.0, 4.0];
        let sal = salience(
            &mag,
            &freqs,
            &harmonics,
            None,
            false,
            0.0,
            InterpKind::Linear,
        );

        assert_eq!(sal.shape(), &[n_freq, n_frames]);
    }

    #[test]
    fn test_salience_with_weights() {
        use crate::convert::fft_frequencies;
        use crate::io;
        use crate::spectrum::{StftConfig, stft};

        let signal = io::tone(440.0, 22050, 0.3);
        let stft_result = stft(&signal, &StftConfig::default()).unwrap();

        let n_freq = stft_result.shape()[0];
        let n_frames = stft_result.shape()[1];
        let mut mag = Array2::<f32>::zeros((n_freq, n_frames));
        for f in 0..n_freq {
            for t in 0..n_frames {
                mag[(f, t)] = stft_result[(f, t)].norm();
            }
        }

        let freqs = fft_frequencies(22050, 2048);
        let harmonics = vec![1.0, 2.0, 3.0, 4.0];
        let weights = vec![1.0, 0.5, 0.33, 0.25];
        let sal = salience(
            &mag,
            &freqs,
            &harmonics,
            Some(&weights),
            false,
            0.0,
            InterpKind::Linear,
        );

        assert_eq!(sal.shape(), &[n_freq, n_frames]);

        // Should have some non-zero values
        let has_nonzero = sal.iter().any(|&x| x > 0.0);
        assert!(has_nonzero, "Salience should have non-zero values");
    }

    #[test]
    fn test_salience_filter_peaks() {
        use crate::convert::fft_frequencies;
        use crate::io;
        use crate::spectrum::{StftConfig, stft};

        let signal = io::tone(440.0, 22050, 0.5);
        let stft_result = stft(&signal, &StftConfig::default()).unwrap();

        let n_freq = stft_result.shape()[0];
        let n_frames = stft_result.shape()[1];
        let mut mag = Array2::<f32>::zeros((n_freq, n_frames));
        for f in 0..n_freq {
            for t in 0..n_frames {
                mag[(f, t)] = stft_result[(f, t)].norm();
            }
        }

        let freqs = fft_frequencies(22050, 2048);
        let harmonics = vec![1.0, 2.0];
        let sal_peaks = salience(
            &mag,
            &freqs,
            &harmonics,
            None,
            true,
            0.0,
            InterpKind::Linear,
        );
        let sal_all = salience(
            &mag,
            &freqs,
            &harmonics,
            None,
            false,
            0.0,
            InterpKind::Linear,
        );

        // Filtered version should have more zeros
        let zeros_peaks = sal_peaks.iter().filter(|&&x| x == 0.0).count();
        let zeros_all = sal_all.iter().filter(|&&x| x == 0.0).count();

        assert!(
            zeros_peaks >= zeros_all,
            "Peak-filtered salience should have at least as many zeros"
        );
    }

    #[test]
    fn test_salience_empty() {
        let s = Array2::<f32>::zeros((0, 0));
        let freqs: Vec<f32> = vec![];
        let harmonics = vec![1.0, 2.0];

        let sal = salience(&s, &freqs, &harmonics, None, false, 0.0, InterpKind::Linear);

        assert_eq!(sal.shape(), &[0, 0]);
    }

    #[test]
    fn test_f0_harmonics_shape() {
        use crate::convert::fft_frequencies;
        use crate::io;
        use crate::spectrum::{StftConfig, stft};

        let signal = io::tone(440.0, 22050, 0.3);
        let stft_result = stft(&signal, &StftConfig::default()).unwrap();

        let n_freq = stft_result.shape()[0];
        let n_frames = stft_result.shape()[1];
        let mut mag = Array2::<f32>::zeros((n_freq, n_frames));
        for f in 0..n_freq {
            for t in 0..n_frames {
                mag[(f, t)] = stft_result[(f, t)].norm();
            }
        }

        let freqs = fft_frequencies(22050, 2048);
        let f0: Vec<f32> = vec![440.0; n_frames];
        let harmonics: Vec<f32> = (1..=12).map(|i| i as f32).collect();

        let f0_harm = f0_harmonics(&mag, &f0, &freqs, &harmonics, InterpKind::Linear, 0.0).unwrap();

        assert_eq!(f0_harm.shape()[0], 12);
        assert_eq!(f0_harm.shape()[1], n_frames);
    }

    #[test]
    fn test_f0_harmonics_constant_f0() {
        use crate::convert::fft_frequencies;
        use crate::io;
        use crate::spectrum::{StftConfig, stft};

        let signal = io::tone(440.0, 22050, 0.5);
        let stft_result = stft(&signal, &StftConfig::default()).unwrap();

        let n_freq = stft_result.shape()[0];
        let n_frames = stft_result.shape()[1];
        let mut mag = Array2::<f32>::zeros((n_freq, n_frames));
        for f in 0..n_freq {
            for t in 0..n_frames {
                mag[(f, t)] = stft_result[(f, t)].norm();
            }
        }

        let freqs = fft_frequencies(22050, 2048);
        let f0: Vec<f32> = vec![440.0; n_frames];
        let harmonics = vec![1.0, 2.0, 3.0];

        let f0_harm = f0_harmonics(&mag, &f0, &freqs, &harmonics, InterpKind::Linear, 0.0).unwrap();

        // First harmonic (at f0=440) should have high energy
        let mid_frame = n_frames / 2;
        let h1_energy = f0_harm[(0, mid_frame)];

        assert!(
            h1_energy > 0.0,
            "Energy at fundamental should be positive for a 440Hz tone"
        );
    }

    #[test]
    fn test_f0_harmonics_nan_f0() {
        let x = Array2::from_elem((100, 5), 1.0);
        let freqs: Vec<f32> = (0..100).map(|i| i as f32 * 10.0).collect();
        let f0 = vec![100.0, f32::NAN, 100.0, -1.0, 100.0];
        let harmonics = vec![1.0, 2.0];

        let f0_harm =
            f0_harmonics(&x, &f0, &freqs, &harmonics, InterpKind::Linear, -999.0).unwrap();

        // Frames with NaN or invalid f0 should have fill_value
        assert_eq!(f0_harm[(0, 1)], -999.0);
        assert_eq!(f0_harm[(1, 1)], -999.0);
        assert_eq!(f0_harm[(0, 3)], -999.0);
        assert_eq!(f0_harm[(1, 3)], -999.0);
    }

    #[test]
    fn test_f0_harmonics_empty() {
        let x = Array2::<f32>::zeros((0, 0));
        let freqs: Vec<f32> = vec![];
        let f0: Vec<f32> = vec![];
        let harmonics = vec![1.0, 2.0];

        let f0_harm = f0_harmonics(&x, &f0, &freqs, &harmonics, InterpKind::Linear, 0.0).unwrap();

        assert_eq!(f0_harm.shape(), &[2, 0]);
    }

    #[test]
    fn test_f0_harmonics_varying_f0() {
        use crate::convert::fft_frequencies;
        use crate::io;
        use crate::spectrum::{StftConfig, stft};

        // Create a signal with varying frequency (chirp)
        let signal = io::chirp(220.0, 880.0, 22050, 0.5);
        let stft_result = stft(&signal, &StftConfig::default()).unwrap();

        let n_freq = stft_result.shape()[0];
        let n_frames = stft_result.shape()[1];
        let mut mag = Array2::<f32>::zeros((n_freq, n_frames));
        for f in 0..n_freq {
            for t in 0..n_frames {
                mag[(f, t)] = stft_result[(f, t)].norm();
            }
        }

        let freqs = fft_frequencies(22050, 2048);

        // Varying f0 (linear sweep)
        let f0: Vec<f32> = (0..n_frames)
            .map(|i| 220.0 + (880.0 - 220.0) * i as f32 / n_frames as f32)
            .collect();
        let harmonics = vec![1.0, 2.0];

        let f0_harm = f0_harmonics(&mag, &f0, &freqs, &harmonics, InterpKind::Linear, 0.0).unwrap();

        assert_eq!(f0_harm.shape()[0], 2);
        assert_eq!(f0_harm.shape()[1], n_frames);
    }

    #[test]
    fn test_find_local_maxima() {
        let mut x = Array2::<f32>::zeros((10, 2));

        // Frame 0: peak at index 3 and 7
        x[(0, 0)] = 0.0;
        x[(1, 0)] = 1.0;
        x[(2, 0)] = 2.0;
        x[(3, 0)] = 5.0; // peak
        x[(4, 0)] = 3.0;
        x[(5, 0)] = 1.0;
        x[(6, 0)] = 2.0;
        x[(7, 0)] = 4.0; // peak
        x[(8, 0)] = 1.0;
        x[(9, 0)] = 0.0;

        // Frame 1: peak at index 5
        x[(5, 1)] = 10.0;

        let peaks = find_local_maxima(&x);

        assert_eq!(peaks.len(), 2);
        assert!(peaks[0].contains(&3));
        assert!(peaks[0].contains(&7));
        assert!(peaks[1].contains(&5));
    }
}
