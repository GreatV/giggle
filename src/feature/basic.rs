use crate::frame;
use ndarray::Array2;

/// Compute the zero-crossing rate of an audio signal.
///
/// The zero-crossing rate is the rate at which a signal changes from
/// positive to negative or back. This feature is heavily used in speech
/// recognition and music information retrieval.
///
/// # Arguments
/// * `y` - Input audio signal
///
/// # Returns
/// The fraction of samples that are zero-crossings (0.0 to 1.0)
///
/// # Example
/// ```
/// use giggle::feature::basic::zero_crossing_rate;
///
/// let signal = vec![1.0, -1.0, 1.0, -1.0]; // Alternates every sample
/// let zcr = zero_crossing_rate(&signal);
/// assert_eq!(zcr, 1.0); // 3 crossings / 3 intervals = 1.0
/// ```
pub fn zero_crossing_rate(y: &[f32]) -> f32 {
    if y.len() < 2 {
        return 0.0;
    }
    let mut count = 0usize;
    for i in 1..y.len() {
        let prev = y[i - 1];
        let curr = y[i];
        if (prev >= 0.0 && curr < 0.0) || (prev < 0.0 && curr >= 0.0) {
            count += 1;
        }
    }
    count as f32 / (y.len() - 1) as f32
}

/// Compute the zero-crossing rate for each frame of a signal.
///
/// This is a framed version of `zero_crossing_rate` that computes the ZCR
/// for overlapping windows of the signal.
///
/// # Arguments
/// * `y` - Input audio signal
/// * `frame_length` - Length of each frame
/// * `hop_length` - Number of samples between frames
/// * `center` - If true, pad the signal to center frames
///
/// # Returns
/// Vector of ZCR values, one per frame
///
/// # Example
/// ```
/// use giggle::feature::basic::zero_crossing_rate_frames;
///
/// let signal = vec![1.0f32; 22050];
/// let zcr_frames = zero_crossing_rate_frames(&signal, 2048, 512, true).unwrap();
/// assert!(!zcr_frames.is_empty());
/// ```
pub fn zero_crossing_rate_frames(
    y: &[f32],
    frame_length: usize,
    hop_length: usize,
    center: bool,
) -> crate::Result<Vec<f32>> {
    let frames = frame::frame_signal(y, frame_length, hop_length, center)?;
    Ok(frames
        .iter()
        .map(|frame| zero_crossing_rate(frame))
        .collect())
}

/// Compute the root mean square (RMS) energy of a signal.
///
/// RMS is a measure of the average power or amplitude of a signal.
///
/// # Arguments
/// * `y` - Input audio signal
///
/// # Returns
/// The RMS value (0.0 for silence, higher for louder signals)
///
/// # Example
/// ```
/// use giggle::feature::basic::rms;
///
/// let signal = vec![1.0, 1.0, 1.0, 1.0];
/// let rms_value = rms(&signal);
/// assert_eq!(rms_value, 1.0); // sqrt((1+1+1+1)/4) = 1.0
/// ```
pub fn rms(y: &[f32]) -> f32 {
    if y.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0f32;
    for v in y {
        sum += v * v;
    }
    (sum / y.len() as f32).sqrt()
}

/// Compute the RMS energy for each frame of a signal.
///
/// This is a framed version of `rms` that computes the RMS energy
/// for overlapping windows of the signal.
///
/// # Arguments
/// * `y` - Input audio signal
/// * `frame_length` - Length of each frame
/// * `hop_length` - Number of samples between frames
/// * `center` - If true, pad the signal to center frames
///
/// # Returns
/// Vector of RMS values, one per frame
///
/// # Example
/// ```
/// use giggle::feature::basic::rms_frames;
///
/// let signal = vec![0.5f32; 22050];
/// let rms_vals = rms_frames(&signal, 2048, 512, true).unwrap();
/// assert!(!rms_vals.is_empty());
/// assert!(rms_vals[0] > 0.0); // RMS should be positive
/// ```
pub fn rms_frames(
    y: &[f32],
    frame_length: usize,
    hop_length: usize,
    center: bool,
) -> crate::Result<Vec<f32>> {
    let frames = frame::frame_signal(y, frame_length, hop_length, center)?;
    Ok(frames.iter().map(|frame| rms(frame)).collect())
}

pub fn rms_matrix(y: &Array2<f32>) -> Vec<f32> {
    let channels = y.shape().first().copied().unwrap_or(0);
    let frames = y.shape().get(1).copied().unwrap_or(0);
    if channels == 0 || frames == 0 {
        return Vec::new();
    }

    let mut out = Vec::with_capacity(channels);
    for ch in 0..channels {
        let row = y.row(ch);
        out.push(rms(row.as_slice().unwrap_or(&[])));
    }
    out
}

/// Compute spectral centroid (weighted mean of frequencies).
/// spec: magnitude spectrogram (freq_bins x time_frames)
/// freq_bins: frequency values for each bin
pub fn spectral_centroid(spec: &Array2<f32>, freq_bins: &[f32]) -> crate::Result<Vec<f32>> {
    let (n_freq, n_frames) = (spec.shape()[0], spec.shape()[1]);
    if n_freq != freq_bins.len() {
        return Err(crate::Error::ShapeMismatch {
            expected: format!("freq_bins.len() == {}", n_freq),
            got: format!("{}", freq_bins.len()),
        });
    }
    if n_freq == 0 {
        return Ok(Vec::new());
    }

    let mut centroids = Vec::with_capacity(n_frames);
    for t in 0..n_frames {
        let mut weighted_sum = 0.0f32;
        let mut total = 0.0f32;
        for f in 0..n_freq {
            let mag = spec[(f, t)];
            weighted_sum += freq_bins[f] * mag;
            total += mag;
        }
        centroids.push(if total > 1e-10 {
            weighted_sum / total
        } else {
            0.0
        });
    }
    Ok(centroids)
}

/// Compute spectral bandwidth (weighted standard deviation around centroid).
/// spec: magnitude spectrogram (freq_bins x time_frames)
/// freq_bins: frequency values for each bin
pub fn spectral_bandwidth(spec: &Array2<f32>, freq_bins: &[f32]) -> crate::Result<Vec<f32>> {
    let (n_freq, n_frames) = (spec.shape()[0], spec.shape()[1]);
    if n_freq != freq_bins.len() {
        return Err(crate::Error::ShapeMismatch {
            expected: format!("freq_bins.len() == {}", n_freq),
            got: format!("{}", freq_bins.len()),
        });
    }
    if n_freq == 0 {
        return Ok(Vec::new());
    }

    let centroids = spectral_centroid(spec, freq_bins)?;
    let mut bandwidths = Vec::with_capacity(n_frames);

    for t in 0..n_frames {
        let centroid = centroids[t];
        let mut weighted_variance = 0.0f32;
        let mut total = 0.0f32;
        for f in 0..n_freq {
            let mag = spec[(f, t)];
            let diff = freq_bins[f] - centroid;
            weighted_variance += diff * diff * mag;
            total += mag;
        }
        bandwidths.push(if total > 1e-10 {
            (weighted_variance / total).sqrt()
        } else {
            0.0
        });
    }
    Ok(bandwidths)
}

/// Compute spectral rolloff (frequency below which roll_percent of energy is contained).
/// spec: magnitude spectrogram (freq_bins x time_frames)
/// freq_bins: frequency values for each bin
/// roll_percent: rolloff threshold (e.g. 0.85 for 85%)
pub fn spectral_rolloff(
    spec: &Array2<f32>,
    freq_bins: &[f32],
    roll_percent: f32,
) -> crate::Result<Vec<f32>> {
    let (n_freq, n_frames) = (spec.shape()[0], spec.shape()[1]);
    if n_freq != freq_bins.len() {
        return Err(crate::Error::ShapeMismatch {
            expected: format!("freq_bins.len() == {}", n_freq),
            got: format!("{}", freq_bins.len()),
        });
    }
    if n_freq == 0 {
        return Ok(Vec::new());
    }

    let mut rolloffs = Vec::with_capacity(n_frames);
    for t in 0..n_frames {
        let mut total = 0.0f32;
        for f in 0..n_freq {
            total += spec[(f, t)];
        }

        let threshold = total * roll_percent;
        let mut cumsum = 0.0f32;
        let mut rolloff_freq = freq_bins[n_freq - 1];

        for f in 0..n_freq {
            cumsum += spec[(f, t)];
            if cumsum >= threshold {
                rolloff_freq = freq_bins[f];
                break;
            }
        }
        rolloffs.push(rolloff_freq);
    }
    Ok(rolloffs)
}

/// Compute spectral flatness (ratio of geometric mean to arithmetic mean).
/// spec: magnitude spectrogram (freq_bins x time_frames)
/// power: exponent to apply to spec values (default 2.0 to match librosa)
pub fn spectral_flatness_with_power(spec: &Array2<f32>, power: f32) -> crate::Result<Vec<f32>> {
    let (n_freq, n_frames) = (spec.shape()[0], spec.shape()[1]);
    let mut flatness = Vec::with_capacity(n_frames);
    const AMIN: f64 = 1e-10;

    for t in 0..n_frames {
        let mut log_sum = 0.0f64;
        let mut arith_sum = 0.0f64;

        for f in 0..n_freq {
            let mag = spec[(f, t)] as f64;
            let val = mag.powf(power as f64);
            // Add epsilon to avoid log(0), matching librosa behavior
            log_sum += (val + AMIN).ln();
            arith_sum += val;
        }

        if n_freq > 0 {
            let geom_mean = (log_sum / n_freq as f64).exp();
            let arith_mean = arith_sum / n_freq as f64;
            if arith_mean > AMIN {
                flatness.push((geom_mean / arith_mean) as f32);
            } else {
                flatness.push(0.0);
            }
        } else {
            flatness.push(0.0);
        }
    }
    Ok(flatness)
}

/// Compute spectral flatness (ratio of geometric mean to arithmetic mean).
/// spec: magnitude spectrogram (freq_bins x time_frames)
/// Uses power=2.0 by default to match librosa
pub fn spectral_flatness(spec: &Array2<f32>) -> crate::Result<Vec<f32>> {
    spectral_flatness_with_power(spec, 2.0)
}

/// Compute polynomial features from spectrogram.
/// Fits a polynomial to the spectrum and returns coefficients.
///
/// # Arguments
/// * `spec` - Magnitude spectrogram (freq_bins x time_frames)
/// * `order` - Polynomial order (default 1 for linear fit)
///
/// # Returns
/// Array of polynomial coefficients (order+1 x time_frames)
pub fn poly_features(spec: &Array2<f32>, order: usize) -> crate::Result<Array2<f32>> {
    let (n_freq, n_frames) = (spec.shape()[0], spec.shape()[1]);

    if n_freq == 0 || n_frames == 0 {
        return Ok(Array2::<f32>::zeros((0, 0)));
    }

    let mut coeffs = Array2::<f32>::zeros((order + 1, n_frames));

    // Normalized x values: 0 to 1
    let x: Vec<f32> = (0..n_freq)
        .map(|i| i as f32 / (n_freq - 1).max(1) as f32)
        .collect();

    for t in 0..n_frames {
        let y: Vec<f32> = (0..n_freq).map(|f| spec[(f, t)]).collect();

        // Fit polynomial using least squares
        let poly_coeffs = fit_polynomial(&x, &y, order);

        for (i, &coef) in poly_coeffs.iter().enumerate() {
            coeffs[(i, t)] = coef;
        }
    }

    Ok(coeffs)
}

/// Fit polynomial using least squares.
/// Returns coefficients [c0, c1, c2, ...] where y ≈ c0 + c1*x + c2*x^2 + ...
fn fit_polynomial(x: &[f32], y: &[f32], order: usize) -> Vec<f32> {
    let n = x.len();
    if n == 0 || n != y.len() {
        return vec![0.0; order + 1];
    }

    let m = order + 1;

    // Build Vandermonde matrix: X[i,j] = x[i]^j
    let mut vander = vec![vec![0.0f64; m]; n];
    for i in 0..n {
        let xi = x[i] as f64;
        vander[i][0] = 1.0;
        for j in 1..m {
            vander[i][j] = vander[i][j - 1] * xi;
        }
    }

    // Solve normal equations: (X^T X) c = X^T y
    // Build X^T X
    let mut xtx = vec![vec![0.0f64; m]; m];
    for i in 0..m {
        for j in 0..m {
            let mut sum = 0.0f64;
            for row in vander.iter().take(n) {
                sum += row[i] * row[j];
            }
            xtx[i][j] = sum;
        }
    }

    // Build X^T y
    let mut xty = vec![0.0f64; m];
    for i in 0..m {
        let mut sum = 0.0f64;
        for k in 0..n {
            sum += vander[k][i] * y[k] as f64;
        }
        xty[i] = sum;
    }

    // Solve using Gaussian elimination
    let mut a = xtx.clone();
    let mut b = xty.clone();

    // Forward elimination
    for i in 0..m {
        // Find pivot
        let mut max_row = i;
        for k in i + 1..m {
            if a[k][i].abs() > a[max_row][i].abs() {
                max_row = k;
            }
        }

        // Swap rows
        a.swap(i, max_row);
        b.swap(i, max_row);

        // Skip if pivot is too small
        if a[i][i].abs() < 1e-10 {
            continue;
        }

        // Eliminate column
        for k in i + 1..m {
            let factor = a[k][i] / a[i][i];
            b[k] -= factor * b[i];
            let row_i: Vec<f64> = a[i][i..m].to_vec();
            for (aj, &ai) in a[k][i..m].iter_mut().zip(row_i.iter()) {
                *aj -= factor * ai;
            }
        }
    }

    // Back substitution
    let mut coeffs = vec![0.0f32; m];
    for i in (0..m).rev() {
        if a[i][i].abs() < 1e-10 {
            coeffs[i] = 0.0;
            continue;
        }

        let mut sum = b[i];
        for j in i + 1..m {
            sum -= a[i][j] * coeffs[j] as f64;
        }
        coeffs[i] = (sum / a[i][i]) as f32;
    }

    coeffs
}

/// Compute delta features (temporal derivatives) of a feature matrix.
///
/// Computes the approximate derivative by convolving with a weighted difference kernel.
/// This is commonly used to capture temporal dynamics in audio features (e.g., MFCCs).
///
/// # Arguments
/// * `data` - Input feature matrix with shape (n_features, n_frames)
/// * `width` - Width of the difference kernel (default: 9)
/// * `order` - Order of the derivative (1 = delta, 2 = delta-delta, default: 1)
/// * `axis` - Axis along which to compute differences (default: 1 for time axis)
///
/// # Returns
/// Delta features with same shape as input
///
/// # Example
/// ```
/// use giggle::feature::basic::delta;
/// use ndarray::Array2;
///
/// let features = Array2::from_shape_vec((2, 10),
///     (0..20).map(|x| x as f32).collect()).unwrap();
/// let deltas = delta(&features, 9, 1, 1).unwrap();
/// assert_eq!(deltas.shape(), features.shape());
/// ```
pub fn delta(
    data: &Array2<f32>,
    width: usize,
    order: usize,
    axis: usize,
) -> crate::Result<Array2<f32>> {
    if data.is_empty() || order == 0 {
        return Ok(data.clone());
    }

    if axis > 1 {
        return Err(crate::Error::InvalidParameter {
            name: "axis",
            value: axis.to_string(),
            reason: "must be 0 or 1 for 2D arrays".to_string(),
        });
    }

    let _shape = data.shape();
    let mut result = data.clone();

    for _ in 0..order {
        result = delta_once(&result, width, axis);
    }

    Ok(result)
}

/// Helper function to compute first-order delta once
fn delta_once(data: &Array2<f32>, width: usize, axis: usize) -> Array2<f32> {
    let shape = data.shape();
    let n_rows = shape[0];
    let n_cols = shape[1];

    if width < 3 {
        // Width too small, return zeros
        return Array2::zeros((n_rows, n_cols));
    }

    // Compute regression weights
    let half = width / 2;
    let mut weights = vec![0.0f32; width];
    let mut denom = 0.0f32;

    for (i, w) in weights.iter_mut().enumerate().take(width) {
        let pos = i as isize - half as isize;
        *w = pos as f32;
        denom += (pos * pos) as f32;
    }

    if denom == 0.0 {
        denom = 1.0;
    }

    // Normalize weights
    for w in &mut weights {
        *w /= denom;
    }

    let mut result = Array2::zeros((n_rows, n_cols));

    if axis == 1 {
        // Compute delta along time axis (columns)
        for row in 0..n_rows {
            for col in 0..n_cols {
                let mut delta_val = 0.0f32;

                for (k, &w) in weights.iter().enumerate().take(width) {
                    let offset = k as isize - half as isize;
                    let src_col = col as isize + offset;

                    // Reflect padding at boundaries
                    let src_col = if src_col < 0 {
                        (-src_col) as usize
                    } else if src_col >= n_cols as isize {
                        (2 * n_cols as isize - src_col - 2) as usize
                    } else {
                        src_col as usize
                    };

                    let src_col = src_col.min(n_cols - 1);
                    delta_val += w * data[(row, src_col)];
                }

                result[(row, col)] = delta_val;
            }
        }
    } else {
        // Compute delta along feature axis (rows)
        for col in 0..n_cols {
            for row in 0..n_rows {
                let mut delta_val = 0.0f32;

                for (k, &w) in weights.iter().enumerate().take(width) {
                    let offset = k as isize - half as isize;
                    let src_row = row as isize + offset;

                    // Reflect padding at boundaries
                    let src_row = if src_row < 0 {
                        (-src_row) as usize
                    } else if src_row >= n_rows as isize {
                        (2 * n_rows as isize - src_row - 2) as usize
                    } else {
                        src_row as usize
                    };

                    let src_row = src_row.min(n_rows - 1);
                    delta_val += w * data[(src_row, col)];
                }

                result[(row, col)] = delta_val;
            }
        }
    }

    result
}

/// Short-term history embedding: vertically concatenate a data
/// vector or matrix with delayed copies of itself.
///
/// Each column `data[:, i]` is mapped to:
/// ```text
/// data[..., i] -> [data[..., i],
///                  data[..., i - delay],
///                  ...
///                  data[..., i - (n_steps-1)*delay]]
/// ```
///
/// For columns `i < (n_steps - 1) * delay`, the data will be padded with zeros.
///
/// # Arguments
/// * `data` - Input data matrix of shape `(d, t)` where `d` is the feature dimension
///   and `t` is the number of time steps. If 1D, it will be reshaped to `(1, t)`.
/// * `n_steps` - Number of steps back in time to stack (must be >= 1)
/// * `delay` - Number of columns to step (can be positive or negative)
///   - Positive values embed from the past (previous columns)
///   - Negative values embed from the future (subsequent columns)
///
/// # Returns
/// Array of shape `(n_steps * d, t)` with data augmented with lagged copies.
///
/// # Errors
/// Returns an error if `n_steps < 1` or `delay == 0`.
///
/// # Example
/// ```
/// use giggle::feature::basic::stack_memory;
/// use ndarray::Array2;
///
/// // Keep two steps (current and previous)
/// let data = Array2::from_shape_vec((1, 6), vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0]).unwrap();
/// let result = stack_memory(&data, 2, 1).unwrap();
///
/// // Result has shape (2, 6)
/// // Row 0: [-3, -2, -1, 0, 1, 2] (original)
/// // Row 1: [0, -3, -2, -1, 0, 1] (delayed by 1, zero-padded)
/// assert_eq!(result.shape(), &[2, 6]);
/// ```
pub fn stack_memory(data: &Array2<f32>, n_steps: usize, delay: i32) -> crate::Result<Array2<f32>> {
    if n_steps < 1 {
        return Err(crate::Error::InvalidSize {
            name: "n_steps",
            value: n_steps,
            reason: "must be a positive integer",
        });
    }
    if delay == 0 {
        return Err(crate::Error::InvalidParameter {
            name: "delay",
            value: "0".to_string(),
            reason: "must be a non-zero integer".to_string(),
        });
    }

    let shape = data.shape();
    let d = shape[0];
    let t = shape[1];

    if t == 0 {
        return Err(crate::Error::EmptyAudio);
    }

    // Calculate padding needed
    let pad_amount = ((n_steps - 1) as i32 * delay.abs()) as usize;

    // Create padded data
    let mut padded = Array2::zeros((d, t + pad_amount));

    if delay > 0 {
        // Pad the beginning (for past embedding)
        for row in 0..d {
            for col in 0..t {
                padded[(row, col + pad_amount)] = data[(row, col)];
            }
        }
    } else {
        // Pad the end (for future embedding)
        for row in 0..d {
            for col in 0..t {
                padded[(row, col)] = data[(row, col)];
            }
        }
    }

    // Construct output array
    let mut result = Array2::zeros((d * n_steps, t));

    // Populate the output array
    for step in 0..n_steps {
        let target_row_start = step * d;

        if delay > 0 {
            // For positive delay: nth block is original shifted left by n*delay steps
            let q = n_steps - 1 - step;
            let src_start = q as i32 * delay;

            for row in 0..d {
                for col in 0..t {
                    let src_col = (src_start as usize) + col;
                    result[(target_row_start + row, col)] = padded[(row, src_col)];
                }
            }
        } else {
            // For negative delay: embed from future
            let src_start = step as i32 * (-delay);

            for row in 0..d {
                for col in 0..t {
                    let src_col = (src_start as usize) + col;
                    result[(target_row_start + row, col)] = padded[(row, src_col)];
                }
            }
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_stack_memory_basic() {
        // Test basic stack_memory with 2 steps
        // data = [-3, -2, -1, 0, 1, 2] as row vector
        let data = Array2::from_shape_vec((1, 6), vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0]).unwrap();
        let result = stack_memory(&data, 2, 1).unwrap();

        // Result should have shape (2, 6)
        assert_eq!(result.shape(), &[2, 6]);

        // Row 0: original data [-3, -2, -1, 0, 1, 2]
        assert_eq!(
            result.row(0).to_vec(),
            vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0]
        );

        // Row 1: delayed by 1, zero-padded at start [0, -3, -2, -1, 0, 1]
        assert_eq!(
            result.row(1).to_vec(),
            vec![0.0, -3.0, -2.0, -1.0, 0.0, 1.0]
        );
    }

    #[test]
    fn test_stack_memory_three_steps() {
        // Test with 3 steps
        let data = Array2::from_shape_vec((1, 6), vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0]).unwrap();
        let result = stack_memory(&data, 3, 1).unwrap();

        // Result should have shape (3, 6)
        assert_eq!(result.shape(), &[3, 6]);

        // Row 0: original [-3, -2, -1, 0, 1, 2]
        assert_eq!(
            result.row(0).to_vec(),
            vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0]
        );

        // Row 1: delayed by 1 [0, -3, -2, -1, 0, 1]
        assert_eq!(
            result.row(1).to_vec(),
            vec![0.0, -3.0, -2.0, -1.0, 0.0, 1.0]
        );

        // Row 2: delayed by 2 [0, 0, -3, -2, -1, 0]
        assert_eq!(
            result.row(2).to_vec(),
            vec![0.0, 0.0, -3.0, -2.0, -1.0, 0.0]
        );
    }

    #[test]
    fn test_stack_memory_delay_2() {
        // Test with delay=2
        let data = Array2::from_shape_vec((1, 6), vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0]).unwrap();
        let result = stack_memory(&data, 3, 2).unwrap();

        // Result should have shape (3, 6)
        assert_eq!(result.shape(), &[3, 6]);

        // Row 0: original [-3, -2, -1, 0, 1, 2]
        assert_eq!(
            result.row(0).to_vec(),
            vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0]
        );

        // Row 1: delayed by 2 [0, 0, -3, -2, -1, 0]
        assert_eq!(
            result.row(1).to_vec(),
            vec![0.0, 0.0, -3.0, -2.0, -1.0, 0.0]
        );

        // Row 2: delayed by 4 [0, 0, 0, 0, -3, -2]
        assert_eq!(result.row(2).to_vec(), vec![0.0, 0.0, 0.0, 0.0, -3.0, -2.0]);
    }

    #[test]
    fn test_stack_memory_negative_delay() {
        // Test with negative delay (embed from future)
        let data = Array2::from_shape_vec((1, 6), vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0]).unwrap();
        let result = stack_memory(&data, 2, -1).unwrap();

        // Result should have shape (2, 6)
        assert_eq!(result.shape(), &[2, 6]);

        // Row 0: original [-3, -2, -1, 0, 1, 2]
        assert_eq!(
            result.row(0).to_vec(),
            vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0]
        );

        // Row 1: from future (shifted right by 1) [-2, -1, 0, 1, 2, 0]
        assert_eq!(result.row(1).to_vec(), vec![-2.0, -1.0, 0.0, 1.0, 2.0, 0.0]);
    }

    #[test]
    fn test_stack_memory_2d_data() {
        // Test with 2D data (multiple features)
        let data =
            Array2::from_shape_vec((2, 4), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let result = stack_memory(&data, 2, 1).unwrap();

        // Result should have shape (4, 4) - 2 features * 2 steps
        assert_eq!(result.shape(), &[4, 4]);

        // Rows 0-1: original data at time t
        assert_eq!(result.row(0).to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(result.row(1).to_vec(), vec![5.0, 6.0, 7.0, 8.0]);

        // Rows 2-3: delayed data at time t-1 (zero-padded at start)
        assert_eq!(result.row(2).to_vec(), vec![0.0, 1.0, 2.0, 3.0]);
        assert_eq!(result.row(3).to_vec(), vec![0.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn test_stack_memory_single_step() {
        // Test with n_steps=1 (should return original data)
        let data =
            Array2::from_shape_vec((2, 4), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let result = stack_memory(&data, 1, 1).unwrap();

        // Result should be same as input
        assert_eq!(result.shape(), &[2, 4]);
        assert_eq!(result, data);
    }

    #[test]
    fn test_stack_memory_error_n_steps_zero() {
        let data = Array2::from_shape_vec((1, 4), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let result = stack_memory(&data, 0, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_stack_memory_error_delay_zero() {
        let data = Array2::from_shape_vec((1, 4), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let result = stack_memory(&data, 2, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_stack_memory_empty_data() {
        let data = Array2::from_shape_vec((2, 0), Vec::<f32>::new()).unwrap();
        let result = stack_memory(&data, 2, 1);
        assert!(result.is_err());
    }

    fn create_test_spectrum() -> (Array2<f32>, Vec<f32>) {
        let spec = Array2::from_shape_vec(
            (5, 3),
            vec![
                0.1, 0.2, 0.3, 0.5, 1.0, 0.8, 1.0, 2.0, 1.5, 0.8, 1.5, 1.0, 0.3, 0.5, 0.4,
            ],
        )
        .unwrap();
        let freq_bins = vec![100.0, 200.0, 300.0, 400.0, 500.0];
        (spec, freq_bins)
    }

    #[test]
    fn test_spectral_centroid() {
        let (spec, freq_bins) = create_test_spectrum();
        let centroids = spectral_centroid(&spec, &freq_bins).unwrap();

        assert_eq!(centroids.len(), 3);
        for &c in &centroids {
            assert!((100.0..=500.0).contains(&c));
        }

        let uniform_spec = Array2::from_shape_vec(
            (5, 2),
            vec![0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        )
        .unwrap();
        let uniform_centroids = spectral_centroid(&uniform_spec, &freq_bins).unwrap();
        assert!(uniform_centroids[0] > 250.0);
        assert_relative_eq!(uniform_centroids[1], 300.0, epsilon = 1.0);
    }

    #[test]
    fn test_spectral_bandwidth() {
        let (spec, freq_bins) = create_test_spectrum();
        let bandwidths = spectral_bandwidth(&spec, &freq_bins).unwrap();

        assert_eq!(bandwidths.len(), 3);
        for &bw in &bandwidths {
            assert!((0.0..=500.0).contains(&bw));
        }
    }

    #[test]
    fn test_spectral_rolloff() {
        let (spec, freq_bins) = create_test_spectrum();
        let rolloffs = spectral_rolloff(&spec, &freq_bins, 0.85).unwrap();

        assert_eq!(rolloffs.len(), 3);
        for &r in &rolloffs {
            assert!((100.0..=500.0).contains(&r));
        }
    }

    #[test]
    fn test_spectral_flatness() {
        let (spec, _) = create_test_spectrum();
        let flatness = spectral_flatness(&spec).unwrap();

        assert_eq!(flatness.len(), 3);
        for &f in &flatness {
            assert!((0.0..=1.0).contains(&f));
        }
    }

    #[test]
    fn test_spectral_flatness_uniform() {
        let spec =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).unwrap();

        let flatness = spectral_flatness(&spec).unwrap();
        for &f in &flatness {
            assert_relative_eq!(f, 1.0, epsilon = 0.01);
        }
    }

    #[test]
    fn test_spectral_flatness_impulse() {
        let spec = Array2::from_shape_vec((4, 1), vec![0.1, 1.0, 0.1, 0.1]).unwrap();

        let flatness = spectral_flatness(&spec).unwrap();
        assert!(flatness[0] < 0.9);
        assert!(flatness[0] > 0.0);
    }

    #[test]
    fn test_spectral_features_empty() {
        let spec = Array2::<f32>::zeros((0, 0));
        let freq_bins = vec![];

        assert_eq!(spectral_centroid(&spec, &freq_bins).unwrap().len(), 0);
        assert_eq!(spectral_bandwidth(&spec, &freq_bins).unwrap().len(), 0);
        assert_eq!(spectral_rolloff(&spec, &freq_bins, 0.85).unwrap().len(), 0);
        assert_eq!(spectral_flatness(&spec).unwrap().len(), 0);
    }

    #[test]
    fn test_poly_features() {
        // Linear spectrum
        let spec = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0],
        )
        .unwrap();

        let coeffs = poly_features(&spec, 1).unwrap();
        assert_eq!(coeffs.shape(), &[2, 2]);

        // For linear data, should get good fit
        assert!(coeffs[(0, 0)].abs() < 2.0); // Intercept
        assert!(coeffs[(1, 0)] > 0.0); // Positive slope
    }

    #[test]
    fn test_poly_features_order2() {
        let spec = Array2::from_shape_vec((4, 1), vec![1.0, 4.0, 9.0, 16.0]).unwrap();

        let coeffs = poly_features(&spec, 2).unwrap();
        assert_eq!(coeffs.shape(), &[3, 1]);

        // Should capture quadratic shape
        assert!(coeffs[(2, 0)].abs() > 0.1); // Quadratic term should be significant
    }

    #[test]
    fn test_poly_features_empty() {
        let spec = Array2::<f32>::zeros((0, 0));
        let coeffs = poly_features(&spec, 1).unwrap();
        assert_eq!(coeffs.shape(), &[0, 0]);
    }

    #[test]
    fn test_delta_shape() {
        let data = Array2::from_shape_vec((3, 10), (0..30).map(|x| x as f32).collect()).unwrap();
        let deltas = super::delta(&data, 9, 1, 1).unwrap();
        assert_eq!(deltas.shape(), data.shape());
    }

    #[test]
    fn test_delta_linear_ramp() {
        // For a linear ramp, the first derivative should be constant
        let data = Array2::from_shape_vec((1, 20), (0..20).map(|x| x as f32).collect()).unwrap();
        let deltas = super::delta(&data, 9, 1, 1).unwrap();

        // Check that derivative is relatively constant (around 1.0)
        for col in 5..15 {
            let val = deltas[(0, col)];
            assert!(
                val > 0.5 && val < 1.5,
                "Linear ramp should have constant derivative around 1.0, got {}",
                val
            );
        }
    }

    #[test]
    fn test_delta_constant() {
        // Constant signal should have near-zero derivative
        let data = Array2::from_shape_vec((2, 15), vec![5.0; 30]).unwrap();
        let deltas = super::delta(&data, 9, 1, 1).unwrap();

        for val in deltas.iter() {
            assert!(
                val.abs() < 1e-5,
                "Constant should have zero derivative, got {}",
                val
            );
        }
    }

    #[test]
    fn test_delta_delta() {
        // Second-order derivative (delta-delta)
        // For a linear ramp, first delta is constant, second delta should be near zero
        let data = Array2::from_shape_vec((1, 20), (0..20).map(|x| x as f32).collect()).unwrap();
        let delta2 = super::delta(&data, 9, 2, 1).unwrap();

        assert_eq!(delta2.shape(), data.shape());

        // For linear function, second derivative should be near zero
        for col in 5..15 {
            let val = delta2[(0, col)];
            assert!(
                val.abs() < 0.5,
                "Linear ramp should have near-zero second derivative, got {}",
                val
            );
        }

        // Test with constant signal - both first and second derivatives should be zero
        let constant = Array2::from_shape_vec((1, 20), vec![5.0; 20]).unwrap();
        let delta2_const = super::delta(&constant, 9, 2, 1).unwrap();

        for val in delta2_const.iter() {
            assert!(
                val.abs() < 1e-5,
                "Constant should have zero second derivative, got {}",
                val
            );
        }
    }

    #[test]
    fn test_delta_empty() {
        let data = Array2::<f32>::zeros((0, 0));
        let deltas = super::delta(&data, 9, 1, 1).unwrap();
        assert_eq!(deltas.shape(), &[0, 0]);
    }

    #[test]
    fn test_delta_order_zero() {
        let data = Array2::from_shape_vec((2, 5), (0..10).map(|x| x as f32).collect()).unwrap();
        let deltas = super::delta(&data, 9, 0, 1).unwrap();

        // Order 0 should return original data
        assert_eq!(deltas, data);
    }
}
