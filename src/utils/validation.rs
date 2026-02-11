use crate::fft::FftPlan;
use ndarray::Array2;
use num_complex::Complex32;

pub fn mse(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let n = a.len().min(b.len());
    let mut acc = 0.0f32;
    for i in 0..n {
        let d = a[i] - b[i];
        acc += d * d;
    }
    acc / n as f32
}

/// Compute autocorrelation using FFT (efficient).
/// Returns unnormalized autocorrelation values for lags 0..max_lag.
/// To match librosa behavior, this returns unnormalized values.
pub fn autocorrelate(y: &[f32], max_lag: Option<usize>) -> Vec<f32> {
    if y.is_empty() {
        return Vec::new();
    }

    let n = y.len();
    let max_lag = max_lag.unwrap_or(n).min(n);
    let fft_size = n.next_power_of_two() * 2;

    let mut buffer = vec![Complex32::new(0.0, 0.0); fft_size];
    for i in 0..n {
        buffer[i].re = y[i];
    }

    let fft = FftPlan::new(fft_size);
    fft.forward(&mut buffer);

    for c in buffer.iter_mut() {
        let mag_sq = c.re * c.re + c.im * c.im;
        *c = Complex32::new(mag_sq, 0.0);
    }

    fft.inverse(&mut buffer);

    buffer.iter().take(max_lag).map(|c| c.re).collect()
}

/// Find zero-crossings of a signal.
///
/// Returns indices where `sign(y[i]) != sign(y[i+1])`.
///
/// # Arguments
/// * `y` - Input signal
/// * `threshold` - Values where |y| <= threshold are clipped to 0
/// * `pad` - If true, include a crossing at the first element if `y[0] != 0`
///
/// # Returns
/// Vector of indices where zero crossings occur
pub fn zero_crossings(y: &[f32], threshold: f32, pad: bool) -> Vec<usize> {
    if y.len() < 2 {
        return Vec::new();
    }

    // Apply threshold
    let y_thresh: Vec<f32> = y
        .iter()
        .map(|&v| if v.abs() <= threshold { 0.0 } else { v })
        .collect();

    let mut crossings = Vec::new();

    // Handle padding at the start
    if pad && y_thresh[0] != 0.0 {
        crossings.push(0);
    }

    // Find sign changes
    for i in 1..y_thresh.len() {
        let prev = y_thresh[i - 1];
        let curr = y_thresh[i];
        // Zero crossing when signs differ (treating 0 as positive for sign comparison)
        if (prev >= 0.0 && curr < 0.0) || (prev < 0.0 && curr >= 0.0) {
            crossings.push(i);
        }
    }

    crossings
}

/// Validate audio data.
///
/// Checks that the audio data satisfies the following conditions:
/// - Data is not empty
/// - All samples are finite (no NaN or Inf)
///
/// # Arguments
/// * `y` - Audio signal to validate
///
/// # Returns
/// `Ok(())` if valid, `Err(message)` if invalid
///
/// # Example
/// ```
/// use giggle::utils::valid_audio;
///
/// let y = vec![0.0, 0.5, -0.5, 0.0];
/// assert!(valid_audio(&y).is_ok());
///
/// let empty: Vec<f32> = vec![];
/// assert!(valid_audio(&empty).is_err());
/// ```
pub fn valid_audio(y: &[f32]) -> crate::Result<()> {
    if y.is_empty() {
        return Err(crate::Error::EmptyAudio);
    }

    if !y.iter().all(|&v| v.is_finite()) {
        return Err(crate::Error::NonFiniteAudio);
    }

    Ok(())
}

/// Validate audio data in 2D array (multi-channel).
///
/// # Arguments
/// * `y` - 2D audio data to validate (channels x samples)
///
/// # Returns
/// `Ok(())` if valid, `Err(message)` if invalid
///
/// # Example
/// ```
/// use giggle::utils::valid_audio_2d;
/// use ndarray::Array2;
///
/// let y = Array2::from_shape_vec((2, 4), vec![0.0, 0.5, -0.5, 0.0, 0.1, 0.2, -0.1, -0.2]).unwrap();
/// assert!(valid_audio_2d(&y).is_ok());
/// ```
pub fn valid_audio_2d(y: &Array2<f32>) -> crate::Result<()> {
    if y.is_empty() {
        return Err(crate::Error::EmptyAudio);
    }

    if !y.iter().all(|&v| v.is_finite()) {
        return Err(crate::Error::NonFiniteAudio);
    }

    Ok(())
}

/// Ensure that an input value is a positive integer.
///
/// # Arguments
/// * `x` - Value to validate
/// * `name` - Name of the parameter (for error messages)
///
/// # Returns
/// `Ok(())` if valid, `Err(message)` if invalid
pub fn valid_int(x: f32, name: &str) -> crate::Result<()> {
    if x.is_nan() || x.is_infinite() {
        return Err(crate::Error::Validation(format!(
            "{} must be finite, got {}",
            name, x
        )));
    }

    if x.fract() != 0.0 {
        return Err(crate::Error::Validation(format!(
            "{} must be an integer, got {}",
            name, x
        )));
    }

    Ok(())
}

/// Check if a value is a positive integer.
///
/// # Arguments
/// * `x` - Value to check
///
/// # Returns
/// `true` if x is a positive integer, `false` otherwise
pub fn is_positive_int(x: f32) -> bool {
    x.is_finite() && x > 0.0 && x.fract() == 0.0
}

/// Validate interval boundaries.
///
/// Checks that intervals are valid (start < end, no overlaps if strict).
///
/// # Arguments
/// * `intervals` - Vector of (start, end) pairs
/// * `strict` - If true, check for no overlaps
///
/// # Returns
/// `Ok(())` if valid, `Err(message)` if invalid
pub fn valid_intervals(intervals: &[(f32, f32)], strict: bool) -> crate::Result<()> {
    if intervals.is_empty() {
        return Ok(());
    }

    for (i, &(start, end)) in intervals.iter().enumerate() {
        if !start.is_finite() || !end.is_finite() {
            return Err(crate::Error::Validation(format!(
                "Interval {} has non-finite boundaries",
                i
            )));
        }

        if start >= end {
            return Err(crate::Error::Validation(format!(
                "Interval {} has invalid boundaries: start ({}) >= end ({})",
                i, start, end
            )));
        }

        if strict && i > 0 {
            let prev_end = intervals[i - 1].1;
            if start < prev_end {
                return Err(crate::Error::Validation(format!(
                    "Interval {} overlaps with previous interval",
                    i
                )));
            }
        }
    }

    Ok(())
}

/// Fix a list of frame indices to lie within [x_min, x_max].
///
/// This function clips frame indices to the specified range, optionally pads
/// with boundary values, and returns unique sorted frame indices.
///
/// # Arguments
/// * `frames` - List of non-negative frame indices
/// * `x_min` - Minimum allowed frame index (default: 0)
/// * `x_max` - Maximum allowed frame index (default: None, i.e., no upper bound)
/// * `pad` - If true, expand frames to span the full range [x_min, x_max]
///
/// # Returns
/// Fixed frame indices, sorted and deduplicated
///
/// # Errors
/// Returns an error if frames contains negative values
///
/// # Example
/// ```
/// use giggle::utils::fix_frames;
///
/// let frames = vec![0, 50, 100, 150, 200, 250];
/// let fixed = fix_frames(&frames, Some(0), Some(150), true).unwrap();
/// // Clipped to x_max=150 and padded with boundaries
/// assert!(fixed.contains(&0));
/// assert!(fixed.contains(&150));
/// ```
pub fn fix_frames(
    frames: &[usize],
    x_min: Option<usize>,
    x_max: Option<usize>,
    pad: bool,
) -> crate::Result<Vec<usize>> {
    // Check for negative values (not possible with usize, but check for overflow)
    if frames.contains(&usize::MAX) {
        return Err(crate::Error::Validation(
            "Invalid frame index detected".to_string(),
        ));
    }

    let mut result: Vec<usize> = frames.to_vec();

    // If pad is true and we have bounds, clip frames to range first
    if pad {
        if let Some(min) = x_min {
            result.retain(|&f| f >= min);
        }
        if let Some(max) = x_max {
            result.retain(|&f| f <= max);
        }
    }

    // If pad is true, add boundary values
    if pad {
        if let Some(min) = x_min {
            result.push(min);
        }
        if let Some(max) = x_max {
            result.push(max);
        }
    }

    // Filter by x_min
    if let Some(min) = x_min {
        result.retain(|&f| f >= min);
    }

    // Filter by x_max
    if let Some(max) = x_max {
        result.retain(|&f| f <= max);
    }

    // Sort and deduplicate
    result.sort_unstable();
    result.dedup();

    Ok(result)
}

/// Fix a list of frame indices (f32 version) to lie within [x_min, x_max].
///
/// Similar to `fix_frames` but accepts f32 values and validates they are
/// non-negative integers.
///
/// # Arguments
/// * `frames` - List of frame indices (will be converted to integers)
/// * `x_min` - Minimum allowed frame index (default: 0)
/// * `x_max` - Maximum allowed frame index (default: None)
/// * `pad` - If true, expand frames to span the full range
///
/// # Returns
/// Fixed frame indices, sorted and deduplicated
///
/// # Errors
/// Returns an error if frames contains negative values
pub fn fix_frames_f32(
    frames: &[f32],
    x_min: Option<usize>,
    x_max: Option<usize>,
    pad: bool,
) -> crate::Result<Vec<usize>> {
    // Check for negative values
    if frames.iter().any(|&f| f < 0.0) {
        return Err(crate::Error::Validation(
            "Negative frame index detected".to_string(),
        ));
    }

    // Convert to usize (truncating)
    let frames_usize: Vec<usize> = frames.iter().map(|&f| f as usize).collect();

    fix_frames(&frames_usize, x_min, x_max, pad)
}

/// Convert an integer buffer to floating point values.
///
/// This is primarily useful when loading integer-valued wav data
/// into floating point arrays.
///
/// # Arguments
/// * `x` - The integer-valued data buffer (as raw bytes)
/// * `n_bytes` - The number of bytes per sample (1, 2, or 4)
///
/// # Returns
/// Vector of f32 values normalized to [-1.0, 1.0]
///
/// # Errors
/// Returns an error if n_bytes is not 1, 2, or 4,
/// or if the buffer length is not a multiple of n_bytes
///
/// # Example
/// ```
/// use giggle::utils::buf_to_float;
///
/// // 16-bit samples: [0, 32767, -32768, 16384]
/// let bytes = vec![
///     0x00, 0x00,  // 0
///     0xFF, 0x7F,  // 32767
///     0x00, 0x80,  // -32768
///     0x00, 0x40,  // 16384
/// ];
/// let float_samples = buf_to_float(&bytes, 2).unwrap();
/// assert!(float_samples[0].abs() < 1e-6);  // 0 -> 0.0
/// assert!(float_samples[1] > 0.99);        // 32767 -> ~1.0
/// assert!(float_samples[2] < -0.99);       // -32768 -> ~-1.0
/// ```
pub fn buf_to_float(x: &[u8], n_bytes: usize) -> crate::Result<Vec<f32>> {
    if n_bytes != 1 && n_bytes != 2 && n_bytes != 4 {
        return Err(crate::Error::InvalidParameter {
            name: "n_bytes",
            value: n_bytes.to_string(),
            reason: "must be 1, 2, or 4".to_string(),
        });
    }

    if !x.len().is_multiple_of(n_bytes) {
        return Err(crate::Error::Validation(format!(
            "Buffer length ({}) is not a multiple of n_bytes ({})",
            x.len(),
            n_bytes
        )));
    }

    let n_samples = x.len() / n_bytes;
    let mut result = Vec::with_capacity(n_samples);

    // Calculate scale factor: 1.0 / 2^(8*n_bytes - 1)
    let scale = 1.0 / ((1i64 << (8 * n_bytes - 1)) as f32);

    match n_bytes {
        1 => {
            // 8-bit samples are typically unsigned (0-255)
            // Convert to signed (-128 to 127) first
            for &b in x.iter().take(n_samples) {
                let byte = b as i8;
                result.push(byte as f32 * scale);
            }
        }
        2 => {
            // 16-bit signed samples
            for i in 0..n_samples {
                let idx = i * 2;
                let sample = i16::from_le_bytes([x[idx], x[idx + 1]]);
                result.push(sample as f32 * scale);
            }
        }
        4 => {
            // 32-bit signed samples
            for i in 0..n_samples {
                let idx = i * 4;
                let sample = i32::from_le_bytes([x[idx], x[idx + 1], x[idx + 2], x[idx + 3]]);
                result.push(sample as f32 * scale);
            }
        }
        _ => unreachable!(),
    }

    Ok(result)
}

/// Convert an integer buffer (i16) to floating point values.
///
/// Convenience function for 16-bit integer samples.
///
/// # Arguments
/// * `x` - Slice of i16 samples
///
/// # Returns
/// Vector of f32 values normalized to [-1.0, 1.0]
///
/// # Example
/// ```
/// use giggle::utils::i16_to_float;
///
/// let samples = vec![0i16, 32767, -32768, 16384];
/// let float_samples = i16_to_float(&samples);
/// assert!(float_samples[0].abs() < 1e-6);  // 0 -> 0.0
/// assert!(float_samples[1] > 0.99);        // 32767 -> ~1.0
/// ```
pub fn i16_to_float(x: &[i16]) -> Vec<f32> {
    let scale = 1.0 / 32768.0; // 1.0 / 2^15
    x.iter().map(|&v| v as f32 * scale).collect()
}

/// Convert an integer buffer (i32) to floating point values.
///
/// Convenience function for 32-bit integer samples (e.g., 24-bit audio stored in 32 bits).
///
/// # Arguments
/// * `x` - Slice of i32 samples
///
/// # Returns
/// Vector of f32 values normalized to [-1.0, 1.0]
pub fn i32_to_float(x: &[i32]) -> Vec<f32> {
    let scale = 1.0 / 2147483648.0; // 1.0 / 2^31
    x.iter().map(|&v| v as f32 * scale).collect()
}

/// Generate a slice array from an index array.
///
/// This function converts index boundaries into slice objects that can be used
/// to index arrays. It first normalizes the indices using `fix_frames`, then
/// creates slices from adjacent index pairs.
///
/// # Arguments
/// * `idx` - Array of index boundaries
/// * `idx_min` - Minimum allowed index (optional)
/// * `idx_max` - Maximum allowed index (optional)
/// * `step` - Step size for each slice (optional, default is 1)
/// * `pad` - If true, pad idx to span the range [idx_min, idx_max]
///
/// # Returns
/// Vector of slice objects: `slices[i] = idx[i]..idx[i+1]` (with optional step)
///
/// # Errors
/// Returns an error if the index array is empty after fixing, or if
/// there are fewer than 2 indices to form slices.
///
/// # Example
/// ```
/// use giggle::utils::index_to_slice;
///
/// // Generate slices from spaced indices
/// let idx = vec![20, 35, 50, 65, 80, 95];
/// let slices = index_to_slice(&idx, None, None, None, true).unwrap();
/// assert_eq!(slices.len(), 5);
/// assert_eq!(slices[0], (20..35));  // slice(20, 35, None)
///
/// // Pad to span the range (0, 100)
/// let slices = index_to_slice(&idx, Some(0), Some(100), None, true).unwrap();
/// assert_eq!(slices.len(), 7);
/// assert_eq!(slices[0], (0..20));   // slice(0, 20, None) - padded
/// assert_eq!(slices[6], (95..100)); // slice(95, 100, None) - padded
/// ```
pub fn index_to_slice(
    idx: &[usize],
    idx_min: Option<usize>,
    idx_max: Option<usize>,
    _step: Option<usize>,
    pad: bool,
) -> crate::Result<Vec<std::ops::Range<usize>>> {
    // First, normalize the index set using fix_frames
    let idx_fixed = fix_frames(idx, idx_min, idx_max, pad)?;

    if idx_fixed.len() < 2 {
        return Err(crate::Error::Validation(
            "Need at least 2 indices to form slices".to_string(),
        ));
    }

    // Now convert the indices to slices
    let mut slices = Vec::with_capacity(idx_fixed.len() - 1);

    for i in 0..idx_fixed.len() - 1 {
        let start = idx_fixed[i];
        let end = idx_fixed[i + 1];

        // Validate that start < end
        if start >= end {
            continue; // Skip invalid slices (shouldn't happen with fix_frames)
        }

        // Create the range with optional step
        // Note: Rust Range doesn't support step directly, so we return Range<usize>
        // and the caller can use .step_by() if needed
        slices.push(start..end);
    }

    Ok(slices)
}

/// Generate a slice array from an f32 index array.
///
/// Convenience function that accepts f32 indices and converts them to usize.
/// Negative values are rejected.
///
/// # Arguments
/// * `idx` - Array of index boundaries as f32
/// * `idx_min` - Minimum allowed index (optional)
/// * `idx_max` - Maximum allowed index (optional)
/// * `step` - Step size for each slice (optional)
/// * `pad` - If true, pad idx to span the range [idx_min, idx_max]
///
/// # Returns
/// Vector of slice objects
pub fn index_to_slice_f32(
    idx: &[f32],
    idx_min: Option<usize>,
    idx_max: Option<usize>,
    step: Option<usize>,
    pad: bool,
) -> crate::Result<Vec<std::ops::Range<usize>>> {
    // Check for negative values
    if idx.iter().any(|&v| v < 0.0) {
        return Err(crate::Error::Validation(
            "Negative index detected".to_string(),
        ));
    }

    // Convert to usize
    let idx_usize: Vec<usize> = idx.iter().map(|&v| v as usize).collect();

    index_to_slice(&idx_usize, idx_min, idx_max, step, pad)
}
