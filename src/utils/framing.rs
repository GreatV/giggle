use ndarray::Array2;

/// Pad an array to a specified length, centering the data.
/// If the array is longer than length, it is trimmed from both ends.
pub fn pad_center<T: Clone>(data: &[T], length: usize, fill_value: T) -> Vec<T> {
    if data.len() >= length {
        let start = (data.len() - length) / 2;
        return data[start..start + length].to_vec();
    }

    let pad_total = length - data.len();
    let pad_left = pad_total / 2;

    let mut result = Vec::with_capacity(length);
    result.resize(pad_left, fill_value.clone());
    result.extend_from_slice(data);
    result.resize(length, fill_value);
    result
}

/// Pad or trim an array to exactly the specified length.
/// Padding is added to the right, trimming is done from the right.
pub fn fix_length<T: Clone>(data: &[T], length: usize, fill_value: T) -> Vec<T> {
    if data.len() >= length {
        data[..length].to_vec()
    } else {
        let mut result = data.to_vec();
        result.resize(length, fill_value);
        result
    }
}

/// Expand array size to next valid length for efficient FFT computation.
/// The next valid length is typically the next power of 2, but can also
/// include products of small primes (2, 3, 5) for more efficient FFTs.
pub fn expand_to(length: usize, min_length: Option<usize>) -> usize {
    let target = min_length.unwrap_or(length).max(length);

    // For simplicity, just use next power of 2
    // A more sophisticated version would check for smooth numbers (2^a * 3^b * 5^c)
    if target == 0 {
        return 1;
    }
    target.next_power_of_two()
}

/// Compute the number of frames that fit in a signal of given length.
pub fn frame_count(length: usize, frame_length: usize, hop_length: usize) -> usize {
    if frame_length > length {
        return 0;
    }
    1 + (length - frame_length) / hop_length
}

/// Compute a "tiny" value for the given floating point type.
/// This is used as a small threshold to avoid division by zero.
pub fn tiny<T>(_val: &T) -> f32 {
    1e-8
}

/// Generic framing utility that slices a 1D array into overlapping frames.
/// Returns an Array2 where each column is a frame.
///
/// # Arguments
/// * `data` - Input array to frame
/// * `frame_length` - Length of each frame
/// * `hop_length` - Number of samples to advance between frames
/// * `pad_value` - Value to use for padding when centering
/// * `center` - If true, pad the signal to center frames
pub fn frame_array<T: Clone + Copy>(
    data: &[T],
    frame_length: usize,
    hop_length: usize,
    pad_value: T,
    center: bool,
) -> Array2<T> {
    use ndarray::Array2;

    if frame_length == 0 || hop_length == 0 || data.is_empty() {
        return Array2::from_shape_vec((frame_length, 0), vec![]).unwrap();
    }

    let pad = if center { frame_length / 2 } else { 0 };
    let padded_len = data.len() + 2 * pad;

    if padded_len < frame_length {
        return Array2::from_shape_vec((frame_length, 0), vec![]).unwrap();
    }

    let n_frames = 1 + (padded_len - frame_length) / hop_length;

    // Build in column-major order: iterate over rows first, then columns
    let mut frames = Vec::with_capacity(frame_length * n_frames);

    for i in 0..frame_length {
        for frame_idx in 0..n_frames {
            let start = frame_idx * hop_length;
            let pos = start + i;
            let val = if pos < pad || pos >= pad + data.len() {
                pad_value
            } else {
                data[pos - pad]
            };
            frames.push(val);
        }
    }

    Array2::from_shape_vec((frame_length, n_frames), frames).unwrap()
}

/// Slice a data array into (overlapping) frames.
///
/// This function slices a 1D or 2D array into overlapping frames.
/// For a 1D input of length `n`, the output has shape `(frame_length, n_frames)`
/// where `n_frames = 1 + (n - frame_length) / hop_length`.
///
/// Each column `frames[:, i]` contains a contiguous slice of the input
/// `data[i * hop_length : i * hop_length + frame_length]`.
///
/// # Arguments
/// * `data` - Input array to frame (1D slice)
/// * `frame_length` - Length of each frame (must be > 0)
/// * `hop_length` - Number of steps to advance between frames (must be > 0)
///
/// # Returns
/// Array2 of shape `(frame_length, n_frames)` where each column is a frame.
///
/// # Errors
/// Returns an error if `frame_length` > data length, or if hop_length < 1.
///
/// # Example
/// ```
/// use giggle::utils::frame;
///
/// let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let frames = frame(&data, 3, 2).unwrap();
/// // frames shape is (3, 3)
/// // Column 0: [0, 1, 2]
/// // Column 1: [2, 3, 4]
/// // Column 2: [4, 5, 6]
/// assert_eq!(frames.shape(), &[3, 3]);
/// ```
pub fn frame(data: &[f32], frame_length: usize, hop_length: usize) -> crate::Result<Array2<f32>> {
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
    if data.len() < frame_length {
        return Err(crate::Error::Validation(format!(
            "Input is too short (n={}) for frame_length={}",
            data.len(),
            frame_length
        )));
    }

    // Calculate number of frames
    let n_frames = 1 + (data.len() - frame_length) / hop_length;

    // Build frames - each column is a frame
    // We need to build in column-major order for ndarray
    let mut frames = Vec::with_capacity(frame_length * n_frames);

    // Iterate row-first (frame sample index), then column (frame index)
    for i in 0..frame_length {
        for frame_idx in 0..n_frames {
            let start = frame_idx * hop_length;
            frames.push(data[start + i]);
        }
    }

    // Reshape to (frame_length, n_frames) - Fortran (column-major) order
    Array2::from_shape_vec((frame_length, n_frames), frames)
        .map_err(|e| crate::Error::Validation(format!("Failed to create frame array: {}", e)))
}
