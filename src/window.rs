/// Compute a periodic Hann (raised cosine) window.
///
/// The Hann window is one of the most commonly used windows in spectral
/// analysis. It has good frequency resolution and moderate spectral leakage.
///
/// # Arguments
/// * `n` - Window length
///
/// # Returns
/// Hann window of length `n`
pub fn hann(n: usize) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let m = n as f32;
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / m).cos())
        .collect()
}

/// Compute a Hamming window.
///
/// The Hamming window is similar to the Hann window but with slightly
/// different coefficients that reduce the first side lobe level.
///
/// # Arguments
/// * `n` - Window length
///
/// # Returns
/// Hamming window of length `n`
pub fn hamming(n: usize) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let m = n as f32;
    (0..n)
        .map(|i| 0.54 - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / m).cos())
        .collect()
}

/// Compute a Blackman window.
///
/// The Blackman window provides better side lobe suppression than Hann
/// or Hamming windows, at the cost of wider main lobe.
///
/// # Arguments
/// * `n` - Window length
///
/// # Returns
/// Blackman window of length `n`
pub fn blackman(n: usize) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let m = n as f32;
    (0..n)
        .map(|i| {
            let a = 2.0 * std::f32::consts::PI * i as f32 / m;
            0.42 - 0.5 * a.cos() + 0.08 * (2.0 * a).cos()
        })
        .collect()
}

/// Compute a Bartlett (triangular) window.
///
/// The Bartlett window is a triangular window that tapers linearly
/// from zero at the edges to a peak in the center.
///
/// # Arguments
/// * `n` - Window length
///
/// # Returns
/// Bartlett window of length `n`
pub fn bartlett(n: usize) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let m = n as f32;
    (0..n)
        .map(|i| 1.0 - ((i as f32 - m / 2.0).abs() / (m / 2.0)))
        .collect()
}

/// Window type specification for get_window function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowType {
    Hann,
    Hamming,
    Blackman,
    Bartlett,
}

impl WindowType {
    /// Parse a window type from a string.
    ///
    /// # Arguments
    /// * `name` - Window name (case-insensitive)
    ///
    /// # Returns
    /// Some(WindowType) if recognized, None otherwise
    pub fn parse(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "hann" | "hanning" => Some(WindowType::Hann),
            "hamming" => Some(WindowType::Hamming),
            "blackman" => Some(WindowType::Blackman),
            "bartlett" | "triangle" => Some(WindowType::Bartlett),
            _ => None,
        }
    }
}

/// Get a window of the specified type and length.
///
/// This function provides a unified interface for creating windows,
/// similar to librosa.filters.get_window().
///
/// # Arguments
/// * `window` - Window specification (name or type)
/// * `n` - Window length
///
/// # Returns
/// Window vector of length n
///
/// # Example
/// ```
/// use giggle::window::{get_window, get_window_from_str, WindowType};
///
/// // Using WindowType
/// let w1 = get_window(WindowType::Hann, 512);
///
/// // Using string name
/// let w2 = get_window_from_str("hamming", 512).unwrap();
/// ```
pub fn get_window(window: WindowType, n: usize) -> Vec<f32> {
    match window {
        WindowType::Hann => hann(n),
        WindowType::Hamming => hamming(n),
        WindowType::Blackman => blackman(n),
        WindowType::Bartlett => bartlett(n),
    }
}

/// Get a window from a string specification.
///
/// # Arguments
/// * `name` - Window name (case-insensitive): "hann", "hamming", "blackman", "bartlett"
/// * `n` - Window length
///
/// # Returns
/// Some(window) if name is recognized, None otherwise
pub fn get_window_from_str(name: &str, n: usize) -> Option<Vec<f32>> {
    WindowType::parse(name).map(|wtype| get_window(wtype, n))
}

/// Build a two-dimensional diagonal filter kernel.
///
/// This is primarily used for smoothing recurrence or self-similarity matrices.
/// The filter enhances diagonal structures in the input.
///
/// # Arguments
/// * `window` - Window type for the filter
/// * `n` - Length of the filter (produces an n x n kernel)
/// * `slope` - Slope of the diagonal (1.0 = 45 degrees)
/// * `zero_mean` - If true, produce a zero-mean filter (enhances paths, suppresses blocks)
///
/// # Returns
/// 2D filter kernel as a flattened vector (row-major, n x n)
///
/// # Example
/// ```
/// use giggle::window::{diagonal_filter, WindowType};
///
/// // Create a 7x7 diagonal filter with Hann window
/// let kernel = diagonal_filter(WindowType::Hann, 7, 1.0, false);
/// assert_eq!(kernel.len(), 49); // 7 * 7
///
/// // Sum should be approximately 1.0 for averaging filter
/// let sum: f32 = kernel.iter().sum();
/// assert!((sum - 1.0).abs() < 0.01);
/// ```
pub fn diagonal_filter(window: WindowType, n: usize, slope: f32, zero_mean: bool) -> Vec<f32> {
    use ndarray::Array2;

    if n == 0 {
        return Vec::new();
    }

    // Get the window function
    let win_1d = get_window(window, n);

    // Create a diagonal matrix with the window values
    let mut kernel = Array2::<f32>::zeros((n, n));
    for i in 0..n {
        kernel[(i, i)] = win_1d[i];
    }

    // If slope != 1.0, we need to rotate the kernel
    // For simplicity, we implement rotation for common cases
    let angle = slope.atan(); // angle in radians
    let target_angle = std::f32::consts::FRAC_PI_4; // 45 degrees

    if (angle - target_angle).abs() > 0.01 {
        // Rotate the kernel
        // Using simple nearest-neighbor rotation for robustness
        let rotation_angle = target_angle - angle; // How much to rotate from 45 deg
        kernel = rotate_kernel(&kernel, rotation_angle);
    }

    // Clip to non-negative values
    for val in kernel.iter_mut() {
        *val = val.max(0.0);
    }

    // Normalize to sum to 1
    let sum: f32 = kernel.iter().sum();
    if sum > 0.0 {
        for val in kernel.iter_mut() {
            *val /= sum;
        }
    }

    // Apply zero-mean if requested
    if zero_mean {
        let mean: f32 = kernel.iter().sum::<f32>() / (n * n) as f32;
        for val in kernel.iter_mut() {
            *val -= mean;
        }
    }

    // Return as flattened row-major vector
    kernel.into_raw_vec()
}

/// Build a diagonal filter from a string window specification.
///
/// # Arguments
/// * `window_name` - Window name ("hann", "hamming", etc.)
/// * `n` - Filter size
/// * `slope` - Diagonal slope
/// * `zero_mean` - If true, produce zero-mean filter
///
/// # Returns
/// 2D filter kernel as flattened vector, or empty if window name is invalid
pub fn diagonal_filter_from_str(
    window_name: &str,
    n: usize,
    slope: f32,
    zero_mean: bool,
) -> Vec<f32> {
    match WindowType::parse(window_name) {
        Some(wtype) => diagonal_filter(wtype, n, slope, zero_mean),
        None => Vec::new(),
    }
}

/// Rotate a 2D kernel by a given angle (in radians).
/// Uses bilinear interpolation for smooth rotation.
fn rotate_kernel(kernel: &ndarray::Array2<f32>, angle: f32) -> ndarray::Array2<f32> {
    use ndarray::Array2;

    let (rows, cols) = kernel.dim();
    let mut rotated = Array2::<f32>::zeros((rows, cols));

    let cos_a = angle.cos();
    let sin_a = angle.sin();

    let center_y = (rows as f32 - 1.0) / 2.0;
    let center_x = (cols as f32 - 1.0) / 2.0;

    for y in 0..rows {
        for x in 0..cols {
            // Translate to center, rotate, translate back
            let dy = y as f32 - center_y;
            let dx = x as f32 - center_x;

            let src_x = dx * cos_a + dy * sin_a + center_x;
            let src_y = -dx * sin_a + dy * cos_a + center_y;

            // Bilinear interpolation
            let x0 = src_x.floor() as i32;
            let y0 = src_y.floor() as i32;
            let x1 = x0 + 1;
            let y1 = y0 + 1;

            let fx = src_x - x0 as f32;
            let fy = src_y - y0 as f32;

            let get_val = |py: i32, px: i32| -> f32 {
                if py >= 0 && py < rows as i32 && px >= 0 && px < cols as i32 {
                    kernel[(py as usize, px as usize)]
                } else {
                    0.0
                }
            };

            let v00 = get_val(y0, x0);
            let v01 = get_val(y0, x1);
            let v10 = get_val(y1, x0);
            let v11 = get_val(y1, x1);

            rotated[(y, x)] = (1.0 - fx) * (1.0 - fy) * v00
                + fx * (1.0 - fy) * v01
                + (1.0 - fx) * fy * v10
                + fx * fy * v11;
        }
    }

    rotated
}

/// Compute the sum of squared window values for overlap-add processing.
/// This is used to check if the NOLA (nonzero overlap-add) constraint is satisfied,
/// which is necessary for perfect STFT/ISTFT reconstruction.
///
/// # Arguments
/// * `window` - The window function values
/// * `n_frames` - Number of frames
/// * `hop_length` - Number of samples between frames
/// * `win_length` - Length of the window (if None, uses window.len())
/// * `n_fft` - FFT size (if None, uses window.len())
///
/// # Returns
/// An array of length (n_frames - 1) * hop_length + win_length showing the
/// sum of squared window values at each position.
pub fn window_sumsquare(
    window: &[f32],
    n_frames: usize,
    hop_length: usize,
    win_length: Option<usize>,
    n_fft: Option<usize>,
) -> Vec<f32> {
    if window.is_empty() || n_frames == 0 || hop_length == 0 {
        return Vec::new();
    }

    let win_len = win_length.unwrap_or(window.len());
    let fft_size = n_fft.unwrap_or(window.len());

    // The output length is the span covered by all frames
    let out_len = if n_frames > 0 {
        (n_frames - 1) * hop_length + fft_size
    } else {
        0
    };

    let mut wss = vec![0.0f32; out_len];

    // For each frame, add the squared window values
    for frame_idx in 0..n_frames {
        let offset = frame_idx * hop_length;

        for (i, &w) in window.iter().enumerate().take(win_len.min(window.len())) {
            let pos = offset + i;
            if pos < out_len {
                wss[pos] += w * w;
            }
        }
    }

    wss
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_window_sumsquare_basic() {
        let window = hann(512);
        let n_frames = 10;
        let hop_length = 256;

        let wss = window_sumsquare(&window, n_frames, hop_length, None, Some(512));

        // Check output length
        let expected_len = (n_frames - 1) * hop_length + 512;
        assert_eq!(wss.len(), expected_len);

        // For 50% overlap with Hann window, NOLA should be satisfied
        // In the middle region, wss should be approximately constant and > 0
        let middle_start = 512;
        let middle_end = expected_len - 512;

        for (i, &val) in wss.iter().enumerate().take(middle_end).skip(middle_start) {
            assert!(val > 0.0, "NOLA violated at position {}", i);
        }
    }

    #[test]
    fn test_window_sumsquare_nola() {
        // For 50% overlap with Hann window, check NOLA in the stable region
        let window = hann(512);
        let n_frames = 10;
        let hop_length = 256;

        let wss = window_sumsquare(&window, n_frames, hop_length, None, Some(512));

        // Check that values in the stable middle region are all positive (NOLA satisfied)
        // The stable region starts after first complete overlap and ends before last
        let stable_start = 512; // After first frame is fully overlapped
        let stable_end = wss.len().saturating_sub(512);

        if stable_end > stable_start {
            // All values should be positive (NOLA satisfied)
            for (i, &val) in wss.iter().enumerate().take(stable_end).skip(stable_start) {
                assert!(val > 0.0, "NOLA violated at position {}: value {}", i, val);
            }

            // Check that we have reasonable overlap energy (> 0.3 for Hann + 50% overlap)
            let min_val = wss[stable_start..stable_end]
                .iter()
                .cloned()
                .fold(f32::INFINITY, f32::min);
            assert!(min_val > 0.3, "Minimum NOLA value {} is too low", min_val);
        }
    }

    #[test]
    fn test_window_sumsquare_empty() {
        let window = vec![];
        let wss = window_sumsquare(&window, 10, 256, None, None);
        assert_eq!(wss.len(), 0);
    }

    #[test]
    fn test_window_sumsquare_single_frame() {
        let window = hann(512);
        let wss = window_sumsquare(&window, 1, 256, None, Some(512));

        assert_eq!(wss.len(), 512);
        // With single frame, wss should just be window squared
        for i in 0..512 {
            assert_relative_eq!(wss[i], window[i] * window[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_hann_window() {
        let w = hann(512);
        assert_eq!(w.len(), 512);

        // Check all values are in valid range
        assert!(w.iter().all(|&v| (0.0..=1.0).contains(&v)));

        // First and last should be near zero (periodic Hann window)
        assert!(w[0] < 0.1);

        // Middle should be close to 1.0
        assert!(w[256] > 0.9);
    }

    #[test]
    fn test_window_type_from_str() {
        assert_eq!(WindowType::parse("hann"), Some(WindowType::Hann));
        assert_eq!(WindowType::parse("Hann"), Some(WindowType::Hann));
        assert_eq!(WindowType::parse("hanning"), Some(WindowType::Hann));
        assert_eq!(WindowType::parse("hamming"), Some(WindowType::Hamming));
        assert_eq!(WindowType::parse("blackman"), Some(WindowType::Blackman));
        assert_eq!(WindowType::parse("bartlett"), Some(WindowType::Bartlett));
        assert_eq!(WindowType::parse("triangle"), Some(WindowType::Bartlett));
        assert_eq!(WindowType::parse("unknown"), None);
    }

    #[test]
    fn test_get_window() {
        let n = 256;

        // Test each window type
        let hann_w = get_window(WindowType::Hann, n);
        assert_eq!(hann_w.len(), n);
        assert_eq!(hann_w, hann(n));

        let hamming_w = get_window(WindowType::Hamming, n);
        assert_eq!(hamming_w.len(), n);
        assert_eq!(hamming_w, hamming(n));

        let blackman_w = get_window(WindowType::Blackman, n);
        assert_eq!(blackman_w.len(), n);
        assert_eq!(blackman_w, blackman(n));

        let bartlett_w = get_window(WindowType::Bartlett, n);
        assert_eq!(bartlett_w.len(), n);
        assert_eq!(bartlett_w, bartlett(n));
    }

    #[test]
    fn test_get_window_from_str() {
        let n = 128;

        // Test valid names
        assert!(get_window_from_str("hann", n).is_some());
        assert!(get_window_from_str("hamming", n).is_some());
        assert!(get_window_from_str("blackman", n).is_some());
        assert!(get_window_from_str("bartlett", n).is_some());

        // Test case insensitivity
        assert_eq!(get_window_from_str("HANN", n), Some(hann(n)));

        // Test invalid name
        assert!(get_window_from_str("invalid", n).is_none());

        // Test consistency
        let w1 = get_window_from_str("hann", n).unwrap();
        let w2 = get_window(WindowType::Hann, n);
        assert_eq!(w1, w2);
    }
}
