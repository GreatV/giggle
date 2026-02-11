//! Visualization functions for audio analysis.
//!
//! This module provides plotting utilities similar to librosa.display.
//! Enable with the `display` feature in Cargo.toml:
//!
//! ```toml
//! [dependencies]
//! giggle = { version = "0.1", features = ["display"] }
//! ```

use ndarray::Array2;
use std::str::FromStr;

/// Color map types for spectrograms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorMap {
    /// Viridis colormap (perceptually uniform, colorblind-friendly)
    Viridis,
    /// Magma colormap (perceptually uniform, dark background)
    Magma,
    /// Inferno colormap (perceptually uniform, high contrast)
    Inferno,
    /// Plasma colormap (perceptually uniform)
    Plasma,
    /// Grayscale colormap
    Grayscale,
    /// Coolwarm diverging colormap
    Coolwarm,
}

impl ColorMap {
    /// Convert a normalized value (0.0 to 1.0) to RGB color.
    pub fn to_rgb(&self, value: f32) -> (u8, u8, u8) {
        let v = value.clamp(0.0, 1.0);
        match self {
            ColorMap::Viridis => viridis(v),
            ColorMap::Magma => magma(v),
            ColorMap::Inferno => inferno(v),
            ColorMap::Plasma => plasma(v),
            ColorMap::Grayscale => {
                let g = (v * 255.0) as u8;
                (g, g, g)
            }
            ColorMap::Coolwarm => coolwarm(v),
        }
    }
}

impl FromStr for ColorMap {
    type Err = ();

    /// Parse colormap from string name.
    fn from_str(name: &str) -> Result<Self, Self::Err> {
        match name.to_lowercase().as_str() {
            "viridis" => Ok(ColorMap::Viridis),
            "magma" => Ok(ColorMap::Magma),
            "inferno" => Ok(ColorMap::Inferno),
            "plasma" => Ok(ColorMap::Plasma),
            "grayscale" | "gray" | "grey" => Ok(ColorMap::Grayscale),
            "coolwarm" => Ok(ColorMap::Coolwarm),
            _ => Err(()),
        }
    }
}

/// Viridis colormap implementation.
fn viridis(t: f32) -> (u8, u8, u8) {
    // Simplified viridis approximation
    let r = (0.267004 + t * (0.003991 + t * (1.096452 + t * (-2.146305 + t * 1.167419))))
        .clamp(0.0, 1.0);
    let g = (0.004874 + t * (1.015861 + t * (-0.107203 + t * (-0.449175 + t * 0.539506))))
        .clamp(0.0, 1.0);
    let b = (0.329415 + t * (1.421511 + t * (-2.482568 + t * (1.871714 + t * (-0.140092)))))
        .clamp(0.0, 1.0);
    ((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

/// Magma colormap implementation.
fn magma(t: f32) -> (u8, u8, u8) {
    let r = (0.001462 + t * (0.169823 + t * (2.240361 + t * (-1.106994)))).clamp(0.0, 1.0);
    let g = (0.000466 + t * (0.100897 + t * (0.699060 + t * (0.203185)))).clamp(0.0, 1.0);
    let b = (0.013866 + t * (0.563622 + t * (-0.543021 + t * (0.966020)))).clamp(0.0, 1.0);
    ((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

/// Inferno colormap implementation.
fn inferno(t: f32) -> (u8, u8, u8) {
    let r = (0.001462 + t * (0.277106 + t * (2.421006 + t * (-1.699106)))).clamp(0.0, 1.0);
    let g = (0.000466 + t * (0.031596 + t * (0.867279 + t * (0.101106)))).clamp(0.0, 1.0);
    let b = (0.013866 + t * (0.711039 + t * (-1.213691 + t * (0.489140)))).clamp(0.0, 1.0);
    ((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

/// Plasma colormap implementation.
fn plasma(t: f32) -> (u8, u8, u8) {
    let r = (0.050383 + t * (2.028154 + t * (-1.078553))).clamp(0.0, 1.0);
    let g = (0.029803 + t * (0.260172 + t * (0.709818))).clamp(0.0, 1.0);
    let b = (0.527975 + t * (0.548977 + t * (-1.076952 + t * (1.0)))).clamp(0.0, 1.0);
    ((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

/// Coolwarm diverging colormap implementation.
fn coolwarm(t: f32) -> (u8, u8, u8) {
    // Blue (cold) to red (warm)
    let r = if t < 0.5 {
        (0.230 + t * 1.540).clamp(0.0, 1.0)
    } else {
        1.0
    };
    let g = if t < 0.5 {
        (0.299 + t * 0.802).clamp(0.0, 1.0)
    } else {
        (1.0 - (t - 0.5) * 1.6).clamp(0.0, 1.0)
    };
    let b = if t < 0.5 {
        1.0
    } else {
        (1.0 - (t - 0.5) * 1.8).clamp(0.0, 1.0)
    };
    ((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

/// Convert a 2D spectrogram to RGB image data.
///
/// # Arguments
/// * `data` - 2D array (frequency x time) in dB or linear scale
/// * `cmap` - Colormap to use
/// * `vmin` - Minimum value for scaling (if None, uses data min)
/// * `vmax` - Maximum value for scaling (if None, uses data max)
///
/// # Returns
/// RGB image data as (width, height, pixels) where pixels is Vec<u8> in RGB format
///
/// # Example
/// ```
/// use giggle::display::{spectrogram_to_rgb, ColorMap};
/// use ndarray::Array2;
///
/// let spec = Array2::<f32>::zeros((128, 100));
/// let (width, height, pixels) = spectrogram_to_rgb(&spec, ColorMap::Viridis, None, None);
/// assert_eq!(width, 100);
/// assert_eq!(height, 128);
/// assert_eq!(pixels.len(), 100 * 128 * 3);
/// ```
pub fn spectrogram_to_rgb(
    data: &Array2<f32>,
    cmap: ColorMap,
    vmin: Option<f32>,
    vmax: Option<f32>,
) -> (usize, usize, Vec<u8>) {
    let (n_freq, n_time) = data.dim();

    if n_freq == 0 || n_time == 0 {
        return (0, 0, Vec::new());
    }

    let data_min = vmin.unwrap_or_else(|| data.iter().cloned().fold(f32::INFINITY, f32::min));
    let data_max = vmax.unwrap_or_else(|| data.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    let range = (data_max - data_min).max(1e-10);

    // Create RGB buffer (note: frequency axis is flipped for display, low freq at bottom)
    let mut pixels = Vec::with_capacity(n_freq * n_time * 3);

    // Iterate from top (high freq) to bottom (low freq) for standard spectrogram display
    for f in (0..n_freq).rev() {
        for t in 0..n_time {
            let val = data[(f, t)];
            let normalized = ((val - data_min) / range).clamp(0.0, 1.0);
            let (r, g, b) = cmap.to_rgb(normalized);
            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
        }
    }

    (n_time, n_freq, pixels)
}

/// Convert a waveform to RGB image data.
///
/// # Arguments
/// * `y` - Audio signal
/// * `width` - Output image width in pixels
/// * `height` - Output image height in pixels
/// * `color` - Line color (R, G, B)
/// * `bg_color` - Background color (R, G, B)
///
/// # Returns
/// RGB image data as Vec<u8> in RGB format (width * height * 3 bytes)
pub fn waveform_to_rgb(
    y: &[f32],
    width: usize,
    height: usize,
    color: (u8, u8, u8),
    bg_color: (u8, u8, u8),
) -> Vec<u8> {
    if y.is_empty() || width == 0 || height == 0 {
        return Vec::new();
    }

    // Initialize with background color
    let mut pixels = vec![0u8; width * height * 3];
    for i in 0..(width * height) {
        pixels[i * 3] = bg_color.0;
        pixels[i * 3 + 1] = bg_color.1;
        pixels[i * 3 + 2] = bg_color.2;
    }

    // Find min/max for scaling
    let y_min = y.iter().cloned().fold(f32::INFINITY, f32::min);
    let y_max = y.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let y_range = (y_max - y_min).max(1e-10);

    // Downsample signal to width
    let samples_per_pixel = y.len() as f32 / width as f32;

    let mut prev_row: Option<usize> = None;

    for x in 0..width {
        // Get sample range for this pixel
        let start_idx = (x as f32 * samples_per_pixel) as usize;
        let end_idx = (((x + 1) as f32 * samples_per_pixel) as usize).min(y.len());

        if start_idx >= y.len() {
            break;
        }

        // Find min/max in this range for envelope display
        let mut local_min = f32::INFINITY;
        let mut local_max = f32::NEG_INFINITY;
        for &sample in &y[start_idx..end_idx] {
            local_min = local_min.min(sample);
            local_max = local_max.max(sample);
        }

        // Convert to pixel rows (0 = top, height-1 = bottom)
        let row_min = ((1.0 - (local_max - y_min) / y_range) * (height - 1) as f32) as usize;
        let row_max = ((1.0 - (local_min - y_min) / y_range) * (height - 1) as f32) as usize;

        // Draw vertical line from row_min to row_max
        for row in row_min..=row_max.min(height - 1) {
            let idx = (row * width + x) * 3;
            pixels[idx] = color.0;
            pixels[idx + 1] = color.1;
            pixels[idx + 2] = color.2;
        }

        // Connect to previous column if needed
        if let Some(prev) = prev_row {
            let mid_row = (row_min + row_max) / 2;
            let (start, end) = if prev < mid_row {
                (prev, mid_row)
            } else {
                (mid_row, prev)
            };
            for row in start..=end.min(height - 1) {
                let idx = (row * width + x) * 3;
                pixels[idx] = color.0;
                pixels[idx + 1] = color.1;
                pixels[idx + 2] = color.2;
            }
        }

        prev_row = Some((row_min + row_max) / 2);
    }

    pixels
}

/// Save RGB pixel data as a PPM image file.
///
/// PPM is a simple uncompressed format that can be opened by most image viewers.
///
/// # Arguments
/// * `path` - Output file path (should end in .ppm)
/// * `width` - Image width
/// * `height` - Image height
/// * `pixels` - RGB pixel data (width * height * 3 bytes)
pub fn save_ppm(path: &str, width: usize, height: usize, pixels: &[u8]) -> std::io::Result<()> {
    use std::io::Write;

    let mut file = std::fs::File::create(path)?;
    writeln!(file, "P6")?;
    writeln!(file, "{} {}", width, height)?;
    writeln!(file, "255")?;
    file.write_all(pixels)?;
    Ok(())
}

/// Save a spectrogram as a PPM image.
///
/// # Arguments
/// * `data` - 2D array (frequency x time)
/// * `path` - Output file path
/// * `cmap` - Colormap to use
/// * `vmin` - Minimum value for scaling
/// * `vmax` - Maximum value for scaling
///
/// # Example
/// ```ignore
/// use giggle::display::{save_spectrogram, ColorMap};
/// use ndarray::Array2;
///
/// let spec = Array2::<f32>::zeros((128, 100));
/// save_spectrogram(&spec, "spectrogram.ppm", ColorMap::Viridis, None, None).unwrap();
/// ```
pub fn save_spectrogram(
    data: &Array2<f32>,
    path: &str,
    cmap: ColorMap,
    vmin: Option<f32>,
    vmax: Option<f32>,
) -> std::io::Result<()> {
    let (width, height, pixels) = spectrogram_to_rgb(data, cmap, vmin, vmax);
    save_ppm(path, width, height, &pixels)
}

/// Save a waveform as a PPM image.
///
/// # Arguments
/// * `y` - Audio signal
/// * `path` - Output file path
/// * `width` - Image width
/// * `height` - Image height
///
/// # Example
/// ```ignore
/// use giggle::display::save_waveform;
///
/// let signal = vec![0.0f32; 22050];
/// save_waveform(&signal, "waveform.ppm", 800, 200).unwrap();
/// ```
pub fn save_waveform(y: &[f32], path: &str, width: usize, height: usize) -> std::io::Result<()> {
    let pixels = waveform_to_rgb(y, width, height, (31, 119, 180), (255, 255, 255));
    save_ppm(path, width, height, &pixels)
}

/// Scale a spectrogram image by a factor (nearest-neighbor interpolation).
pub fn scale_image(
    pixels: &[u8],
    width: usize,
    height: usize,
    scale: usize,
) -> (usize, usize, Vec<u8>) {
    if scale <= 1 || pixels.is_empty() {
        return (width, height, pixels.to_vec());
    }

    let new_width = width * scale;
    let new_height = height * scale;
    let mut new_pixels = Vec::with_capacity(new_width * new_height * 3);

    for y in 0..new_height {
        let src_y = y / scale;
        for x in 0..new_width {
            let src_x = x / scale;
            let src_idx = (src_y * width + src_x) * 3;
            new_pixels.push(pixels[src_idx]);
            new_pixels.push(pixels[src_idx + 1]);
            new_pixels.push(pixels[src_idx + 2]);
        }
    }

    (new_width, new_height, new_pixels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_colormap_viridis() {
        let (r, _g, _b) = ColorMap::Viridis.to_rgb(0.0);
        assert!(r < 100); // Dark at low values

        let (r, g, _b) = ColorMap::Viridis.to_rgb(1.0);
        assert!(g > 200 || r > 200); // Bright at high values
    }

    #[test]
    fn test_colormap_grayscale() {
        let (r, g, b) = ColorMap::Grayscale.to_rgb(0.0);
        assert_eq!((r, g, b), (0, 0, 0));

        let (r, g, b) = ColorMap::Grayscale.to_rgb(1.0);
        assert_eq!((r, g, b), (255, 255, 255));

        let (r, g, b) = ColorMap::Grayscale.to_rgb(0.5);
        assert_eq!(r, g);
        assert_eq!(g, b);
        assert!(r > 100 && r < 150);
    }

    #[test]
    fn test_colormap_bounds() {
        // Test that all colormaps handle out-of-bounds values
        for cmap in [
            ColorMap::Viridis,
            ColorMap::Magma,
            ColorMap::Inferno,
            ColorMap::Plasma,
            ColorMap::Grayscale,
            ColorMap::Coolwarm,
        ] {
            let (_r, _g, _b) = cmap.to_rgb(-0.5);
            // u8 values are inherently <= 255; just verify no panic on out-of-range input

            let (_r, _g, _b) = cmap.to_rgb(1.5);
            // u8 values are inherently <= 255; just verify no panic on out-of-range input
        }
    }

    #[test]
    fn test_colormap_from_str() {
        assert_eq!("viridis".parse(), Ok(ColorMap::Viridis));
        assert_eq!("MAGMA".parse(), Ok(ColorMap::Magma));
        assert_eq!("gray".parse(), Ok(ColorMap::Grayscale));
        assert_eq!("unknown".parse::<ColorMap>(), Err(()));
    }

    #[test]
    fn test_spectrogram_to_rgb() {
        let data = Array2::from_shape_fn((64, 100), |(f, t)| (f as f32 + t as f32) / 164.0);

        let (width, height, pixels) = spectrogram_to_rgb(&data, ColorMap::Viridis, None, None);

        assert_eq!(width, 100);
        assert_eq!(height, 64);
        assert_eq!(pixels.len(), 100 * 64 * 3);
    }

    #[test]
    fn test_spectrogram_to_rgb_empty() {
        let data = Array2::<f32>::zeros((0, 0));
        let (width, height, pixels) = spectrogram_to_rgb(&data, ColorMap::Viridis, None, None);
        assert_eq!(width, 0);
        assert_eq!(height, 0);
        assert!(pixels.is_empty());
    }

    #[test]
    fn test_waveform_to_rgb() {
        let y: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.01).sin()).collect();
        let pixels = waveform_to_rgb(&y, 200, 100, (0, 0, 255), (255, 255, 255));

        assert_eq!(pixels.len(), 200 * 100 * 3);
    }

    #[test]
    fn test_waveform_to_rgb_empty() {
        let pixels = waveform_to_rgb(&[], 200, 100, (0, 0, 255), (255, 255, 255));
        assert!(pixels.is_empty());
    }

    #[test]
    fn test_scale_image() {
        let pixels = vec![255u8, 0, 0, 0, 255, 0]; // 2x1 image: red, green
        let (w, h, scaled) = scale_image(&pixels, 2, 1, 2);

        assert_eq!(w, 4);
        assert_eq!(h, 2);
        assert_eq!(scaled.len(), 4 * 2 * 3);
    }
}
