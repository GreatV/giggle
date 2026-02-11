use ndarray::Array2;
use num_complex::Complex32;

/// Harmonic-Percussive Source Separation using median filtering.
///
/// # Arguments
/// * `stft` - Complex STFT spectrogram (freq_bins x time_frames)
/// * `kernel_size` - (harmonic_kernel, percussive_kernel) for median filtering
/// * `power` - Exponent for Wiener-like masking (default 2.0)
/// * `margin` - Margin for soft masking in dB (default 1.0)
pub fn hpss(
    stft: &Array2<Complex32>,
    kernel_size: (usize, usize),
    power: f32,
    margin: f32,
) -> (Array2<Complex32>, Array2<Complex32>) {
    let (n_freq, n_frames) = (stft.shape()[0], stft.shape()[1]);

    let mut mag = Array2::<f32>::zeros((n_freq, n_frames));
    for ((i, j), &val) in stft.indexed_iter() {
        mag[(i, j)] = val.norm();
    }

    let harmonic = median_filter_2d(&mag, (1, kernel_size.0));
    let percussive = median_filter_2d(&mag, (kernel_size.1, 1));

    let margin_factor = 10.0f32.powf(margin / 20.0);
    let mut mask_h = Array2::<f32>::zeros((n_freq, n_frames));
    let mut mask_p = Array2::<f32>::zeros((n_freq, n_frames));

    for i in 0..n_freq {
        for j in 0..n_frames {
            let h = harmonic[(i, j)].powf(power);
            let p = percussive[(i, j)].powf(power);
            let total = h + p + 1e-10;
            mask_h[(i, j)] = (h * margin_factor) / total;
            mask_p[(i, j)] = (p * margin_factor) / total;
        }
    }

    let mut stft_h = Array2::<Complex32>::zeros((n_freq, n_frames));
    let mut stft_p = Array2::<Complex32>::zeros((n_freq, n_frames));

    for i in 0..n_freq {
        for j in 0..n_frames {
            stft_h[(i, j)] = stft[(i, j)] * mask_h[(i, j)];
            stft_p[(i, j)] = stft[(i, j)] * mask_p[(i, j)];
        }
    }

    (stft_h, stft_p)
}

/// 2D median filter with specified kernel size.
fn median_filter_2d(input: &Array2<f32>, kernel_size: (usize, usize)) -> Array2<f32> {
    let (n_freq, n_frames) = (input.shape()[0], input.shape()[1]);
    let mut output = Array2::<f32>::zeros((n_freq, n_frames));
    let (kh, kw) = kernel_size;

    for i in 0..n_freq {
        for j in 0..n_frames {
            let mut window = Vec::new();

            let i_start = i.saturating_sub(kh / 2);
            let i_end = (i + kh / 2 + 1).min(n_freq);
            let j_start = j.saturating_sub(kw / 2);
            let j_end = (j + kw / 2 + 1).min(n_frames);

            for ii in i_start..i_end {
                for jj in j_start..j_end {
                    window.push(input[(ii, jj)]);
                }
            }

            window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            output[(i, j)] = if window.is_empty() {
                0.0
            } else {
                window[window.len() / 2]
            };
        }
    }

    output
}

/// Extract harmonic component from STFT.
pub fn harmonic(stft: &Array2<Complex32>, kernel_size: usize) -> Array2<Complex32> {
    hpss(stft, (kernel_size, 17), 2.0, 1.0).0
}

/// Extract percussive component from STFT.
pub fn percussive(stft: &Array2<Complex32>, kernel_size: usize) -> Array2<Complex32> {
    hpss(stft, (31, kernel_size), 2.0, 1.0).1
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn create_test_stft() -> Array2<Complex32> {
        Array2::from_shape_vec(
            (10, 20),
            (0..200)
                .map(|i| Complex32::new((i as f32 * 0.1).sin(), (i as f32 * 0.05).cos()))
                .collect(),
        )
        .unwrap()
    }

    #[test]
    fn test_hpss_shape() {
        let stft = create_test_stft();
        let (h, p) = hpss(&stft, (31, 17), 2.0, 1.0);

        assert_eq!(h.shape(), stft.shape());
        assert_eq!(p.shape(), stft.shape());
    }

    #[test]
    fn test_hpss_energy_conservation() {
        let stft = create_test_stft();
        let (h, p) = hpss(&stft, (31, 17), 2.0, 1.0);

        let original_energy: f32 = stft.iter().map(|c| c.norm_sqr()).sum();
        let harmonic_energy: f32 = h.iter().map(|c| c.norm_sqr()).sum();
        let percussive_energy: f32 = p.iter().map(|c| c.norm_sqr()).sum();

        assert!(harmonic_energy > 0.0);
        assert!(percussive_energy > 0.0);
        assert!(harmonic_energy + percussive_energy <= original_energy * 1.5);
    }

    #[test]
    fn test_harmonic_extraction() {
        let stft = create_test_stft();
        let h = harmonic(&stft, 31);

        assert_eq!(h.shape(), stft.shape());
        assert!(h.iter().any(|c| c.norm() > 0.0));
    }

    #[test]
    fn test_percussive_extraction() {
        let stft = create_test_stft();
        let p = percussive(&stft, 17);

        assert_eq!(p.shape(), stft.shape());
        assert!(p.iter().any(|c| c.norm() > 0.0));
    }

    #[test]
    fn test_hpss_empty() {
        let stft = Array2::<Complex32>::zeros((0, 0));
        let (h, p) = hpss(&stft, (31, 17), 2.0, 1.0);

        assert_eq!(h.shape(), &[0, 0]);
        assert_eq!(p.shape(), &[0, 0]);
    }

    #[test]
    fn test_median_filter_basic() {
        let input =
            Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
                .unwrap();

        let filtered = median_filter_2d(&input, (3, 3));
        assert_eq!(filtered.shape(), input.shape());
        assert_relative_eq!(filtered[(1, 1)], 5.0, epsilon = 0.01);
    }
}
