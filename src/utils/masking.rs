use ndarray::Array2;

/// Soft masking for source separation.
/// Computes soft masks from multiple magnitude spectrograms that sum to 1.
///
/// # Arguments
/// * `reference` - Reference magnitude spectrogram
/// * `others` - Other magnitude spectrograms to compare against
/// * `power` - Exponent for the Wiener-like mask (default: 1.0)
/// * `split_zeros` - If true, entries with all zeros are distributed evenly
///
/// # Returns
/// Soft mask with same shape as reference, values in [0, 1]
pub fn softmask(
    reference: &Array2<f32>,
    others: &[&Array2<f32>],
    power: f32,
    split_zeros: bool,
) -> Array2<f32> {
    let shape = reference.shape();
    let mut mask = Array2::<f32>::zeros((shape[0], shape[1]));

    for i in 0..shape[0] {
        for j in 0..shape[1] {
            let ref_val = reference[(i, j)].abs().powf(power);
            let mut total = ref_val;

            for other in others {
                if i < other.shape()[0] && j < other.shape()[1] {
                    total += other[(i, j)].abs().powf(power);
                }
            }

            if total > 1e-10 {
                mask[(i, j)] = ref_val / total;
            } else if split_zeros {
                // Distribute evenly when all are zero
                mask[(i, j)] = 1.0 / (1.0 + others.len() as f32);
            } else {
                mask[(i, j)] = 0.0;
            }
        }
    }

    mask
}

/// Binary masking for source separation.
///
/// Creates a hard (binary) mask by selecting the source with maximum magnitude
/// at each time-frequency bin.
///
/// # Arguments
/// * `reference` - Reference magnitude spectrogram
/// * `others` - Other magnitude spectrograms to compare against
///
/// # Returns
/// Binary mask (0 or 1) with same shape as reference
///
/// # Example
/// ```
/// use giggle::utils::binary_mask;
/// use ndarray::Array2;
///
/// let ref_spec = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 0.5, 1.0, 3.0, 0.8]).unwrap();
/// let other = Array2::from_shape_vec((2, 3), vec![0.5, 1.5, 1.0, 0.8, 2.0, 1.5]).unwrap();
/// let mask = binary_mask(&ref_spec, &[&other]);
/// ```
pub fn binary_mask(reference: &Array2<f32>, others: &[&Array2<f32>]) -> Array2<f32> {
    let shape = reference.shape();
    let mut mask = Array2::<f32>::zeros((shape[0], shape[1]));

    for i in 0..shape[0] {
        for j in 0..shape[1] {
            let ref_val = reference[(i, j)].abs();
            let mut is_max = true;

            for other in others {
                if i < other.shape()[0] && j < other.shape()[1] && other[(i, j)].abs() > ref_val {
                    is_max = false;
                    break;
                }
            }

            mask[(i, j)] = if is_max { 1.0 } else { 0.0 };
        }
    }

    mask
}

/// Compute soft masks for multiple sources simultaneously.
///
/// Returns one mask per source such that all masks sum to 1 at each bin.
///
/// # Arguments
/// * `sources` - Slice of magnitude spectrograms, one per source
/// * `power` - Exponent for Wiener-like masking (default: 1.0)
/// * `split_zeros` - Distribute evenly when all sources are zero
///
/// # Returns
/// Vector of soft masks, one per source
///
/// # Example
/// ```
/// use giggle::utils::split_softmask;
/// use ndarray::Array2;
///
/// let s1 = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]).unwrap();
/// let s2 = Array2::from_shape_vec((2, 3), vec![0.5, 1.0, 1.5, 0.5, 1.0, 1.5]).unwrap();
/// let masks = split_softmask(&[&s1, &s2], 1.0, true);
/// assert_eq!(masks.len(), 2);
/// ```
pub fn split_softmask(sources: &[&Array2<f32>], power: f32, split_zeros: bool) -> Vec<Array2<f32>> {
    if sources.is_empty() {
        return Vec::new();
    }

    let shape = sources[0].shape();
    let n_sources = sources.len();
    let mut masks = vec![Array2::<f32>::zeros((shape[0], shape[1])); n_sources];

    for i in 0..shape[0] {
        for j in 0..shape[1] {
            let values: Vec<f32> = sources
                .iter()
                .map(|s| {
                    if i < s.shape()[0] && j < s.shape()[1] {
                        s[(i, j)].abs().powf(power)
                    } else {
                        0.0
                    }
                })
                .collect();

            let total: f32 = values.iter().sum();

            if total > 1e-10 {
                for (k, mask) in masks.iter_mut().enumerate() {
                    mask[(i, j)] = values[k] / total;
                }
            } else if split_zeros {
                // Distribute evenly
                let val = 1.0 / n_sources as f32;
                for mask in &mut masks {
                    mask[(i, j)] = val;
                }
            } else {
                // Leave as zeros
            }
        }
    }

    masks
}

/// Power-based soft masking (Wiener filtering variant).
///
/// Applies soft masking with a power parameter that controls the separation
/// aggressiveness. Higher power makes the mask more selective.
///
/// # Arguments
/// * `reference` - Reference magnitude spectrogram
/// * `mixture` - Mixture magnitude spectrogram
/// * `power` - Exponent for masking (typical range: 0.5-2.0)
///
/// # Returns
/// Soft mask with same shape as reference
pub fn power_softmask(reference: &Array2<f32>, mixture: &Array2<f32>, power: f32) -> Array2<f32> {
    let shape = reference.shape();
    let mut mask = Array2::<f32>::zeros((shape[0], shape[1]));

    for i in 0..shape[0] {
        for j in 0..shape[1] {
            if i < mixture.shape()[0] && j < mixture.shape()[1] {
                let ref_val = reference[(i, j)].abs().powf(power);
                let mix_val = mixture[(i, j)].abs().powf(power);

                if mix_val > 1e-10 {
                    mask[(i, j)] = (ref_val / mix_val).min(1.0);
                }
            }
        }
    }

    mask
}
