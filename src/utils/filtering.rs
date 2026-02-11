use super::{lag_to_recurrence, recurrence_matrix, recurrence_to_lag};
use ndarray::Array2;

/// Nearest-neighbor filter for denoising spectrograms.
///
/// Suppresses non-local noise by retaining only time-frequency bins
/// that are similar to their neighbors. Uses horizontal (time) and
/// vertical (frequency) filtering with median aggregation.
///
/// # Arguments
/// * `data` - Input spectrogram (n_freq x n_time)
/// * `rec` - Recurrence/similarity matrix (n_time x n_time), or None for self-similarity
/// * `aggregate` - Aggregation method: "median" or "mean"
/// * `axis` - Filter axis: 0 for frequency, 1 for time (default: -1 for both)
///
/// # Returns
/// Filtered spectrogram with same shape as input
///
/// # Example
/// ```
/// use giggle::utils::nn_filter;
/// use ndarray::Array2;
///
/// let spec = Array2::from_shape_vec((4, 8), vec![1.0; 32]).unwrap();
/// let filtered = nn_filter(&spec, None, "median", -1);
/// assert_eq!(filtered.shape(), spec.shape());
/// ```
pub fn nn_filter(
    data: &Array2<f32>,
    rec: Option<&Array2<f32>>,
    aggregate: &str,
    axis: i32,
) -> Array2<f32> {
    let n_freq = data.shape()[0];
    let n_time = data.shape()[1];

    if n_freq == 0 || n_time == 0 {
        return data.clone();
    }

    // Compute recurrence if not provided (using correlation)
    let rec_matrix = if let Some(r) = rec {
        if r.shape() != [n_time, n_time] {
            // Invalid recurrence matrix
            return data.clone();
        }
        r.clone()
    } else {
        // Compute simple correlation-based recurrence
        let mut rec_mat = Array2::zeros((n_time, n_time));
        for i in 0..n_time {
            for j in 0..n_time {
                let mut corr = 0.0f64;
                let mut norm_i = 0.0f64;
                let mut norm_j = 0.0f64;

                for k in 0..n_freq {
                    let vi = data[(k, i)] as f64;
                    let vj = data[(k, j)] as f64;
                    corr += vi * vj;
                    norm_i += vi * vi;
                    norm_j += vj * vj;
                }

                let norm_prod = (norm_i * norm_j).sqrt();
                if norm_prod > 1e-10 {
                    rec_mat[(i, j)] = (corr / norm_prod) as f32;
                }
            }
        }
        rec_mat
    };

    let mut result = data.clone();

    // Apply filter based on axis
    if axis == 1 || axis == -1 {
        // Filter along time axis
        for i in 0..n_freq {
            for t in 0..n_time {
                let mut neighbors = Vec::new();

                // Find similar time frames
                for t2 in 0..n_time {
                    if rec_matrix[(t, t2)] > 0.5 {
                        // Threshold for similarity
                        neighbors.push(data[(i, t2)]);
                    }
                }

                if !neighbors.is_empty() {
                    result[(i, t)] = match aggregate {
                        "median" => {
                            neighbors.sort_by(|a, b| a.partial_cmp(b).unwrap());
                            neighbors[neighbors.len() / 2]
                        }
                        "mean" => neighbors.iter().sum::<f32>() / neighbors.len() as f32,
                        _ => data[(i, t)],
                    };
                }
            }
        }
    }

    if axis == 0 || (axis == -1 && axis != 1) {
        // Filter along frequency axis
        for t in 0..n_time {
            for i in 0..n_freq {
                // Use local frequency neighbors
                let mut neighbors = Vec::new();
                let window = 3;

                for di in -window..=window {
                    let i2 = (i as i32 + di).max(0).min(n_freq as i32 - 1) as usize;
                    neighbors.push(result[(i2, t)]);
                }

                result[(i, t)] = match aggregate {
                    "median" => {
                        neighbors.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        neighbors[neighbors.len() / 2]
                    }
                    "mean" => neighbors.iter().sum::<f32>() / neighbors.len() as f32,
                    _ => result[(i, t)],
                };
            }
        }
    }

    result
}

/// Time-lag filter for enhancing repetitive structures.
///
/// Filters spectrograms in the lag domain to suppress sparse/non-repetitive
/// components while enhancing periodic patterns.
///
/// # Arguments
/// * `data` - Input spectrogram (n_freq x n_time)
/// * `lag` - Lag indices to retain (in frames), or None for automatic
/// * `aggregate` - Aggregation method: "mean", "median", or "max"
/// * `norm` - Normalization: None, Some("L1"), or Some("L2")
///
/// # Returns
/// Filtered spectrogram with enhanced repetitive structures
///
/// # Example
/// ```
/// use giggle::utils::timelag_filter;
/// use ndarray::Array2;
///
/// let spec = Array2::from_shape_vec((4, 8), (0..32).map(|x| x as f32).collect()).unwrap();
/// let filtered = timelag_filter(&spec, None, "mean", None);
/// assert_eq!(filtered.shape(), spec.shape());
/// ```
pub fn timelag_filter(
    data: &Array2<f32>,
    lag: Option<&[usize]>,
    aggregate: &str,
    norm: Option<&str>,
) -> Array2<f32> {
    let n_freq = data.shape()[0];
    let n_time = data.shape()[1];

    if n_freq == 0 || n_time == 0 {
        return data.clone();
    }

    // Convert to lag representation
    let rec = recurrence_matrix(data, "affinity", "cosine", 0.0);
    let lag_matrix = recurrence_to_lag(&rec, true, 1);

    // Determine which lags to keep
    let lags_to_keep = if let Some(l) = lag {
        l.to_vec()
    } else {
        // Keep lags with high average similarity
        let mut lag_scores = Vec::new();
        for lag_idx in 0..lag_matrix.shape()[1].min(n_time) {
            let mut sum = 0.0f32;
            let mut count = 0;
            for t in 0..n_time {
                if t + lag_idx < n_time {
                    sum += lag_matrix[(t, lag_idx)].abs();
                    count += 1;
                }
            }
            let score = if count > 0 { sum / count as f32 } else { 0.0 };
            lag_scores.push((lag_idx, score));
        }

        // Keep top 50% of lags
        lag_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let keep_count = (lag_scores.len() / 2).max(1);
        lag_scores
            .iter()
            .take(keep_count)
            .map(|&(idx, _)| idx)
            .collect()
    };

    // Filter lag matrix
    let mut filtered_lag = Array2::zeros(lag_matrix.dim());
    for &lag_idx in &lags_to_keep {
        if lag_idx < lag_matrix.shape()[1] {
            for t in 0..n_time {
                filtered_lag[(t, lag_idx)] = lag_matrix[(t, lag_idx)];
            }
        }
    }

    // Convert back to recurrence
    let filtered_rec = lag_to_recurrence(&filtered_lag, 1);

    // Apply to data
    let mut result = Array2::zeros((n_freq, n_time));
    for t in 0..n_time {
        for f in 0..n_freq {
            let mut values = Vec::new();
            let mut weights = Vec::new();

            for t2 in 0..n_time {
                let weight = filtered_rec[(t, t2)];
                if weight > 0.1 {
                    values.push(data[(f, t2)]);
                    weights.push(weight);
                }
            }

            if !values.is_empty() {
                let val = match aggregate {
                    "median" => {
                        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        values[values.len() / 2]
                    }
                    "max" => values.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
                    _ => {
                        // Weighted mean
                        let sum: f32 = values.iter().zip(weights.iter()).map(|(v, w)| v * w).sum();
                        let weight_sum: f32 = weights.iter().sum();
                        if weight_sum > 0.0 {
                            sum / weight_sum
                        } else {
                            0.0
                        }
                    }
                };
                result[(f, t)] = val;
            }
        }
    }

    // Apply normalization if requested
    if let Some(norm_type) = norm {
        match norm_type {
            "L1" => {
                for f in 0..n_freq {
                    let sum: f32 = (0..n_time).map(|t| result[(f, t)].abs()).sum();
                    if sum > 1e-10 {
                        for t in 0..n_time {
                            result[(f, t)] /= sum;
                        }
                    }
                }
            }
            "L2" => {
                for f in 0..n_freq {
                    let sum_sq: f32 = (0..n_time).map(|t| result[(f, t)].powi(2)).sum();
                    let norm = sum_sq.sqrt();
                    if norm > 1e-10 {
                        for t in 0..n_time {
                            result[(f, t)] /= norm;
                        }
                    }
                }
            }
            _ => {}
        }
    }

    result
}
