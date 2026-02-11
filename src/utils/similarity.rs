use ndarray::Array2;

/// Compute cross-similarity matrix between two feature sequences.
///
/// Computes pairwise cosine similarity between frames of two feature matrices.
/// Higher values indicate more similar feature vectors.
///
/// # Arguments
/// * `data_ref` - Reference feature matrix (n_features x n_frames_ref)
/// * `data_est` - Comparison feature matrix (n_features x n_frames_est)
/// * `metric` - Distance metric: "cosine", "euclidean", or "manhattan"
///
/// # Returns
/// Similarity matrix (n_frames_ref x n_frames_est)
///
/// # Example
/// ```
/// use giggle::utils::cross_similarity;
/// use ndarray::Array2;
///
/// let ref_feat = Array2::from_shape_vec((2, 3), vec![
///     1.0, 2.0, 3.0,
///     1.0, 2.0, 3.0,
/// ]).unwrap();
/// let est_feat = Array2::from_shape_vec((2, 2), vec![
///     1.0, 3.0,
///     1.0, 3.0,
/// ]).unwrap();
///
/// let sim = cross_similarity(&ref_feat, &est_feat, "cosine");
/// assert_eq!(sim.shape(), &[3, 2]);
/// ```
pub fn cross_similarity(
    data_ref: &Array2<f32>,
    data_est: &Array2<f32>,
    metric: &str,
) -> Array2<f32> {
    let n_features = data_ref.shape()[0];
    let n_frames_ref = data_ref.shape()[1];
    let n_frames_est = data_est.shape()[1];

    if data_est.shape()[0] != n_features {
        // Feature dimensions must match
        return Array2::zeros((n_frames_ref, n_frames_est));
    }

    let mut similarity = Array2::zeros((n_frames_ref, n_frames_est));

    match metric {
        "cosine" => {
            // Compute cosine similarity
            for i in 0..n_frames_ref {
                for j in 0..n_frames_est {
                    let mut dot_product = 0.0f64;
                    let mut norm_ref = 0.0f64;
                    let mut norm_est = 0.0f64;

                    for k in 0..n_features {
                        let v_ref = data_ref[(k, i)] as f64;
                        let v_est = data_est[(k, j)] as f64;

                        dot_product += v_ref * v_est;
                        norm_ref += v_ref * v_ref;
                        norm_est += v_est * v_est;
                    }

                    let norm_product = (norm_ref * norm_est).sqrt();
                    if norm_product > 1e-10 {
                        similarity[(i, j)] = (dot_product / norm_product) as f32;
                    }
                }
            }
        }
        "euclidean" => {
            // Compute negative Euclidean distance (higher = more similar)
            for i in 0..n_frames_ref {
                for j in 0..n_frames_est {
                    let mut dist_sq = 0.0f64;

                    for k in 0..n_features {
                        let diff = data_ref[(k, i)] as f64 - data_est[(k, j)] as f64;
                        dist_sq += diff * diff;
                    }

                    similarity[(i, j)] = -(dist_sq.sqrt() as f32);
                }
            }
        }
        "manhattan" => {
            // Compute negative Manhattan distance
            for i in 0..n_frames_ref {
                for j in 0..n_frames_est {
                    let mut dist = 0.0f64;

                    for k in 0..n_features {
                        let diff = (data_ref[(k, i)] - data_est[(k, j)]).abs() as f64;
                        dist += diff;
                    }

                    similarity[(i, j)] = -(dist as f32);
                }
            }
        }
        _ => {
            // Unknown metric, return zeros
        }
    }

    similarity
}

/// Compute self-similarity (recurrence) matrix for a feature sequence.
///
/// A recurrence matrix identifies similar frames within a single feature matrix,
/// useful for finding repetitive structures in music.
///
/// # Arguments
/// * `data` - Feature matrix (n_features x n_frames)
/// * `mode` - Comparison mode: "connectivity" (binary) or "affinity" (continuous similarity)
/// * `metric` - Distance metric: "cosine", "euclidean", or "manhattan"
/// * `threshold` - Similarity threshold for connectivity mode (default: 0.0)
///
/// # Returns
/// Recurrence matrix (n_frames x n_frames)
///
/// # Example
/// ```
/// use giggle::utils::recurrence_matrix;
/// use ndarray::Array2;
///
/// let features = Array2::from_shape_vec((2, 4), vec![
///     1.0, 2.0, 1.0, 3.0,
///     1.0, 2.0, 1.0, 3.0,
/// ]).unwrap();
///
/// let rec = recurrence_matrix(&features, "connectivity", "cosine", 0.9);
/// assert_eq!(rec.shape(), &[4, 4]);
/// ```
pub fn recurrence_matrix(
    data: &Array2<f32>,
    mode: &str,
    metric: &str,
    threshold: f32,
) -> Array2<f32> {
    let _n_features = data.shape()[0];
    let n_frames = data.shape()[1];

    // Compute self-similarity
    let similarity = cross_similarity(data, data, metric);

    if mode == "connectivity" {
        // Binary connectivity based on threshold
        let mut result = Array2::zeros((n_frames, n_frames));
        for i in 0..n_frames {
            for j in 0..n_frames {
                if similarity[(i, j)] >= threshold {
                    result[(i, j)] = 1.0;
                }
            }
        }
        result
    } else {
        // Return continuous similarity (affinity)
        similarity
    }
}

/// Convert recurrence matrix to lag matrix.
///
/// Converts a recurrence matrix (time vs time) to a lag matrix (time vs lag),
/// where each column represents a specific time offset.
///
/// # Arguments
/// * `rec` - Recurrence matrix (n_frames x n_frames)
/// * `pad` - Whether to pad with zeros for uniform size
/// * `axis` - Axis along which to compute lags (default: 1)
///
/// # Returns
/// Lag matrix where column k represents lag k
///
/// # Example
/// ```
/// use giggle::utils::{recurrence_matrix, recurrence_to_lag};
/// use ndarray::Array2;
///
/// let features = Array2::from_shape_vec((2, 4), vec![1.0; 8]).unwrap();
/// let rec = recurrence_matrix(&features, "affinity", "cosine", 0.0);
/// let lag = recurrence_to_lag(&rec, true, 1);
/// ```
pub fn recurrence_to_lag(rec: &Array2<f32>, pad: bool, axis: usize) -> Array2<f32> {
    let n = rec.shape()[0];

    if axis != 1 {
        // Only support axis=1 for now
        return rec.clone();
    }

    if !pad {
        // No padding: extract diagonals
        let mut lag = Array2::zeros((n, n));
        for offset in 0..n {
            for i in 0..(n - offset) {
                lag[(i, offset)] = rec[(i, i + offset)];
            }
        }
        lag
    } else {
        // With padding: full size maintained
        let mut lag = Array2::zeros((n, n));
        for offset in 0..n {
            for i in 0..(n - offset) {
                lag[(i, offset)] = rec[(i, i + offset)];
            }
        }
        lag
    }
}

/// Convert lag matrix back to recurrence matrix.
///
/// Inverse of recurrence_to_lag, reconstructs a symmetric recurrence matrix
/// from a lag representation.
///
/// # Arguments
/// * `lag` - Lag matrix (n_frames x n_lags)
/// * `axis` - Axis along which lags were computed (default: 1)
///
/// # Returns
/// Recurrence matrix (n_frames x n_frames)
///
/// # Example
/// ```
/// use giggle::utils::lag_to_recurrence;
/// use ndarray::Array2;
///
/// let lag = Array2::from_shape_vec((4, 4), vec![
///     1.0, 0.5, 0.2, 0.1,
///     1.0, 0.5, 0.2, 0.0,
///     1.0, 0.5, 0.0, 0.0,
///     1.0, 0.0, 0.0, 0.0,
/// ]).unwrap();
/// let rec = lag_to_recurrence(&lag, 1);
/// assert_eq!(rec.shape(), &[4, 4]);
/// ```
pub fn lag_to_recurrence(lag: &Array2<f32>, axis: usize) -> Array2<f32> {
    let n = lag.shape()[0];

    if axis != 1 {
        return lag.clone();
    }

    let mut rec = Array2::zeros((n, n));

    // Fill upper triangle from lag matrix
    for offset in 0..lag.shape()[1].min(n) {
        for i in 0..(n - offset) {
            let val = lag[(i, offset)];
            rec[(i, i + offset)] = val;
            rec[(i + offset, i)] = val; // Symmetric
        }
    }

    rec
}
