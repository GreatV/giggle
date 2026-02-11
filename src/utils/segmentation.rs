use ndarray::Array2;

/// Bottom-up temporal segmentation using constrained agglomerative clustering.
///
/// Uses Ward's method with temporal constraints (only adjacent segments can merge)
/// to partition data into k contiguous segments.
///
/// # Arguments
/// * `data` - Feature matrix (n_features x n_frames)
/// * `k` - Number of segments to produce
///
/// # Returns
/// Left-boundary frame indices of detected segments (always starts with 0)
///
/// # Example
/// ```
/// use giggle::utils::agglomerative;
/// use ndarray::Array2;
///
/// let features = Array2::from_shape_vec((12, 20), (0..240).map(|x| (x as f32).sin()).collect()).unwrap();
/// let bounds = agglomerative(&features, 4);
/// assert_eq!(bounds[0], 0);
/// assert_eq!(bounds.len(), 4);
/// ```
pub fn agglomerative(data: &Array2<f32>, k: usize) -> Vec<usize> {
    let n_features = data.shape()[0];
    let n_frames = data.shape()[1];

    if n_frames == 0 || k == 0 {
        return vec![0];
    }

    if k >= n_frames {
        return (0..n_frames).collect();
    }

    // Initialize: each frame is its own segment
    // Store segment boundaries as (start, end) pairs
    let mut segments: Vec<(usize, usize)> = (0..n_frames).map(|i| (i, i + 1)).collect();

    // Compute feature vectors for each frame (column of data)
    let get_segment_mean = |segs: &[(usize, usize)], idx: usize| -> Vec<f32> {
        let (start, end) = segs[idx];
        let len = end - start;
        let mut mean = vec![0.0f32; n_features];
        for frame in start..end {
            for f in 0..n_features {
                mean[f] += data[(f, frame)];
            }
        }
        for m in mean.iter_mut().take(n_features) {
            *m /= len as f32;
        }
        mean
    };

    // Ward's distance between adjacent segments
    let ward_distance = |mean1: &[f32], n1: usize, mean2: &[f32], n2: usize| -> f32 {
        let mut dist_sq = 0.0f32;
        for i in 0..mean1.len() {
            let diff = mean1[i] - mean2[i];
            dist_sq += diff * diff;
        }
        let factor = (n1 * n2) as f32 / (n1 + n2) as f32;
        factor * dist_sq
    };

    // Merge until we have k segments
    while segments.len() > k {
        let mut best_merge_idx = 0;
        let mut best_distance = f32::INFINITY;

        // Find the pair of adjacent segments with minimum distance
        for i in 0..segments.len() - 1 {
            let mean1 = get_segment_mean(&segments, i);
            let mean2 = get_segment_mean(&segments, i + 1);
            let n1 = segments[i].1 - segments[i].0;
            let n2 = segments[i + 1].1 - segments[i + 1].0;

            let dist = ward_distance(&mean1, n1, &mean2, n2);
            if dist < best_distance {
                best_distance = dist;
                best_merge_idx = i;
            }
        }

        // Merge the best pair
        let merged = (segments[best_merge_idx].0, segments[best_merge_idx + 1].1);
        segments[best_merge_idx] = merged;
        segments.remove(best_merge_idx + 1);
    }

    // Extract left boundaries
    segments.iter().map(|(start, _)| *start).collect()
}

/// Sub-divide segment boundaries by feature clustering.
///
/// Given a set of frame boundaries and a data matrix, each successive
/// interval is partitioned into n_segments by constrained agglomerative
/// clustering.
///
/// # Arguments
/// * `data` - Feature matrix (n_features x n_frames)
/// * `frames` - Array of segment boundaries (e.g., from beat_track or onset_detect)
/// * `n_segments` - Maximum number of sub-segments per interval
///
/// # Returns
/// Array of sub-divided segment boundaries
///
/// # Example
/// ```
/// use giggle::utils::subsegment;
/// use ndarray::Array2;
///
/// let features = Array2::from_shape_vec((12, 100), (0..1200).map(|x| (x as f32).sin()).collect()).unwrap();
/// let frames = vec![0, 25, 50, 75, 100];
/// let sub_bounds = subsegment(&features, &frames, 2);
/// assert!(sub_bounds.len() >= frames.len());
/// ```
pub fn subsegment(data: &Array2<f32>, frames: &[usize], n_segments: usize) -> Vec<usize> {
    let n_frames = data.shape()[1];

    if frames.is_empty() || n_segments == 0 || n_frames == 0 {
        return vec![0];
    }

    // Ensure frames are valid and include 0 and n_frames
    let mut fixed_frames: Vec<usize> = frames.iter().cloned().filter(|&f| f < n_frames).collect();
    if fixed_frames.is_empty() || fixed_frames[0] != 0 {
        fixed_frames.insert(0, 0);
    }
    if *fixed_frames.last().unwrap() != n_frames {
        fixed_frames.push(n_frames);
    }
    fixed_frames.sort();
    fixed_frames.dedup();

    let mut boundaries = Vec::new();

    // Process each segment
    for i in 0..fixed_frames.len() - 1 {
        let seg_start = fixed_frames[i];
        let seg_end = fixed_frames[i + 1];
        let seg_len = seg_end - seg_start;

        if seg_len == 0 {
            continue;
        }

        // Extract segment data
        let mut seg_data = Array2::<f32>::zeros((data.shape()[0], seg_len));
        for f in 0..data.shape()[0] {
            for t in 0..seg_len {
                seg_data[(f, t)] = data[(f, seg_start + t)];
            }
        }

        // Apply agglomerative clustering
        let k = n_segments.min(seg_len);
        let sub_bounds = agglomerative(&seg_data, k);

        // Offset back to original frame indices
        for &b in &sub_bounds {
            boundaries.push(seg_start + b);
        }
    }

    boundaries.sort();
    boundaries.dedup();
    boundaries
}

/// Create a 2D diagonal smoothing filter.
///
/// # Arguments
/// * `n` - Length of the filter
/// * `slope` - Slope of the diagonal (1.0 = 45 degrees)
/// * `zero_mean` - If true, make the filter zero-mean
///
/// # Returns
/// 2D filter kernel
fn diagonal_filter(n: usize, slope: f32, zero_mean: bool) -> Array2<f32> {
    if n == 0 {
        return Array2::zeros((1, 1));
    }

    // Create a Hann window
    let window = crate::window::hann(n);

    // Compute the size of the rotated kernel
    let angle = slope.atan();
    let cos_a = angle.cos();
    let sin_a = angle.sin();

    // For a diagonal at angle theta, we need a square large enough
    // to fit the rotated diagonal
    let size = ((n as f32) * (cos_a.abs() + sin_a.abs())).ceil() as usize;
    let size = size.max(n);

    let mut kernel = Array2::<f32>::zeros((size, size));

    // Place the window along the diagonal with the given slope
    let center = size as f32 / 2.0;

    for (i, &w) in window.iter().enumerate().take(n) {
        let t = i as f32 - n as f32 / 2.0;
        let x = center + t * cos_a;
        let y = center + t * sin_a;

        let xi = x.round() as isize;
        let yi = y.round() as isize;

        if xi >= 0 && xi < size as isize && yi >= 0 && yi < size as isize {
            kernel[(yi as usize, xi as usize)] += w;
        }
    }

    // Normalize to sum to 1
    let sum: f32 = kernel.iter().sum();
    if sum > 1e-10 {
        for val in kernel.iter_mut() {
            *val /= sum;
        }
    }

    // Make zero-mean if requested
    if zero_mean {
        let mean = kernel.iter().sum::<f32>() / (kernel.len() as f32);
        for val in kernel.iter_mut() {
            *val -= mean;
        }
    }

    kernel
}

/// 2D convolution for path enhancement.
fn convolve2d(input: &Array2<f32>, kernel: &Array2<f32>) -> Array2<f32> {
    let (in_h, in_w) = (input.shape()[0], input.shape()[1]);
    let (k_h, k_w) = (kernel.shape()[0], kernel.shape()[1]);

    if in_h == 0 || in_w == 0 || k_h == 0 || k_w == 0 {
        return input.clone();
    }

    let pad_h = k_h / 2;
    let pad_w = k_w / 2;

    let mut output = Array2::<f32>::zeros((in_h, in_w));

    for i in 0..in_h {
        for j in 0..in_w {
            let mut sum = 0.0f32;
            for ki in 0..k_h {
                for kj in 0..k_w {
                    let ii = i as isize + ki as isize - pad_h as isize;
                    let jj = j as isize + kj as isize - pad_w as isize;

                    if ii >= 0 && ii < in_h as isize && jj >= 0 && jj < in_w as isize {
                        sum += input[(ii as usize, jj as usize)] * kernel[(ki, kj)];
                    }
                }
            }
            output[(i, j)] = sum;
        }
    }

    output
}

/// Multi-angle path enhancement for self- and cross-similarity matrices.
///
/// Convolves diagonal smoothing filters at multiple angles with a similarity
/// matrix and aggregates by element-wise maximum. This enhances diagonal
/// paths that represent tempo changes in audio.
///
/// # Arguments
/// * `r` - Self- or cross-similarity matrix
/// * `n` - Length of the smoothing filter
/// * `max_ratio` - Maximum tempo ratio to support (default: 2.0)
/// * `min_ratio` - Minimum tempo ratio (default: 1/max_ratio)
/// * `n_filters` - Number of different smoothing filters
/// * `zero_mean` - If true, use zero-mean filters
/// * `clip` - If true, clip output to non-negative values
///
/// # Returns
/// Smoothed similarity matrix
///
/// # Example
/// ```
/// use giggle::utils::{recurrence_matrix, path_enhance};
/// use ndarray::Array2;
///
/// let features = Array2::from_shape_vec((12, 50), (0..600).map(|x| (x as f32).sin()).collect()).unwrap();
/// let rec = recurrence_matrix(&features, "affinity", "cosine", 0.5);
/// let enhanced = path_enhance(&rec, 11, 2.0, None, 5, false, true);
/// assert_eq!(enhanced.shape(), rec.shape());
/// ```
pub fn path_enhance(
    r: &Array2<f32>,
    n: usize,
    max_ratio: f32,
    min_ratio: Option<f32>,
    n_filters: usize,
    zero_mean: bool,
    clip: bool,
) -> Array2<f32> {
    let (h, w) = (r.shape()[0], r.shape()[1]);

    if h == 0 || w == 0 || n == 0 || n_filters == 0 {
        return r.clone();
    }

    let min_r = min_ratio.unwrap_or(1.0 / max_ratio);

    if min_r > max_ratio {
        return r.clone();
    }

    let mut r_smooth: Option<Array2<f32>> = None;

    // Generate filters at evenly spaced ratios (log scale)
    let log_min = min_r.log2();
    let log_max = max_ratio.log2();

    for i in 0..n_filters {
        let ratio = if n_filters > 1 {
            2.0_f32.powf(log_min + (log_max - log_min) * i as f32 / (n_filters - 1) as f32)
        } else {
            (min_r * max_ratio).sqrt()
        };

        let kernel = diagonal_filter(n, ratio, zero_mean);
        let convolved = convolve2d(r, &kernel);

        r_smooth = Some(match r_smooth {
            None => convolved,
            Some(prev) => {
                let mut result = Array2::<f32>::zeros((h, w));
                for i in 0..h {
                    for j in 0..w {
                        result[(i, j)] = prev[(i, j)].max(convolved[(i, j)]);
                    }
                }
                result
            }
        });
    }

    let mut result = r_smooth.unwrap_or_else(|| r.clone());

    // Clip to non-negative if requested
    if clip {
        for val in result.iter_mut() {
            if *val < 0.0 {
                *val = 0.0;
            }
        }
    }

    result
}
