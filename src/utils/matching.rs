/// Match time intervals from reference to estimated based on overlap.
///
/// For each interval in the reference set, finds the estimated interval
/// with maximum overlap. Uses Jaccard index (intersection over union).
///
/// # Arguments
/// * `intervals_ref` - Reference intervals as (start, end) pairs in seconds
/// * `intervals_est` - Estimated intervals as (start, end) pairs in seconds
/// * `threshold` - Minimum Jaccard index to consider a match (default: 0.5)
///
/// # Returns
/// Vector of (ref_idx, est_idx) pairs for matched intervals
///
/// # Example
/// ```
/// use giggle::utils::match_intervals;
///
/// let reference = vec![(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)];
/// let estimated = vec![(0.1, 1.1), (1.9, 3.1)];
/// let matches = match_intervals(&reference, &estimated, 0.3);
/// assert!(matches.len() > 0);
/// ```
pub fn match_intervals(
    intervals_ref: &[(f32, f32)],
    intervals_est: &[(f32, f32)],
    threshold: f32,
) -> Vec<(usize, usize)> {
    let mut matches = Vec::new();

    for (ref_idx, &(ref_start, ref_end)) in intervals_ref.iter().enumerate() {
        if ref_end <= ref_start {
            continue; // Invalid interval
        }

        let mut best_match = None;
        let mut best_score = threshold;

        for (est_idx, &(est_start, est_end)) in intervals_est.iter().enumerate() {
            if est_end <= est_start {
                continue; // Invalid interval
            }

            // Compute Jaccard index (intersection over union)
            let intersection_start = ref_start.max(est_start);
            let intersection_end = ref_end.min(est_end);
            let intersection = (intersection_end - intersection_start).max(0.0);

            let union_start = ref_start.min(est_start);
            let union_end = ref_end.max(est_end);
            let union = union_end - union_start;

            let jaccard = if union > 0.0 {
                intersection / union
            } else {
                0.0
            };

            if jaccard > best_score {
                best_score = jaccard;
                best_match = Some(est_idx);
            }
        }

        if let Some(est_idx) = best_match {
            matches.push((ref_idx, est_idx));
        }
    }

    matches
}

/// Match discrete events (time points) between reference and estimated.
///
/// For each reference event, finds the nearest estimated event within
/// a specified time window.
///
/// # Arguments
/// * `events_ref` - Reference event times in seconds
/// * `events_est` - Estimated event times in seconds
/// * `window` - Maximum time difference to consider a match (default: 0.05 seconds)
///
/// # Returns
/// Vector of (ref_idx, est_idx) pairs for matched events
///
/// # Example
/// ```
/// use giggle::utils::match_events;
///
/// let reference = vec![0.5, 1.0, 1.5, 2.0];
/// let estimated = vec![0.51, 1.02, 1.48];
/// let matches = match_events(&reference, &estimated, 0.1);
/// assert_eq!(matches.len(), 3);
/// ```
pub fn match_events(events_ref: &[f32], events_est: &[f32], window: f32) -> Vec<(usize, usize)> {
    let mut matches = Vec::new();
    let mut used_est = vec![false; events_est.len()];

    for (ref_idx, &ref_time) in events_ref.iter().enumerate() {
        let mut best_match = None;
        let mut best_distance = window;

        for (est_idx, &est_time) in events_est.iter().enumerate() {
            if used_est[est_idx] {
                continue; // Already matched
            }

            let distance = (ref_time - est_time).abs();
            if distance < best_distance {
                best_distance = distance;
                best_match = Some(est_idx);
            }
        }

        if let Some(est_idx) = best_match {
            matches.push((ref_idx, est_idx));
            used_est[est_idx] = true;
        }
    }

    matches
}
