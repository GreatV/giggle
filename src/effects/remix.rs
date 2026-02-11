/// Find all zero crossing indices in the signal.
fn zero_crossings(y: &[f32]) -> Vec<usize> {
    if y.len() < 2 {
        return Vec::new();
    }

    let mut crossings = Vec::new();
    for i in 1..y.len() {
        // A zero crossing occurs when the sign changes
        if (y[i - 1] >= 0.0 && y[i] < 0.0) || (y[i - 1] < 0.0 && y[i] >= 0.0) {
            crossings.push(i);
        }
    }
    crossings
}

/// Find the index in `events` that is closest to `query`.
fn match_event(query: usize, events: &[usize]) -> usize {
    if events.is_empty() {
        return query;
    }

    // Binary search for the closest event
    match events.binary_search(&query) {
        Ok(idx) => events[idx],
        Err(idx) => {
            if idx == 0 {
                events[0]
            } else if idx >= events.len() {
                events[events.len() - 1]
            } else {
                // Compare with neighbors
                let before = events[idx - 1];
                let after = events[idx];
                if query - before <= after - query {
                    before
                } else {
                    after
                }
            }
        }
    }
}

/// Remix an audio signal by re-ordering time intervals.
///
/// # Arguments
/// * `y` - Audio time series (mono)
/// * `intervals` - Iterator of (start, end) sample indices defining segments
/// * `align_zeros` - If true, snap interval boundaries to nearest zero-crossings
///
/// # Returns
/// Remixed audio with segments concatenated in the order specified by `intervals`
///
/// # Example
/// ```
/// use giggle::effects::remix::remix;
///
/// // Create a simple signal
/// let y: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.1).sin()).collect();
///
/// // Define intervals to reverse the order
/// let intervals = vec![(500, 1000), (0, 500)];
///
/// // Remix (without zero-crossing alignment for this example)
/// let remixed = remix(&y, intervals.into_iter(), false);
/// assert_eq!(remixed.len(), 1000);
/// ```
pub fn remix<I>(y: &[f32], intervals: I, align_zeros: bool) -> Vec<f32>
where
    I: IntoIterator<Item = (usize, usize)>,
{
    if y.is_empty() {
        return Vec::new();
    }

    let zeros = if align_zeros {
        let mut z = zero_crossings(y);
        // Add end of signal
        z.push(y.len());
        z
    } else {
        Vec::new()
    };

    let mut output = Vec::new();

    for (start, end) in intervals {
        let (actual_start, actual_end) = if align_zeros && !zeros.is_empty() {
            (match_event(start, &zeros), match_event(end, &zeros))
        } else {
            (start.min(y.len()), end.min(y.len()))
        };

        if actual_start < actual_end && actual_end <= y.len() {
            output.extend_from_slice(&y[actual_start..actual_end]);
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_crossings() {
        let y = vec![1.0, 0.5, -0.5, -1.0, 0.5, 1.0];
        let crossings = zero_crossings(&y);
        assert_eq!(crossings, vec![2, 4]);
    }

    #[test]
    fn test_match_event() {
        let events = vec![10, 20, 30, 40];
        assert_eq!(match_event(5, &events), 10);
        assert_eq!(match_event(15, &events), 10);
        assert_eq!(match_event(16, &events), 20);
        assert_eq!(match_event(25, &events), 20);
        assert_eq!(match_event(50, &events), 40);
    }

    #[test]
    fn test_remix_basic() {
        let y: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let intervals = vec![(50, 100), (0, 50)];
        let remixed = remix(&y, intervals, false);

        assert_eq!(remixed.len(), 100);
        assert_eq!(remixed[0], 50.0);
        assert_eq!(remixed[49], 99.0);
        assert_eq!(remixed[50], 0.0);
        assert_eq!(remixed[99], 49.0);
    }

    #[test]
    fn test_remix_with_align_zeros() {
        // Create signal with zero crossings at indices 25 and 75
        let y: Vec<f32> = (0..100)
            .map(|i| {
                if i < 25 {
                    1.0
                } else if i < 75 {
                    -1.0
                } else {
                    1.0
                }
            })
            .collect();

        let intervals = vec![(20, 80)];
        let remixed = remix(&y, intervals, true);

        // Should snap to zero crossings at 25 and 75
        assert_eq!(remixed.len(), 50); // 75 - 25
    }

    #[test]
    fn test_remix_empty() {
        let y: Vec<f32> = Vec::new();
        let intervals = vec![(0, 10)];
        let remixed = remix(&y, intervals, false);
        assert!(remixed.is_empty());
    }
}
