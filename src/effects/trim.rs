pub fn trim(y: &[f32], top_db: f32) -> (Vec<f32>, (usize, usize)) {
    if y.is_empty() {
        return (Vec::new(), (0, 0));
    }
    let max_amp = y.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    if max_amp <= 0.0 {
        return (Vec::new(), (0, 0));
    }

    let threshold = max_amp / 10f32.powf(top_db / 20.0);

    let mut start = 0usize;
    while start < y.len() && y[start].abs() < threshold {
        start += 1;
    }

    let mut end = y.len();
    while end > start && y[end - 1].abs() < threshold {
        end -= 1;
    }

    if start >= end {
        return (Vec::new(), (0, 0));
    }

    (y[start..end].to_vec(), (start, end))
}

pub fn split(y: &[f32], top_db: f32, min_interval: usize) -> Vec<(usize, usize)> {
    if y.is_empty() {
        return Vec::new();
    }
    let max_amp = y.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    if max_amp <= 0.0 {
        return Vec::new();
    }
    let threshold = max_amp / 10f32.powf(top_db / 20.0);

    let mut intervals = Vec::new();
    let mut in_region = false;
    let mut start = 0usize;

    for (i, v) in y.iter().enumerate() {
        if v.abs() >= threshold {
            if !in_region {
                in_region = true;
                start = i;
            }
        } else if in_region {
            let end = i;
            if end - start >= min_interval {
                intervals.push((start, end));
            }
            in_region = false;
        }
    }

    if in_region {
        let end = y.len();
        if end - start >= min_interval {
            intervals.push((start, end));
        }
    }

    intervals
}
