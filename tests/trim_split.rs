use giggle::effects::trim::{split, trim};

#[test]
fn trim_removes_silence() {
    let mut y = vec![0.0f32; 100];
    y.extend(vec![0.5f32; 50]);
    y.extend(vec![0.0f32; 25]);

    let (trimmed, (start, end)) = trim(&y, 40.0);
    assert_eq!(start, 100);
    assert_eq!(end, 150);
    assert_eq!(trimmed.len(), 50);
}

#[test]
fn split_detects_regions() {
    let mut y = vec![0.0f32; 10];
    y.extend(vec![0.8f32; 20]);
    y.extend(vec![0.0f32; 5]);
    y.extend(vec![0.6f32; 15]);
    let intervals = split(&y, 40.0, 5);
    assert_eq!(intervals.len(), 2);
    assert_eq!(intervals[0], (10, 30));
    assert_eq!(intervals[1], (35, 50));
}
