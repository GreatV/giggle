use giggle::onset::{detect::onset_detect, strength::onset_strength};

#[test]
fn onset_strength_nonempty() {
    let mut y = vec![0.0f32; 512];
    y.extend(vec![1.0f32; 512]);
    let env = onset_strength(&y, 256, 64).unwrap();
    assert!(!env.is_empty());
}

#[test]
fn onset_detect_simple_peak() {
    let mut y = vec![0.0f32; 512];
    y.extend(vec![1.0f32; 512]);
    let peaks = onset_detect(&y, 256, 64, 0.1).unwrap();
    assert!(!peaks.is_empty());
}
