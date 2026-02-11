use giggle::feature::basic;

#[test]
fn zcr_edge_cases() {
    assert_eq!(basic::zero_crossing_rate(&[]), 0.0);
    assert_eq!(basic::zero_crossing_rate(&[0.0]), 0.0);
    let y = [1.0, -1.0, 1.0, -1.0];
    let z = basic::zero_crossing_rate(&y);
    assert!((z - 1.0).abs() < 1e-6);
}

#[test]
fn rms_edge_cases() {
    assert_eq!(basic::rms(&[]), 0.0);
    assert_eq!(basic::rms(&[0.0, 0.0]), 0.0);
    let y = [1.0, -1.0];
    let r = basic::rms(&y);
    assert!((r - 1.0).abs() < 1e-6);
}

#[test]
fn rms_frames_centered() {
    let y = vec![1.0f32; 8];
    let out = basic::rms_frames(&y, 4, 2, true).unwrap();
    assert_eq!(out.len(), 5);
}
