use giggle::feature::tempo::tempo;

#[test]
fn tempo_detects_pulse_train() {
    let sr = 22050u32;
    let hop = 256usize;
    let bpm = 120.0f32;
    let period = (sr as f32 * 60.0 / bpm) as usize;

    let mut y = vec![0.0f32; sr as usize * 2];
    for i in (0..y.len()).step_by(period) {
        if i < y.len() {
            y[i] = 1.0;
        }
    }

    let est = tempo(&y, sr, 512, hop).unwrap();
    assert!((est - bpm).abs() < 10.0, "tempo off: {est}");
}
