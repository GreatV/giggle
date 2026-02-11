use giggle::effects::phase_vocoder::time_stretch;
use giggle::effects::time_pitch::pitch_shift;

#[test]
fn time_stretch_changes_length() {
    let sr = 22050u32;
    let y = vec![0.0f32; sr as usize];
    let stretched = time_stretch(&y, 0.5, 512, 128).unwrap();
    assert!(stretched.len() > y.len());
    let faster = time_stretch(&y, 2.0, 512, 128).unwrap();
    assert!(faster.len() < y.len());
}

#[test]
fn pitch_shift_length_reasonable() {
    let sr = 22050u32;
    let y = vec![0.0f32; sr as usize];
    let shifted = pitch_shift(&y, sr, 4.0, 512, 128).unwrap();
    let diff = if shifted.len() > y.len() {
        shifted.len() - y.len()
    } else {
        y.len() - shifted.len()
    };
    assert!(diff < sr as usize / 2);
}
