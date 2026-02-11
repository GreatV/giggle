use giggle::convert;

#[test]
fn convert_hz_to_midi() {
    // A4 = 440 Hz = MIDI note 69
    let midi = convert::hz_to_midi(&[440.0]);
    assert_eq!(midi[0], 69.0);
}

#[test]
fn convert_midi_to_hz() {
    let hz = convert::midi_to_hz(&[69.0]);
    assert!((hz[0] - 440.0).abs() < 0.1);
}

#[test]
fn convert_note_to_midi() {
    // A4 = 69
    let midi = convert::note_to_midi("A4").unwrap();
    assert_eq!(midi, 69);

    // C4 = 60 (middle C)
    let midi = convert::note_to_midi("C4").unwrap();
    assert_eq!(midi, 60);
}

#[test]
fn convert_midi_to_note() {
    let note = convert::midi_to_note(69);
    assert_eq!(note, "A4");

    let note = convert::midi_to_note(60);
    assert_eq!(note, "C4");
}

#[test]
fn convert_interval_to_fjs() {
    // Perfect fifth (7 semitones)
    let fjs = convert::interval_to_fjs(7.0, "1/1", true);
    assert!(!fjs.is_empty());
}

#[test]
fn convert_times_like() {
    let n_frames = 100;
    let sr = 22050;
    let hop_length = 512;

    let result = convert::times_like(n_frames, sr, hop_length);
    assert_eq!(result.len(), n_frames);
}
