use super::{A4_HZ, MIDI_A4};

/// Convert frequency (Hz) to MIDI note number.
pub fn hz_to_midi(frequencies: &[f32]) -> Vec<f32> {
    frequencies
        .iter()
        .map(|&f| {
            if f > 0.0 {
                12.0 * (f / A4_HZ).log2() + MIDI_A4
            } else {
                0.0
            }
        })
        .collect()
}

/// Convert MIDI note number to frequency (Hz).
pub fn midi_to_hz(notes: &[f32]) -> Vec<f32> {
    notes
        .iter()
        .map(|&n| A4_HZ * 2.0f32.powf((n - MIDI_A4) / 12.0))
        .collect()
}

/// Convert frequency (Hz) to note name.
pub fn hz_to_note(frequency: f32) -> String {
    if frequency <= 0.0 {
        return "N/A".to_string();
    }
    let midi = 12.0 * (frequency / A4_HZ).log2() + MIDI_A4;
    midi_to_note(midi.round() as i32)
}

/// Convert MIDI note number to note name.
pub fn midi_to_note(midi: i32) -> String {
    const NOTES: [&str; 12] = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];
    let octave = (midi / 12) - 1;
    let note = NOTES[(midi % 12) as usize];
    format!("{}{}", note, octave)
}

/// Convert note name to MIDI number.
pub fn note_to_midi(note: &str) -> Option<i32> {
    const NOTE_MAP: [(&str, i32); 12] = [
        ("C", 0),
        ("C#", 1),
        ("D", 2),
        ("D#", 3),
        ("E", 4),
        ("F", 5),
        ("F#", 6),
        ("G", 7),
        ("G#", 8),
        ("A", 9),
        ("A#", 10),
        ("B", 11),
    ];

    let note = note.trim().to_uppercase();
    for i in 1..note.len() {
        let (pitch, octave_str) = note.split_at(i);
        if let Ok(octave) = octave_str.parse::<i32>() {
            for &(name, offset) in &NOTE_MAP {
                if pitch == name {
                    return Some((octave + 1) * 12 + offset);
                }
            }
        }
    }
    None
}

/// Convert note name to frequency (Hz).
pub fn note_to_hz(note: &str) -> Option<f32> {
    note_to_midi(note).map(|midi| A4_HZ * 2.0f32.powf((midi as f32 - MIDI_A4) / 12.0))
}
