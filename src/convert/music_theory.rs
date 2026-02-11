use super::{intervals::parse_note_to_pitch_class, note_to_midi};

// Music Theory: Thaat, Mela, Key, Svara

/// Thaat definitions (Hindustani music theory).
/// Each thaat is a 7-note scale defined by semitone offsets from Sa.
const THAAT_MAP: [(&str, [i32; 7]); 10] = [
    ("bilaval", [0, 2, 4, 5, 7, 9, 11]),  // Ionian/Major
    ("khamaj", [0, 2, 4, 5, 7, 9, 10]),   // Mixolydian
    ("kafi", [0, 2, 3, 5, 7, 9, 10]),     // Dorian
    ("asavari", [0, 2, 3, 5, 7, 8, 10]),  // Aeolian/Natural Minor
    ("bhairavi", [0, 1, 3, 5, 7, 8, 10]), // Phrygian
    ("kalyan", [0, 2, 4, 6, 7, 9, 11]),   // Lydian
    ("marva", [0, 1, 4, 6, 7, 9, 11]),    // Lydian b2
    ("poorvi", [0, 1, 4, 6, 7, 8, 11]),   //
    ("todi", [0, 1, 3, 6, 7, 8, 11]),     //
    ("bhairav", [0, 1, 4, 5, 7, 8, 11]),  // Double harmonic
];

/// Melakarta raga names (72 parent scales in Carnatic music).
const MELAKARTA_NAMES: [&str; 72] = [
    "kanakangi",
    "ratnangi",
    "ganamurthi",
    "vanaspathi",
    "manavathi",
    "tanarupi",
    "senavathi",
    "hanumathodi",
    "dhenuka",
    "natakapriya",
    "kokilapriya",
    "rupavathi",
    "gayakapriya",
    "vakulabharanam",
    "mayamalavagaula",
    "chakravakom",
    "suryakantham",
    "hatakambari",
    "jhankaradhwani",
    "natabhairavi",
    "keeravani",
    "kharaharapriya",
    "gaurimanohari",
    "varunapriya",
    "mararanjini",
    "charukesi",
    "sarasangi",
    "harikambhoji",
    "dheerasankarabharanam",
    "naganandini",
    "yagapriya",
    "ragavardhini",
    "gangeyabhushani",
    "vagadheeswari",
    "sulini",
    "chalanatta",
    "salagam",
    "jalarnavam",
    "jhalavarali",
    "navaneetham",
    "pavani",
    "raghupriya",
    "gavambodhi",
    "bhavapriya",
    "subhapanthuvarali",
    "shadvidhamargini",
    "suvarnangi",
    "divyamani",
    "dhavalambari",
    "namanarayani",
    "kamavardhini",
    "ramapriya",
    "gamanasrama",
    "viswambhari",
    "syamalangi",
    "shanmukhapriya",
    "simhendramadhyamam",
    "hemavathi",
    "dharmavathi",
    "neethimathi",
    "kanthamani",
    "rishabhapriya",
    "latangi",
    "vachaspathi",
    "mechakalyani",
    "chitrambari",
    "sucharitra",
    "jyotisvarupini",
    "dhatuvardhini",
    "nasikabhushani",
    "kosalam",
    "rasikapriya",
];

/// List all supported thaats.
///
/// # Returns
/// A list of thaat names
///
/// # Example
/// ```
/// use giggle::convert::list_thaat;
///
/// let thaats = list_thaat();
/// assert!(thaats.contains(&"bilaval".to_string()));
/// assert_eq!(thaats.len(), 10);
/// ```
pub fn list_thaat() -> Vec<String> {
    THAAT_MAP.iter().map(|(name, _)| name.to_string()).collect()
}

/// List all melakarta ragas by name and index.
///
/// # Returns
/// A vector of (name, index) pairs where index is 1-72
///
/// # Example
/// ```
/// use giggle::convert::list_mela;
///
/// let melas = list_mela();
/// assert_eq!(melas.len(), 72);
/// assert_eq!(melas[0], ("kanakangi".to_string(), 1));
/// ```
pub fn list_mela() -> Vec<(String, usize)> {
    MELAKARTA_NAMES
        .iter()
        .enumerate()
        .map(|(i, &name)| (name.to_string(), i + 1))
        .collect()
}

/// Construct the svara indices (degrees) for a given thaat.
///
/// # Arguments
/// * `thaat` - The name of the thaat (case-insensitive)
///
/// # Returns
/// A list of seven svara indices (0=Sa, 2=Re, 4=Ga, 5=Ma, 7=Pa, 9=Dha, 11=Ni)
///
/// # Example
/// ```
/// use giggle::convert::thaat_to_degrees;
///
/// let degrees = thaat_to_degrees("bilaval").unwrap();
/// assert_eq!(degrees, vec![0, 2, 4, 5, 7, 9, 11]);
///
/// let todi = thaat_to_degrees("todi").unwrap();
/// assert_eq!(todi, vec![0, 1, 3, 6, 7, 8, 11]);
/// ```
pub fn thaat_to_degrees(thaat: &str) -> Option<Vec<i32>> {
    let thaat_lower = thaat.to_lowercase();
    for (name, degrees) in &THAAT_MAP {
        if *name == thaat_lower {
            return Some(degrees.to_vec());
        }
    }
    None
}

/// Construct the svara indices (degrees) for a given melakarta raga.
///
/// # Arguments
/// * `mela` - Either the name (str) or index (1-72) of the melakarta raga
///
/// # Returns
/// A list of seven svara indices (semitones from Sa)
///
/// # Example
/// ```
/// use giggle::convert::{mela_to_degrees, mela_to_degrees_by_index};
///
/// // By index
/// let degrees = mela_to_degrees_by_index(1).unwrap();
/// assert_eq!(degrees, vec![0, 1, 2, 5, 7, 8, 9]);
///
/// // By name
/// let degrees = mela_to_degrees("kanakangi").unwrap();
/// assert_eq!(degrees, vec![0, 1, 2, 5, 7, 8, 9]);
/// ```
pub fn mela_to_degrees(mela: &str) -> Option<Vec<i32>> {
    let mela_lower = mela.to_lowercase();
    for (i, &name) in MELAKARTA_NAMES.iter().enumerate() {
        if name == mela_lower {
            return mela_to_degrees_by_index(i + 1);
        }
    }
    None
}

/// Construct the svara indices for a melakarta by index (1-72).
pub fn mela_to_degrees_by_index(mela: usize) -> Option<Vec<i32>> {
    if mela == 0 || mela > 72 {
        return None;
    }

    let index = mela - 1;
    let mut degrees = vec![0]; // Sa is always 0

    // Determine Ri and Ga based on chakra (lower 36 vs upper 36)
    let lower = index % 36;
    let (ri, ga) = match lower {
        0..=5 => (1, 2),   // Ri1, Ga1
        6..=11 => (1, 3),  // Ri1, Ga2
        12..=17 => (1, 4), // Ri1, Ga3
        18..=23 => (2, 3), // Ri2, Ga2
        24..=29 => (2, 4), // Ri2, Ga3
        _ => (3, 4),       // Ri3, Ga3
    };
    degrees.push(ri);
    degrees.push(ga);

    // Determine Ma (first 36 ragas use Ma1=5, rest use Ma2=6)
    if index < 36 {
        degrees.push(5); // Ma1
    } else {
        degrees.push(6); // Ma2
    }

    // Pa is always 7
    degrees.push(7);

    // Determine Dha and Ni based on position within group of 6
    let upper = index % 6;
    let (dha, ni) = match upper {
        0 => (8, 9),   // Dha1, Ni1
        1 => (8, 10),  // Dha1, Ni2
        2 => (8, 11),  // Dha1, Ni3
        3 => (9, 10),  // Dha2, Ni2
        4 => (9, 11),  // Dha2, Ni3
        _ => (10, 11), // Dha3, Ni3
    };
    degrees.push(dha);
    degrees.push(ni);

    Some(degrees)
}

/// Spell the Carnatic svara names for a given melakarta raga.
///
/// # Arguments
/// * `mela` - The melakarta index (1-72) or name
/// * `abbr` - If true, use single-letter svara names (S, R, G...)
/// * `unicode` - If true, use unicode subscripts for numbering
///
/// # Returns
/// A list of 12 svara names for each pitch class
///
/// # Example
/// ```
/// use giggle::convert::mela_to_svara;
///
/// let svara = mela_to_svara(1, true, true);
/// assert_eq!(svara[0], "S");
/// assert_eq!(svara[7], "P");
/// ```
pub fn mela_to_svara(mela: usize, abbr: bool, unicode: bool) -> Vec<String> {
    if mela == 0 || mela > 72 {
        return vec!["?".to_string(); 12];
    }

    let mela_idx = mela - 1;

    // Base svara map (some entries will be determined by the raga)
    let mut svara_map: Vec<String> = if abbr {
        vec![
            "S".to_string(),  // 0: Sa
            "R₁".to_string(), // 1: Ri1
            "".to_string(),   // 2: Ri2/Ga1
            "".to_string(),   // 3: Ri3/Ga2
            "G₃".to_string(), // 4: Ga3
            "M₁".to_string(), // 5: Ma1
            "M₂".to_string(), // 6: Ma2
            "P".to_string(),  // 7: Pa
            "D₁".to_string(), // 8: Dha1
            "".to_string(),   // 9: Dha2/Ni1
            "".to_string(),   // 10: Dha3/Ni2
            "N₃".to_string(), // 11: Ni3
        ]
    } else {
        vec![
            "Sa".to_string(),
            "Ri₁".to_string(),
            "".to_string(),
            "".to_string(),
            "Ga₃".to_string(),
            "Ma₁".to_string(),
            "Ma₂".to_string(),
            "Pa".to_string(),
            "Dha₁".to_string(),
            "".to_string(),
            "".to_string(),
            "Ni₃".to_string(),
        ]
    };

    // Determine Ri2/Ga1 (index 2)
    let lower = mela_idx % 36;
    svara_map[2] = if lower < 6 {
        if abbr {
            "G₁".to_string()
        } else {
            "Ga₁".to_string()
        }
    } else if abbr {
        "R₂".to_string()
    } else {
        "Ri₂".to_string()
    };

    // Determine Ri3/Ga2 (index 3)
    svara_map[3] = if lower < 30 {
        if abbr {
            "G₂".to_string()
        } else {
            "Ga₂".to_string()
        }
    } else if abbr {
        "R₃".to_string()
    } else {
        "Ri₃".to_string()
    };

    // Determine Dha2/Ni1 (index 9)
    let upper = mela_idx % 6;
    svara_map[9] = if upper == 0 {
        if abbr {
            "N₁".to_string()
        } else {
            "Ni₁".to_string()
        }
    } else if abbr {
        "D₂".to_string()
    } else {
        "Dha₂".to_string()
    };

    // Determine Dha3/Ni2 (index 10)
    svara_map[10] = if upper == 5 {
        if abbr {
            "D₃".to_string()
        } else {
            "Dha₃".to_string()
        }
    } else if abbr {
        "N₂".to_string()
    } else {
        "Ni₂".to_string()
    };

    // Convert to ASCII if not unicode
    if !unicode {
        svara_map = svara_map
            .iter()
            .map(|s| s.replace('₁', "1").replace('₂', "2").replace('₃', "3"))
            .collect();
    }

    svara_map
}

/// Construct the diatonic scale degrees for a given key.
///
/// # Arguments
/// * `key` - Key signature in format "TONIC:SCALE" (e.g., "C:maj", "A:min", "D:dor")
///
/// # Returns
/// An array of 7 semitone numbers (0-11) for the scale degrees
///
/// # Example
/// ```
/// use giggle::convert::key_to_degrees;
///
/// let c_major = key_to_degrees("C:maj").unwrap();
/// assert_eq!(c_major, vec![0, 2, 4, 5, 7, 9, 11]);
///
/// let a_minor = key_to_degrees("A:min").unwrap();
/// assert_eq!(a_minor, vec![9, 11, 0, 2, 4, 5, 7]);
///
/// let d_dorian = key_to_degrees("D:dor").unwrap();
/// assert_eq!(d_dorian, vec![2, 4, 5, 7, 9, 11, 0]);
/// ```
pub fn key_to_degrees(key: &str) -> Option<Vec<i32>> {
    let parts: Vec<&str> = key.split(':').collect();
    if parts.len() != 2 {
        return None;
    }

    let tonic_str = parts[0];
    let scale_str = parts[1].to_lowercase();

    // Parse tonic
    let tonic = parse_note_to_pitch_class(tonic_str)?;

    // Define scale patterns
    let pattern = match scale_str.as_str() {
        "maj" | "major" | "ion" | "ionian" => [0, 2, 4, 5, 7, 9, 11],
        "min" | "minor" | "aeo" | "aeolian" => [0, 2, 3, 5, 7, 8, 10],
        "dor" | "dorian" => [0, 2, 3, 5, 7, 9, 10],
        "phr" | "phryg" | "phrygian" => [0, 1, 3, 5, 7, 8, 10],
        "lyd" | "lydian" => [0, 2, 4, 6, 7, 9, 11],
        "mix" | "mixolyd" | "mixolydian" => [0, 2, 4, 5, 7, 9, 10],
        "loc" | "locr" | "locrian" => [0, 1, 3, 5, 6, 8, 10],
        _ => return None,
    };

    Some(pattern.iter().map(|&d| (d + tonic) % 12).collect())
}

/// List all 12 note names as spelled in a given key.
///
/// # Arguments
/// * `key` - Key signature in format "TONIC:SCALE"
/// * `unicode` - If true, use unicode symbols (♯, ♭)
///
/// # Returns
/// A list of 12 note names starting from C
///
/// # Example
/// ```
/// use giggle::convert::key_to_notes;
///
/// let c_major = key_to_notes("C:maj", true);
/// assert_eq!(c_major[0], "C");
/// assert_eq!(c_major[1], "C♯");
///
/// let f_major = key_to_notes("F:maj", true);
/// assert_eq!(f_major[10], "B♭");
/// ```
pub fn key_to_notes(key: &str, unicode: bool) -> Vec<String> {
    let parts: Vec<&str> = key.split(':').collect();
    if parts.len() != 2 {
        return vec!["?".to_string(); 12];
    }

    let tonic_str = parts[0];
    let scale_str = parts[1].to_lowercase();

    // Parse tonic to determine if we use sharps or flats
    let tonic = match parse_note_to_pitch_class(tonic_str) {
        Some(t) => t,
        None => return vec!["?".to_string(); 12],
    };

    // Check if tonic has a flat or sharp
    let use_flats = tonic_str.contains('b') || tonic_str.contains('♭');
    let use_sharps = tonic_str.contains('#') || tonic_str.contains('♯');

    // For keys without explicit accidentals, determine based on key signature
    let use_sharps = if !use_flats && !use_sharps {
        // Calculate position on circle of fifths
        let major = scale_str == "maj"
            || scale_str == "major"
            || scale_str == "ion"
            || scale_str == "ionian";
        let cof_pos = if major {
            (tonic * 7) % 12
        } else {
            ((tonic + 3) * 7) % 12
        };
        cof_pos < 6 || cof_pos == 0
    } else {
        use_sharps
    };

    let (sharp_sym, flat_sym) = if unicode { ("♯", "♭") } else { ("#", "b") };

    if use_sharps {
        vec![
            "C".to_string(),
            format!("C{}", sharp_sym),
            "D".to_string(),
            format!("D{}", sharp_sym),
            "E".to_string(),
            "F".to_string(),
            format!("F{}", sharp_sym),
            "G".to_string(),
            format!("G{}", sharp_sym),
            "A".to_string(),
            format!("A{}", sharp_sym),
            "B".to_string(),
        ]
    } else {
        vec![
            "C".to_string(),
            format!("D{}", flat_sym),
            "D".to_string(),
            format!("E{}", flat_sym),
            "E".to_string(),
            "F".to_string(),
            format!("G{}", flat_sym),
            "G".to_string(),
            format!("A{}", flat_sym),
            "A".to_string(),
            format!("B{}", flat_sym),
            "B".to_string(),
        ]
    }
}

/// Calculate the note name for a given number of perfect fifths from a unison.
///
/// # Arguments
/// * `unison` - The starting note name (e.g., "C", "Bb")
/// * `fifths` - Number of perfect fifths (positive = up, negative = down)
/// * `unicode` - If true, use unicode accidentals
///
/// # Returns
/// The note name at the specified interval
///
/// # Example
/// ```
/// use giggle::convert::fifths_to_note;
///
/// assert_eq!(fifths_to_note("C", 1, true), "G");
/// assert_eq!(fifths_to_note("C", 6, true), "F♯");
/// assert_eq!(fifths_to_note("G", -3, true), "B♭");
/// ```
pub fn fifths_to_note(unison: &str, fifths: i32, unicode: bool) -> String {
    const COF: [char; 7] = ['F', 'C', 'G', 'D', 'A', 'E', 'B'];

    // Parse the unison note
    let note_char = unison.chars().next().unwrap_or('C').to_ascii_uppercase();
    let accidental_offset: i32 = unison[1..]
        .chars()
        .map(|c| match c {
            '#' | '♯' => 1,
            'b' | '♭' | '!' => -1,
            _ => 0,
        })
        .sum();

    // Find position in circle of fifths
    let base_pos = COF.iter().position(|&c| c == note_char).unwrap_or(1) as i32;

    // Calculate new position
    let new_pos = base_pos + fifths;
    let note_idx = ((new_pos % 7) + 7) % 7;
    let new_note = COF[note_idx as usize];

    // Calculate accidentals (crossing B-F boundary adds sharps/flats)
    let acc_count = accidental_offset + (new_pos / 7)
        - if new_pos < 0 && new_pos % 7 != 0 {
            1
        } else {
            0
        };

    let (sharp_sym, flat_sym, double_sharp, double_flat) = if unicode {
        ("♯", "♭", "𝄪", "𝄫")
    } else {
        ("#", "b", "##", "bb")
    };

    let acc_str = if acc_count >= 0 {
        let doubles = acc_count / 2;
        let singles = acc_count % 2;
        format!(
            "{}{}",
            double_sharp.repeat(doubles as usize),
            sharp_sym.repeat(singles as usize)
        )
    } else {
        let doubles = (-acc_count) / 2;
        let singles = (-acc_count) % 2;
        format!(
            "{}{}",
            double_flat.repeat(doubles as usize),
            flat_sym.repeat(singles as usize)
        )
    };

    format!("{}{}", new_note, acc_str)
}

/// Convert MIDI note to Hindustani svara name.
///
/// # Arguments
/// * `midi` - MIDI note number
/// * `sa` - MIDI note number of Sa (tonic)
/// * `abbr` - If true, use abbreviated names
/// * `unicode` - If true, use unicode for octave markers
///
/// # Returns
/// Svara name with optional octave indicator
///
/// # Example
/// ```
/// use giggle::convert::midi_to_svara_h;
///
/// // Sa = C4 (MIDI 60)
/// assert_eq!(midi_to_svara_h(60, 60, true, true), "S");
/// assert_eq!(midi_to_svara_h(62, 60, true, true), "R");
/// // Upper octave with ASCII markers
/// assert_eq!(midi_to_svara_h(72, 60, true, false), "S^1");
/// ```
pub fn midi_to_svara_h(midi: i32, sa: i32, abbr: bool, unicode: bool) -> String {
    let svara_names_abbr = ["S", "r", "R", "g", "G", "m", "M", "P", "d", "D", "n", "N"];
    let svara_names_full = [
        "Sa", "re", "Re", "ga", "Ga", "ma", "Ma", "Pa", "dha", "Dha", "ni", "Ni",
    ];

    let names = if abbr {
        &svara_names_abbr
    } else {
        &svara_names_full
    };

    let interval = midi - sa;
    let octave = interval.div_euclid(12);
    let degree = interval.rem_euclid(12) as usize;

    let base_name = names[degree].to_string();

    // Add octave markers
    if unicode {
        match octave {
            o if o < 0 => format!("{}{}", base_name, "̣".repeat((-o) as usize)), // Underdot
            0 => base_name,
            o => format!("{}{}", base_name, "̇".repeat(o as usize)), // Overdot
        }
    } else {
        match octave {
            o if o < 0 => format!("{}_{}", base_name, -o),
            0 => base_name,
            o => format!("{}^{}", base_name, o),
        }
    }
}

/// Convert frequency (Hz) to Hindustani svara name.
///
/// # Arguments
/// * `freq` - Frequency in Hz
/// * `sa_freq` - Frequency of Sa (tonic) in Hz
/// * `abbr` - If true, use abbreviated names
/// * `unicode` - If true, use unicode octave markers
///
/// # Example
/// ```
/// use giggle::convert::hz_to_svara_h;
///
/// // Sa = 261.63 Hz (C4)
/// assert_eq!(hz_to_svara_h(261.63, 261.63, true, false), "S");
/// ```
pub fn hz_to_svara_h(freq: f32, sa_freq: f32, abbr: bool, unicode: bool) -> String {
    if freq <= 0.0 || sa_freq <= 0.0 {
        return "?".to_string();
    }

    let midi = (12.0 * (freq / 440.0).log2() + 69.0).round() as i32;
    let sa_midi = (12.0 * (sa_freq / 440.0).log2() + 69.0).round() as i32;

    midi_to_svara_h(midi, sa_midi, abbr, unicode)
}

/// Convert MIDI note to Carnatic svara within a given melakarta raga.
///
/// # Arguments
/// * `midi` - MIDI note number
/// * `sa` - MIDI note number of Sa (tonic), default 60 (C4)
/// * `mela` - Melakarta raga index (1-72) or name
/// * `abbr` - If true, use abbreviated names (S, R1, G1...)
/// * `octave` - If true, decorate svara in neighboring octaves
/// * `unicode` - If true, use unicode subscripts
///
/// # Returns
/// Svara name for the given MIDI note
///
/// # Example
/// ```
/// use giggle::convert::midi_to_svara_c;
///
/// // Sa = C4 (MIDI 60), using melakarta #1 (Kanakangi)
/// assert_eq!(midi_to_svara_c(60, 60, 1, true, false, true), "S");
/// assert_eq!(midi_to_svara_c(61, 60, 1, true, false, true), "R₁");
/// assert_eq!(midi_to_svara_c(67, 60, 1, true, false, true), "P");
/// ```
pub fn midi_to_svara_c(
    midi: i32,
    sa: i32,
    mela: usize,
    abbr: bool,
    octave: bool,
    unicode: bool,
) -> String {
    let svara_num = midi - sa;

    // Get the svara map for this melakarta
    let svara_map = mela_to_svara(mela, abbr, unicode);
    if svara_map[0] == "?" {
        return "?".to_string();
    }

    // Get the svara name (modulo 12)
    let svara_idx = ((svara_num % 12) + 12) % 12;
    let mut svara = svara_map[svara_idx as usize].clone();

    // Add octave decoration if requested
    if octave {
        if (12..24).contains(&svara_num) {
            // Upper octave
            if unicode {
                // Add overdot after first character
                let first_char: String = svara.chars().take(1).collect();
                let rest: String = svara.chars().skip(1).collect();
                svara = format!("{}\u{0307}{}", first_char, rest);
            } else {
                svara.push('\'');
            }
        } else if (-12..0).contains(&svara_num) {
            // Lower octave
            if unicode {
                // Add underdot after first character
                let first_char: String = svara.chars().take(1).collect();
                let rest: String = svara.chars().skip(1).collect();
                svara = format!("{}\u{0323}{}", first_char, rest);
            } else {
                svara.push(',');
            }
        }
    }

    svara
}

/// Convert frequency (Hz) to Carnatic svara within a given melakarta raga.
///
/// Note: This conversion assumes 12-tone equal temperament.
///
/// # Arguments
/// * `freq` - Frequency in Hz
/// * `sa_freq` - Frequency of Sa (tonic) in Hz
/// * `mela` - Melakarta raga index (1-72) or name
/// * `abbr` - If true, use abbreviated names
/// * `octave` - If true, decorate svara in neighboring octaves
/// * `unicode` - If true, use unicode subscripts
///
/// # Returns
/// Svara name for the given frequency
///
/// # Example
/// ```
/// use giggle::convert::hz_to_svara_c;
///
/// // Sa = 261 Hz, using melakarta #36
/// let svara = hz_to_svara_c(261.0, 261.0, 36, true, false, true);
/// assert_eq!(svara, "S");
/// ```
pub fn hz_to_svara_c(
    freq: f32,
    sa_freq: f32,
    mela: usize,
    abbr: bool,
    octave: bool,
    unicode: bool,
) -> String {
    if freq <= 0.0 || sa_freq <= 0.0 {
        return "?".to_string();
    }

    let midi = (12.0 * (freq / 440.0).log2() + 69.0).round() as i32;
    let sa_midi = (12.0 * (sa_freq / 440.0).log2() + 69.0).round() as i32;

    midi_to_svara_c(midi, sa_midi, mela, abbr, octave, unicode)
}

/// Convert western note name to Carnatic svara within a given melakarta raga.
///
/// Note: This conversion assumes 12-tone equal temperament.
///
/// # Arguments
/// * `note` - Western note name (e.g., "C4", "D#5")
/// * `sa` - Note name for Sa (e.g., "C4")
/// * `mela` - Melakarta raga index (1-72) or name
/// * `abbr` - If true, use abbreviated names
/// * `octave` - If true, decorate svara in neighboring octaves
/// * `unicode` - If true, use unicode subscripts
///
/// # Returns
/// Svara name for the given note
///
/// # Example
/// ```
/// use giggle::convert::note_to_svara_c;
///
/// // Sa = C4, using melakarta #1 (Kanakangi)
/// assert_eq!(note_to_svara_c("C4", "C4", 1, true, false, true), "S");
/// assert_eq!(note_to_svara_c("G4", "C4", 1, true, false, true), "P");
/// // Upper octave has overdot (combining character)
/// let upper = note_to_svara_c("C5", "C4", 1, true, true, true);
/// assert!(upper.starts_with("S"));
/// assert!(upper.contains('\u{0307}')); // Combining dot above
/// ```
pub fn note_to_svara_c(
    note: &str,
    sa: &str,
    mela: usize,
    abbr: bool,
    octave: bool,
    unicode: bool,
) -> String {
    let midi = match note_to_midi(note) {
        Some(m) => m,
        None => return "?".to_string(),
    };

    let sa_midi = match note_to_midi(sa) {
        Some(m) => m,
        None => return "?".to_string(),
    };

    midi_to_svara_c(midi, sa_midi, mela, abbr, octave, unicode)
}

/// Convert western note name to Hindustani svara.
///
/// # Arguments
/// * `note` - Western note name (e.g., "C4", "D#5")
/// * `sa` - Note name for Sa (e.g., "C4")
/// * `abbr` - If true, use abbreviated names
/// * `unicode` - If true, use unicode octave markers
///
/// # Returns
/// Svara name for the given note
///
/// # Example
/// ```
/// use giggle::convert::note_to_svara_h;
///
/// // Sa = C4
/// assert_eq!(note_to_svara_h("C4", "C4", true, false), "S");
/// assert_eq!(note_to_svara_h("G4", "C4", true, false), "P");
/// assert_eq!(note_to_svara_h("C5", "C4", true, false), "S^1");
/// ```
pub fn note_to_svara_h(note: &str, sa: &str, abbr: bool, unicode: bool) -> String {
    let midi = match note_to_midi(note) {
        Some(m) => m,
        None => return "?".to_string(),
    };

    let sa_midi = match note_to_midi(sa) {
        Some(m) => m,
        None => return "?".to_string(),
    };

    midi_to_svara_h(midi, sa_midi, abbr, unicode)
}
