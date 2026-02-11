use giggle::{
    feature::{self, chroma},
    io, spectrum, utils, window,
};
use ndarray::{Array2, ArrayD};
use ndarray_npy::read_npy;
use std::path::Path;

fn fixture_path(name: &str) -> Option<std::path::PathBuf> {
    let base = Path::new("tests/data/fixtures");
    let path = base.join(name);
    if path.exists() { Some(path) } else { None }
}

fn rel_mse(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut acc = 0.0f32;
    let mut ref_acc = 0.0f32;
    for i in 0..n {
        let d = a[i] - b[i];
        acc += d * d;
        ref_acc += b[i] * b[i];
    }
    acc / ref_acc.max(1e-12)
}

const MEL_REL_MSE_MAX: f32 = 0.12;
const MFCC_REL_MSE_MAX: f32 = 0.05;
const STFT_REL_MSE_MAX: f32 = 0.05;
const ZCR_REL_MSE_MAX: f32 = 0.02;
const RMS_REL_MSE_MAX: f32 = 0.02;
const CONTRAST_REL_MSE_MAX: f32 = 0.05;
const CHROMA_REL_MSE_MAX: f32 = 0.15;
const SPECTRAL_REL_MSE_MAX: f32 = 0.10;
const DB_REL_MSE_MAX: f32 = 0.05;
const SIGNAL_GEN_REL_MSE_MAX: f32 = 0.05;
const TEMPOGRAM_REL_MSE_MAX: f32 = 0.15;
const AUTOCORR_REL_MSE_MAX: f32 = 0.05;
const NORMALIZE_REL_MSE_MAX: f32 = 0.01;

fn load_vec(name: &str) -> Option<Vec<f32>> {
    let path = fixture_path(name)?;
    let arr: ArrayD<f32> = read_npy(path).ok()?;
    Some(arr.iter().copied().collect())
}

fn load_array2(name: &str) -> Option<Array2<f32>> {
    let path = fixture_path(name)?;
    read_npy(path).ok()
}

#[test]
fn align_mel_mfcc() {
    let sine = match load_vec("sine.npy") {
        Some(v) => v,
        None => return,
    };
    let mel_ref = match load_array2("mel.npy") {
        Some(v) => v,
        None => return,
    };
    let mfcc_ref = match load_array2("mfcc.npy") {
        Some(v) => v,
        None => return,
    };

    let mel = feature::mel::melspectrogram(&sine, 22050, 512, 128, 40).unwrap();
    let mfcc = feature::mfcc::mfcc(&sine, 22050, 13, 512, 128, 40).unwrap();

    let mel_mse = rel_mse(mel.as_slice().unwrap(), mel_ref.as_slice().unwrap());
    let mfcc_mse = rel_mse(mfcc.as_slice().unwrap(), mfcc_ref.as_slice().unwrap());

    println!(
        "MEL relative MSE: {:.6} (threshold: {:.2})",
        mel_mse, MEL_REL_MSE_MAX
    );
    println!(
        "MFCC relative MSE: {:.6} (threshold: {:.2})",
        mfcc_mse, MFCC_REL_MSE_MAX
    );

    assert!(mel_mse < MEL_REL_MSE_MAX, "mel rel mse too high: {mel_mse}");
    assert!(
        mfcc_mse < MFCC_REL_MSE_MAX,
        "mfcc rel mse too high: {mfcc_mse}"
    );
}

#[test]
fn align_stft() {
    let sine = match load_vec("sine.npy") {
        Some(v) => v,
        None => return,
    };
    let stft_r = match load_array2("stft_real.npy") {
        Some(v) => v,
        None => return,
    };
    let stft_i = match load_array2("stft_imag.npy") {
        Some(v) => v,
        None => return,
    };

    let mut cfg = spectrum::StftConfig::default();
    cfg.n_fft = 512;
    cfg.win_length = 512;
    cfg.hop_length = 128;
    cfg.window = window::hann(cfg.win_length);

    let stft = spectrum::stft(&sine, &cfg).unwrap();
    let mut flat = Vec::with_capacity(stft.len() * 2);
    for v in stft.iter() {
        flat.push(v.re);
        flat.push(v.im);
    }
    let mut flat_ref = Vec::with_capacity(stft_r.len() * 2);
    for (r, i) in stft_r.iter().zip(stft_i.iter()) {
        flat_ref.push(*r);
        flat_ref.push(*i);
    }

    let stft_mse = rel_mse(&flat, &flat_ref);
    println!(
        "STFT relative MSE: {:.6} (threshold: {:.2})",
        stft_mse, STFT_REL_MSE_MAX
    );
    assert!(
        stft_mse < STFT_REL_MSE_MAX,
        "stft rel mse too high: {stft_mse}"
    );
}

#[test]
fn align_basic_features() {
    let sine = match load_vec("sine.npy") {
        Some(v) => v,
        None => return,
    };
    let zcr_ref = match load_vec("zcr.npy") {
        Some(v) => v,
        None => return,
    };
    let rms_ref = match load_vec("rms.npy") {
        Some(v) => v,
        None => return,
    };

    let zcr = feature::basic::zero_crossing_rate_frames(&sine, 2048, 512, true).unwrap();
    let rms = feature::basic::rms_frames(&sine, 2048, 512, true).unwrap();

    let zcr_mse = rel_mse(&zcr, &zcr_ref);
    let rms_mse = rel_mse(&rms, &rms_ref);

    println!(
        "ZCR relative MSE: {:.6} (threshold: {:.2})",
        zcr_mse, ZCR_REL_MSE_MAX
    );
    println!(
        "RMS relative MSE: {:.6} (threshold: {:.2})",
        rms_mse, RMS_REL_MSE_MAX
    );

    assert!(zcr_mse < ZCR_REL_MSE_MAX, "zcr rel mse too high: {zcr_mse}");
    assert!(rms_mse < RMS_REL_MSE_MAX, "rms rel mse too high: {rms_mse}");
}

#[test]
fn align_contrast() {
    let sine = match load_vec("sine.npy") {
        Some(v) => v,
        None => return,
    };
    let contrast_ref = match load_array2("contrast.npy") {
        Some(v) => v,
        None => return,
    };

    let contrast = feature::contrast::spectral_contrast(&sine, 22050, 512, 128, 6, 200.0).unwrap();
    let c_mse = rel_mse(
        contrast.as_slice().unwrap(),
        contrast_ref.as_slice().unwrap(),
    );
    println!(
        "CONTRAST relative MSE: {:.6} (threshold: {:.2})",
        c_mse, CONTRAST_REL_MSE_MAX
    );
    assert!(
        c_mse < CONTRAST_REL_MSE_MAX,
        "contrast rel mse too high: {c_mse}"
    );
}

#[test]
fn align_chroma() {
    let sine = match load_vec("sine.npy") {
        Some(v) => v,
        None => return,
    };
    let chroma_ref = match load_array2("chroma.npy") {
        Some(v) => v,
        None => return,
    };

    let chroma_out = chroma::chroma_stft(&sine, 22050, 512, 128, 12, 0.0).unwrap();

    // Convert both to flat vectors for comparison (handle different memory layouts)
    let out_flat: Vec<f32> = chroma_out.iter().copied().collect();
    let ref_flat: Vec<f32> = chroma_ref.iter().copied().collect();

    let c_mse = rel_mse(&out_flat, &ref_flat);
    println!(
        "CHROMA relative MSE: {:.6} (threshold: {:.2})",
        c_mse, CHROMA_REL_MSE_MAX
    );
    assert!(
        c_mse < CHROMA_REL_MSE_MAX,
        "chroma rel mse too high: {c_mse}"
    );
}

#[test]
fn align_spectral_features() {
    let sine = match load_vec("sine.npy") {
        Some(v) => v,
        None => return,
    };
    let centroid_ref = match load_vec("spectral_centroid.npy") {
        Some(v) => v,
        None => return,
    };
    let bandwidth_ref = match load_vec("spectral_bandwidth.npy") {
        Some(v) => v,
        None => return,
    };
    let rolloff_ref = match load_vec("spectral_rolloff.npy") {
        Some(v) => v,
        None => return,
    };
    let flatness_ref = match load_vec("spectral_flatness.npy") {
        Some(v) => v,
        None => return,
    };

    let mut cfg = spectrum::StftConfig::default();
    cfg.n_fft = 512;
    cfg.win_length = 512;
    cfg.hop_length = 128;
    cfg.window = window::hann(cfg.win_length);

    let stft = spectrum::stft(&sine, &cfg).unwrap();
    let (mag, _) = spectrum::magphase(&stft);

    let sr = 22050;
    let freq_bins: Vec<f32> = (0..=256).map(|i| i as f32 * sr as f32 / 512.0).collect();

    let centroid = feature::basic::spectral_centroid(&mag, &freq_bins).unwrap();
    let bandwidth = feature::basic::spectral_bandwidth(&mag, &freq_bins).unwrap();
    let rolloff = feature::basic::spectral_rolloff(&mag, &freq_bins, 0.85).unwrap();
    let flatness = feature::basic::spectral_flatness(&mag).unwrap();

    let c_mse = rel_mse(&centroid, &centroid_ref);
    let b_mse = rel_mse(&bandwidth, &bandwidth_ref);
    let r_mse = rel_mse(&rolloff, &rolloff_ref);
    let f_mse = rel_mse(&flatness, &flatness_ref);

    if c_mse >= SPECTRAL_REL_MSE_MAX {
        eprintln!("centroid first 5: {:?}", &centroid[..5.min(centroid.len())]);
        eprintln!(
            "centroid_ref first 5: {:?}",
            &centroid_ref[..5.min(centroid_ref.len())]
        );
    }
    if f_mse >= SPECTRAL_REL_MSE_MAX {
        eprintln!(
            "flatness first 10: {:?}",
            &flatness[..10.min(flatness.len())]
        );
        eprintln!(
            "flatness_ref first 10: {:?}",
            &flatness_ref[..10.min(flatness_ref.len())]
        );
        eprintln!(
            "flatness range: [{}, {}]",
            flatness.iter().copied().fold(f32::INFINITY, f32::min),
            flatness.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        );
    }

    println!(
        "SPECTRAL_CENTROID relative MSE: {:.6} (threshold: {:.2})",
        c_mse, SPECTRAL_REL_MSE_MAX
    );
    println!(
        "SPECTRAL_BANDWIDTH relative MSE: {:.6} (threshold: {:.2})",
        b_mse, SPECTRAL_REL_MSE_MAX
    );
    println!(
        "SPECTRAL_ROLLOFF relative MSE: {:.6} (threshold: {:.2})",
        r_mse, SPECTRAL_REL_MSE_MAX
    );
    println!(
        "SPECTRAL_FLATNESS relative MSE: {:.6} (threshold: {:.2})",
        f_mse, SPECTRAL_REL_MSE_MAX
    );

    assert!(
        c_mse < SPECTRAL_REL_MSE_MAX,
        "spectral_centroid rel mse too high: {c_mse}"
    );
    assert!(
        b_mse < SPECTRAL_REL_MSE_MAX,
        "spectral_bandwidth rel mse too high: {b_mse}"
    );
    assert!(
        r_mse < SPECTRAL_REL_MSE_MAX,
        "spectral_rolloff rel mse too high: {r_mse}"
    );
    assert!(
        f_mse < SPECTRAL_REL_MSE_MAX,
        "spectral_flatness rel mse too high: {f_mse}"
    );
}

#[test]
fn align_db_conversions() {
    let mel_ref = match load_array2("mel.npy") {
        Some(v) => v,
        None => return,
    };
    let db_power_ref = match load_array2("power_to_db.npy") {
        Some(v) => v,
        None => return,
    };
    let db_amp_ref = match load_array2("amplitude_to_db.npy") {
        Some(v) => v,
        None => return,
    };

    let sine = match load_vec("sine.npy") {
        Some(v) => v,
        None => return,
    };

    let mut cfg = spectrum::StftConfig::default();
    cfg.n_fft = 512;
    cfg.win_length = 512;
    cfg.hop_length = 128;
    cfg.window = window::hann(cfg.win_length);

    let stft = spectrum::stft(&sine, &cfg).unwrap();
    let (mag, _) = spectrum::magphase(&stft);

    let db_power = spectrum::power_to_db(&mel_ref, 1.0, 1e-10, Some(80.0));
    let db_amp = spectrum::amplitude_to_db(&mag, 1.0, 1e-10, Some(80.0));

    let db_power_flat: Vec<f32> = db_power.iter().copied().collect();
    let db_power_ref_flat: Vec<f32> = db_power_ref.iter().copied().collect();
    let db_amp_flat: Vec<f32> = db_amp.iter().copied().collect();
    let db_amp_ref_flat: Vec<f32> = db_amp_ref.iter().copied().collect();

    let p_mse = rel_mse(&db_power_flat, &db_power_ref_flat);
    let a_mse = rel_mse(&db_amp_flat, &db_amp_ref_flat);

    println!(
        "POWER_TO_DB relative MSE: {:.6} (threshold: {:.2})",
        p_mse, DB_REL_MSE_MAX
    );
    println!(
        "AMPLITUDE_TO_DB relative MSE: {:.6} (threshold: {:.2})",
        a_mse, DB_REL_MSE_MAX
    );

    assert!(
        p_mse < DB_REL_MSE_MAX,
        "power_to_db rel mse too high: {p_mse}"
    );
    assert!(
        a_mse < DB_REL_MSE_MAX,
        "amplitude_to_db rel mse too high: {a_mse}"
    );
}

#[test]
fn align_signal_generators() {
    let tone_ref = match load_vec("tone.npy") {
        Some(v) => v,
        None => return,
    };
    let chirp_ref = match load_vec("chirp.npy") {
        Some(v) => v,
        None => return,
    };

    let sr = 22050;
    let tone = io::tone(440.0, sr, 0.5);
    let chirp = io::chirp(100.0, 1000.0, sr, 0.5);

    let t_mse = rel_mse(&tone, &tone_ref);
    let c_mse = rel_mse(&chirp, &chirp_ref);

    println!(
        "TONE relative MSE: {:.6} (threshold: {:.2})",
        t_mse, SIGNAL_GEN_REL_MSE_MAX
    );
    println!(
        "CHIRP relative MSE: {:.6} (threshold: {:.2})",
        c_mse, SIGNAL_GEN_REL_MSE_MAX
    );

    assert!(
        t_mse < SIGNAL_GEN_REL_MSE_MAX,
        "tone rel mse too high: {t_mse}"
    );
    assert!(
        c_mse < SIGNAL_GEN_REL_MSE_MAX,
        "chirp rel mse too high: {c_mse}"
    );
}

#[test]
fn align_tempogram() {
    let onset_ref = match load_vec("onset_strength.npy") {
        Some(v) => v,
        None => return,
    };
    let tempogram_ref = match load_array2("tempogram.npy") {
        Some(v) => v,
        None => return,
    };

    let sr = 22050;
    let hop_length = 128;
    let win_length = 384;

    let tempogram = feature::tempo::tempogram(&onset_ref, sr, hop_length, win_length);

    let t_mse = rel_mse(
        tempogram.as_slice().unwrap(),
        tempogram_ref.as_slice().unwrap(),
    );
    println!(
        "TEMPOGRAM relative MSE: {:.6} (threshold: {:.2})",
        t_mse, TEMPOGRAM_REL_MSE_MAX
    );
    assert!(
        t_mse < TEMPOGRAM_REL_MSE_MAX,
        "tempogram rel mse too high: {t_mse}"
    );
}

#[test]
fn align_autocorrelate() {
    let sine = match load_vec("sine.npy") {
        Some(v) => v,
        None => return,
    };
    let acf_ref = match load_vec("autocorrelate.npy") {
        Some(v) => v,
        None => return,
    };

    let input = &sine[..1000];
    let acf = utils::autocorrelate(input, None);

    let len = acf.len().min(acf_ref.len());
    let a_mse = rel_mse(&acf[..len], &acf_ref[..len]);
    println!(
        "AUTOCORRELATE relative MSE: {:.6} (threshold: {:.2})",
        a_mse, AUTOCORR_REL_MSE_MAX
    );
    assert!(
        a_mse < AUTOCORR_REL_MSE_MAX,
        "autocorrelate rel mse too high: {a_mse}"
    );
}

#[test]
fn align_normalize() {
    let sine = match load_vec("sine.npy") {
        Some(v) => v,
        None => return,
    };
    let norm_ref = match load_vec("normalize_l2.npy") {
        Some(v) => v,
        None => return,
    };

    let input = &sine[..100];
    let normalized = utils::normalize(input, utils::NormType::L2);

    let n_mse = rel_mse(&normalized, &norm_ref);
    println!(
        "NORMALIZE relative MSE: {:.6} (threshold: {:.2})",
        n_mse, NORMALIZE_REL_MSE_MAX
    );
    assert!(
        n_mse < NORMALIZE_REL_MSE_MAX,
        "normalize rel mse too high: {n_mse}"
    );
}
