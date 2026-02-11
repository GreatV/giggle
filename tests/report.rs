use giggle::{feature, spectrum, window};
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

fn mse(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut acc = 0.0f32;
    for i in 0..n {
        let d = a[i] - b[i];
        acc += d * d;
    }
    acc / n as f32
}

fn mean_abs(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut acc = 0.0f32;
    for i in 0..n {
        acc += (a[i] - b[i]).abs();
    }
    acc / n as f32
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut max_v = 0.0f32;
    for i in 0..n {
        let d = (a[i] - b[i]).abs();
        if d > max_v {
            max_v = d;
        }
    }
    max_v
}

fn mean_abs_by_row(a: &Array2<f32>, b: &Array2<f32>) -> Vec<f32> {
    let rows = a.shape()[0].min(b.shape()[0]);
    let cols = a.shape()[1].min(b.shape()[1]);
    let mut out = vec![0.0f32; rows];
    if rows == 0 || cols == 0 {
        return out;
    }
    for r in 0..rows {
        let mut acc = 0.0f32;
        for c in 0..cols {
            acc += (a[(r, c)] - b[(r, c)]).abs();
        }
        out[r] = acc / cols as f32;
    }
    out
}

fn mean_abs_by_col(a: &Array2<f32>, b: &Array2<f32>) -> Vec<f32> {
    let rows = a.shape()[0].min(b.shape()[0]);
    let cols = a.shape()[1].min(b.shape()[1]);
    let mut out = vec![0.0f32; cols];
    if rows == 0 || cols == 0 {
        return out;
    }
    for c in 0..cols {
        let mut acc = 0.0f32;
        for r in 0..rows {
            acc += (a[(r, c)] - b[(r, c)]).abs();
        }
        out[c] = acc / rows as f32;
    }
    out
}

fn top_k(values: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut pairs: Vec<(usize, f32)> = values.iter().cloned().enumerate().collect();
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    pairs.truncate(k);
    pairs
}

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
fn write_alignment_report() {
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
    let stft_r = match load_array2("stft_real.npy") {
        Some(v) => v,
        None => return,
    };
    let stft_i = match load_array2("stft_imag.npy") {
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
    let contrast_ref = match load_array2("contrast.npy") {
        Some(v) => v,
        None => return,
    };

    let mel = feature::mel::melspectrogram(&sine, 22050, 512, 128, 40).unwrap();
    let mfcc = feature::mfcc::mfcc(&sine, 22050, 13, 512, 128, 40).unwrap();
    let zcr = feature::basic::zero_crossing_rate_frames(&sine, 2048, 512, true).unwrap();
    let rms = feature::basic::rms_frames(&sine, 2048, 512, true).unwrap();
    let contrast = feature::contrast::spectral_contrast(&sine, 22050, 512, 128, 6, 200.0).unwrap();

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

    let mel_ref_slice = mel_ref.as_slice().unwrap();
    let mfcc_ref_slice = mfcc_ref.as_slice().unwrap();
    let contrast_ref_slice = contrast_ref.as_slice().unwrap();
    let mel_slice = mel.as_slice().unwrap();
    let mfcc_slice = mfcc.as_slice().unwrap();
    let contrast_slice = contrast.as_slice().unwrap();

    let mel_rel = rel_mse(mel_slice, mel_ref_slice);
    let mfcc_rel = rel_mse(mfcc_slice, mfcc_ref_slice);
    let stft_rel = rel_mse(&flat, &flat_ref);
    let zcr_rel = rel_mse(&zcr, &zcr_ref);
    let rms_rel = rel_mse(&rms, &rms_ref);
    let contrast_rel = rel_mse(contrast_slice, contrast_ref_slice);

    let mel_mse = mse(mel_slice, mel_ref_slice);
    let mfcc_mse = mse(mfcc_slice, mfcc_ref_slice);
    let stft_mse = mse(&flat, &flat_ref);
    let zcr_mse = mse(&zcr, &zcr_ref);
    let rms_mse = mse(&rms, &rms_ref);
    let contrast_mse = mse(contrast_slice, contrast_ref_slice);

    let mel_mean_abs = mean_abs(mel_slice, mel_ref_slice);
    let mfcc_mean_abs = mean_abs(mfcc_slice, mfcc_ref_slice);
    let stft_mean_abs = mean_abs(&flat, &flat_ref);
    let zcr_mean_abs = mean_abs(&zcr, &zcr_ref);
    let rms_mean_abs = mean_abs(&rms, &rms_ref);
    let contrast_mean_abs = mean_abs(contrast_slice, contrast_ref_slice);

    let mel_max_abs = max_abs(mel_slice, mel_ref_slice);
    let mfcc_max_abs = max_abs(mfcc_slice, mfcc_ref_slice);
    let stft_max_abs = max_abs(&flat, &flat_ref);
    let zcr_max_abs = max_abs(&zcr, &zcr_ref);
    let rms_max_abs = max_abs(&rms, &rms_ref);
    let contrast_max_abs = max_abs(contrast_slice, contrast_ref_slice);

    let report = format!(
        "alignment_report\n\
mel: rel_mse={mel_rel} mse={mel_mse} mean_abs={mel_mean_abs} max_abs={mel_max_abs}\n\
mfcc: rel_mse={mfcc_rel} mse={mfcc_mse} mean_abs={mfcc_mean_abs} max_abs={mfcc_max_abs}\n\
stft: rel_mse={stft_rel} mse={stft_mse} mean_abs={stft_mean_abs} max_abs={stft_max_abs}\n\
zcr: rel_mse={zcr_rel} mse={zcr_mse} mean_abs={zcr_mean_abs} max_abs={zcr_max_abs}\n\
rms: rel_mse={rms_rel} mse={rms_mse} mean_abs={rms_mean_abs} max_abs={rms_max_abs}\n\
contrast: rel_mse={contrast_rel} mse={contrast_mse} mean_abs={contrast_mean_abs} max_abs={contrast_max_abs}\n"
    );

    let out_path = Path::new("tests/data/alignment_report.txt");
    std::fs::write(out_path, report).ok();

    let mel_row = mean_abs_by_row(&mel, &mel_ref);
    let mel_col = mean_abs_by_col(&mel, &mel_ref);
    let mfcc_row = mean_abs_by_row(&mfcc, &mfcc_ref);
    let mfcc_col = mean_abs_by_col(&mfcc, &mfcc_ref);
    let contrast_row = mean_abs_by_row(&contrast, &contrast_ref);
    let contrast_col = mean_abs_by_col(&contrast, &contrast_ref);

    let stft_ref =
        Array2::from_shape_vec((stft_r.shape()[0], stft_r.shape()[1] * 2), flat_ref.clone())
            .unwrap_or_else(|_| Array2::zeros((0, 0)));
    let stft_cur = Array2::from_shape_vec((stft_r.shape()[0], stft_r.shape()[1] * 2), flat.clone())
        .unwrap_or_else(|_| Array2::zeros((0, 0)));
    let stft_row = mean_abs_by_row(&stft_cur, &stft_ref);
    let stft_col = mean_abs_by_col(&stft_cur, &stft_ref);

    let detailed = format!(
        "alignment_report_detailed\n\
config: stft(n_fft=512 hop=128 win=512 window=hann center=constant)\n\
config: mel(scale=slaney norm=slaney power=2)\n\
config: mfcc(dct=ortho power_to_db ref=1.0 top_db=80)\n\
mel_top_bands={:?}\n\
mel_top_frames={:?}\n\
mfcc_top_coeffs={:?}\n\
mfcc_top_frames={:?}\n\
contrast_top_bands={:?}\n\
contrast_top_frames={:?}\n\
stft_top_freq_bins={:?}\n\
stft_top_time_bins={:?}\n",
        top_k(&mel_row, 5),
        top_k(&mel_col, 5),
        top_k(&mfcc_row, 5),
        top_k(&mfcc_col, 5),
        top_k(&contrast_row, 5),
        top_k(&contrast_col, 5),
        top_k(&stft_row, 5),
        top_k(&stft_col, 5),
    );

    let detailed_path = Path::new("tests/data/alignment_report_detailed.txt");
    std::fs::write(detailed_path, detailed).ok();
}
