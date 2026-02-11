use crate::effects::phase_vocoder::time_stretch;
use crate::io::resample;
use ndarray::Array2;

pub fn pitch_shift(
    y: &[f32],
    sr: u32,
    n_steps: f32,
    n_fft: usize,
    hop_length: usize,
) -> crate::Result<Vec<f32>> {
    if y.is_empty() {
        return Ok(Vec::new());
    }
    let rate = 2.0f32.powf(n_steps / 12.0);
    let stretched = time_stretch(y, 1.0 / rate, n_fft, hop_length)?;

    let mut data = Array2::<f32>::zeros((1, stretched.len()));
    for (i, v) in stretched.iter().enumerate() {
        data[(0, i)] = *v;
    }

    let target_sr = (sr as f32 * rate) as u32;
    match resample(&data, sr, target_sr) {
        Ok(out) => {
            let mut v = out.row(0).to_vec();
            if v.len() > y.len() {
                v.truncate(y.len());
            }
            Ok(v)
        }
        Err(_) => Ok(stretched),
    }
}
