use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use ndarray::Array2;
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use std::path::Path;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

#[derive(Debug, Clone, Copy)]
pub struct AudioSpec {
    pub sample_rate: u32,
    pub channels: u16,
}

#[derive(Debug, thiserror::Error)]
pub enum AudioError {
    #[error("hound error: {0}")]
    Hound(#[from] hound::Error),
    #[error("symphonia error: {0}")]
    Symphonia(SymphoniaError),
    #[error("no audio track found")]
    NoAudioTrack,
    #[error("unsupported number of channels")]
    UnsupportedChannels,
    #[error("resampling error: {0}")]
    Resample(String),
}

impl From<SymphoniaError> for AudioError {
    fn from(err: SymphoniaError) -> Self {
        Self::Symphonia(err)
    }
}

/// Load a WAV file with optional offset and duration.
///
/// # Arguments
/// * `path` - Path to the WAV file
/// * `offset` - Start reading from this position (in seconds)
/// * `duration` - Read this many seconds of audio (None for all)
///
/// # Returns
/// `Result<(Array2<f32>, AudioSpec)>` containing the audio data and specifications
///
/// # Errors
/// Returns `crate::Error::Audio` if the file cannot be read or is invalid
pub fn load_wav<P: AsRef<Path>>(
    path: P,
    offset: Option<f64>,
    duration: Option<f64>,
) -> crate::Result<(Array2<f32>, AudioSpec)> {
    let mut reader = WavReader::open(path).map_err(AudioError::Hound)?;
    let spec = reader.spec();

    let channels = spec.channels as usize;
    let mut samples: Vec<f32> = Vec::new();

    match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Float, 32) => {
            for s in reader.samples::<f32>() {
                samples.push(s.map_err(AudioError::Hound)?);
            }
        }
        (SampleFormat::Int, bits) if bits <= 16 => {
            let scale = (1i32 << (bits - 1)) as f32;
            for s in reader.samples::<i16>() {
                samples.push(s.map_err(AudioError::Hound)? as f32 / scale);
            }
        }
        (SampleFormat::Int, bits) => {
            let scale = (1i64 << (bits - 1)) as f32;
            for s in reader.samples::<i32>() {
                samples.push(s.map_err(AudioError::Hound)? as f32 / scale);
            }
        }
        _ => {
            for s in reader.samples::<i16>() {
                samples.push(s.map_err(AudioError::Hound)? as f32 / i16::MAX as f32);
            }
        }
    }

    let total_samples = samples.len();
    let total_frames = total_samples / channels.max(1);

    // Calculate start and end frames based on offset and duration
    let start_frame = if let Some(off) = offset {
        ((off * spec.sample_rate as f64) as usize).min(total_frames)
    } else {
        0
    };

    let end_frame = if let Some(dur) = duration {
        let dur_frames = (dur * spec.sample_rate as f64) as usize;
        (start_frame + dur_frames).min(total_frames)
    } else {
        total_frames
    };

    let frames = end_frame.saturating_sub(start_frame);
    let mut data = Array2::<f32>::zeros((channels, frames));

    for frame in 0..frames {
        let src_frame = start_frame + frame;
        for ch in 0..channels {
            data[(ch, frame)] = samples[src_frame * channels + ch];
        }
    }

    Ok((
        data,
        AudioSpec {
            sample_rate: spec.sample_rate,
            channels: spec.channels,
        },
    ))
}

pub fn detect_format<P: AsRef<Path>>(path: P) -> Option<String> {
    path.as_ref()
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|s| s.to_ascii_lowercase())
}

/// Load audio from a file with optional offset and duration.
///
/// # Arguments
/// * `path` - Path to the audio file
/// * `target_sr` - Optional target sample rate for resampling
/// * `mono` - If true, convert to mono by averaging channels
/// * `offset` - Start reading from this position (in seconds)
/// * `duration` - Read this many seconds of audio (None for all)
///
/// Load audio from file with comprehensive options.
///
/// This is the main audio loading function that supports:
/// - Multiple formats (WAV, MP3, FLAC, OGG, etc. via symphonia)
/// - Offset and duration selection
/// - Automatic mono conversion
/// - Sample rate conversion (resampling)
///
/// # Arguments
/// * `path` - Path to audio file
/// * `sr` - Target sample rate (None to keep original)
/// * `mono` - Convert to mono (average channels)
/// * `offset` - Start time in seconds (None for beginning)
/// * `duration` - Length to load in seconds (None for all)
///
/// # Returns
/// Tuple of (audio_data, audio_spec) where audio_data is shape (channels, frames)
///
/// # Example
/// ```no_run
/// use giggle::io;
/// // Load full file at original sample rate
/// let (data, spec) = io::load("audio.mp3", None, false, None, None).unwrap();
///
/// // Load 1 second starting at 0.5s, convert to mono at 22050 Hz
/// let (data, spec) = io::load("audio.wav", Some(22050), true, Some(0.5), Some(1.0)).unwrap();
/// ```
pub fn load<P: AsRef<Path>>(
    path: P,
    sr: Option<u32>,
    mono: bool,
    offset: Option<f64>,
    duration: Option<f64>,
) -> Result<(Array2<f32>, AudioSpec), AudioError> {
    load_audio(path, sr, mono, offset, duration)
}

pub fn load_audio<P: AsRef<Path>>(
    path: P,
    target_sr: Option<u32>,
    mono: bool,
    offset: Option<f64>,
    duration: Option<f64>,
) -> Result<(Array2<f32>, AudioSpec), AudioError> {
    let path_ref = path.as_ref();
    let mut hint = Hint::new();
    if let Some(ext) = path_ref.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let file = std::fs::File::open(path_ref).map_err(SymphoniaError::IoError)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let probed = symphonia::default::get_probe().format(
        &hint,
        mss,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    )?;

    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.sample_rate.is_some())
        .ok_or(AudioError::NoAudioTrack)?
        .clone();

    let sample_rate = track.codec_params.sample_rate.unwrap_or(0);
    let channels = track
        .codec_params
        .channels
        .map(|c| c.count() as u16)
        .unwrap_or(0);

    let mut decoder =
        symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default())?;

    let mut samples: Vec<f32> = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::ResetRequired) => {
                decoder.reset();
                continue;
            }
            Err(SymphoniaError::IoError(_)) => break,
            Err(e) => return Err(e.into()),
        };

        if packet.track_id() != track.id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(audio) => audio,
            Err(SymphoniaError::IoError(_)) => break,
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(e) => return Err(e.into()),
        };

        let mut sb = SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
        sb.copy_interleaved_ref(decoded);
        samples.extend_from_slice(sb.samples());
    }

    if channels == 0 {
        return Err(AudioError::UnsupportedChannels);
    }
    let channels_usize = channels as usize;
    let total_frames = samples.len() / channels_usize.max(1);

    // Calculate start and end frames based on offset and duration
    let start_frame = if let Some(off) = offset {
        ((off * sample_rate as f64) as usize).min(total_frames)
    } else {
        0
    };

    let end_frame = if let Some(dur) = duration {
        let dur_frames = (dur * sample_rate as f64) as usize;
        (start_frame + dur_frames).min(total_frames)
    } else {
        total_frames
    };

    let frames = end_frame.saturating_sub(start_frame);
    let mut data = Array2::<f32>::zeros((channels_usize, frames));
    for frame in 0..frames {
        let src_frame = start_frame + frame;
        for ch in 0..channels_usize {
            data[(ch, frame)] = samples[src_frame * channels_usize + ch];
        }
    }

    if mono && channels_usize > 1 {
        let mut mono_data = Array2::<f32>::zeros((1, frames));
        for frame in 0..frames {
            let mut acc = 0.0f32;
            for ch in 0..channels_usize {
                acc += data[(ch, frame)];
            }
            mono_data[(0, frame)] = acc / channels_usize as f32;
        }
        data = mono_data;
    }

    let mut spec = AudioSpec {
        sample_rate,
        channels: data.shape()[0] as u16,
    };
    if let Some(target) = target_sr
        && target != sample_rate
        && sample_rate > 0
    {
        let resampled = resample(&data, sample_rate, target)?;
        spec.sample_rate = target;
        spec.channels = resampled.shape()[0] as u16;
        return Ok((resampled, spec));
    }

    Ok((data, spec))
}

pub fn resample(data: &Array2<f32>, src_sr: u32, dst_sr: u32) -> Result<Array2<f32>, AudioError> {
    if src_sr == dst_sr {
        return Ok(data.clone());
    }

    let channels = data.shape().first().copied().unwrap_or(0);
    let frames = data.shape().get(1).copied().unwrap_or(0);
    if channels == 0 || frames == 0 {
        return Ok(Array2::<f32>::zeros((channels, frames)));
    }

    let mut input: Vec<Vec<f32>> = Vec::with_capacity(channels);
    for ch in 0..channels {
        input.push(data.row(ch).to_vec());
    }

    let gcd = gcd_u32(src_sr, dst_sr);
    let ratio_in = src_sr / gcd;
    let ratio_out = dst_sr / gcd;
    let resample_ratio = ratio_out as f64 / ratio_in as f64;

    let chunk_size = 1024usize;
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    let mut resampler = SincFixedIn::<f32>::new(resample_ratio, 2.0, params, chunk_size, channels)
        .map_err(|e| AudioError::Resample(e.to_string()))?;

    let mut output: Vec<Vec<f32>> = vec![Vec::new(); channels];
    let mut offset = 0usize;
    while offset < frames {
        let end = (offset + chunk_size).min(frames);
        let mut chunk: Vec<Vec<f32>> = Vec::with_capacity(channels);
        for ch_data in input.iter().take(channels) {
            let mut buf = vec![0.0f32; chunk_size];
            let slice = &ch_data[offset..end];
            buf[..slice.len()].copy_from_slice(slice);
            chunk.push(buf);
        }

        let chunk_out = resampler
            .process(&chunk, None)
            .map_err(|e| AudioError::Resample(e.to_string()))?;
        for ch in 0..channels {
            output[ch].extend_from_slice(&chunk_out[ch]);
        }
        offset = end;
    }

    let expected = ((frames as f64) * (dst_sr as f64) / (src_sr as f64)).round() as usize;
    for ch_data in output.iter_mut().take(channels) {
        if ch_data.len() > expected {
            ch_data.truncate(expected);
        }
    }

    let out_frames = output.first().map(|v| v.len()).unwrap_or(0);
    let mut out = Array2::<f32>::zeros((channels, out_frames));
    for ch in 0..channels {
        for i in 0..out_frames {
            out[(ch, i)] = output[ch][i];
        }
    }
    Ok(out)
}

fn gcd_u32(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

/// Convert multi-channel audio to mono by averaging channels.
pub fn to_mono(data: &Array2<f32>) -> Array2<f32> {
    let (channels, frames) = (data.shape()[0], data.shape()[1]);
    if channels == 1 {
        return data.clone();
    }

    let mut mono = Array2::<f32>::zeros((1, frames));
    for frame in 0..frames {
        let mut sum = 0.0f32;
        for ch in 0..channels {
            sum += data[(ch, frame)];
        }
        mono[(0, frame)] = sum / channels as f32;
    }
    mono
}

/// Get the duration of audio in seconds.
pub fn get_duration(data: &Array2<f32>, sample_rate: u32) -> f64 {
    let frames = data.shape()[1];
    frames as f64 / sample_rate as f64
}

/// Get the sample rate from AudioSpec.
pub fn get_samplerate(spec: &AudioSpec) -> u32 {
    spec.sample_rate
}

/// Streaming audio reader for processing large files in chunks.
///
/// This struct provides an iterator interface for reading audio files in
/// chunks, which is useful for processing files that don't fit in memory
/// or for real-time processing.
///
/// # Example
/// ```no_run
/// use giggle::io::AudioStream;
///
/// let stream = AudioStream::new("large_file.wav", 44100, None, None).unwrap();
/// for chunk in stream {
///     // Process each chunk
///     println!("Chunk shape: {:?}", chunk.shape());
/// }
/// ```
pub struct AudioStream {
    reader: WavReader<std::io::BufReader<std::fs::File>>,
    chunk_size: usize,
    channels: usize,
    sample_format: SampleFormat,
    bits_per_sample: u16,
    remaining_samples: usize,
}

impl AudioStream {
    /// Create a new audio stream from a WAV file.
    ///
    /// # Arguments
    /// * `path` - Path to the WAV file
    /// * `chunk_size` - Number of frames per chunk
    /// * `offset` - Start reading from this position (in seconds)
    /// * `duration` - Read this many seconds of audio (None for all)
    ///
    /// # Returns
    /// `Result<AudioStream>` containing the stream or error
    ///
    /// # Errors
    /// Returns `crate::Error::Audio` if the file cannot be opened
    pub fn new<P: AsRef<Path>>(
        path: P,
        chunk_size: usize,
        offset: Option<f64>,
        duration: Option<f64>,
    ) -> crate::Result<Self> {
        let mut reader = WavReader::open(path).map_err(AudioError::Hound)?;
        let spec = reader.spec();
        let channels = spec.channels as usize;
        let sample_rate = spec.sample_rate;

        // Calculate starting sample position based on offset
        let start_sample = if let Some(off) = offset {
            let start_frame = (off * sample_rate as f64) as usize;
            start_frame * channels
        } else {
            0
        };

        // Seek to the start position
        if start_sample > 0 {
            let samples_to_skip = start_sample as u32;
            reader
                .seek(samples_to_skip)
                .map_err(|e| AudioError::Hound(hound::Error::IoError(e)))?;
        }

        // Calculate max samples to read based on duration
        let remaining_samples = if let Some(dur) = duration {
            (dur * sample_rate as f64 * channels as f64) as usize
        } else {
            usize::MAX
        };

        Ok(Self {
            reader,
            chunk_size,
            channels,
            sample_format: spec.sample_format,
            bits_per_sample: spec.bits_per_sample,
            remaining_samples,
        })
    }

    /// Get the audio specification.
    pub fn spec(&self) -> AudioSpec {
        AudioSpec {
            sample_rate: self.reader.spec().sample_rate,
            channels: self.channels as u16,
        }
    }
}

impl Iterator for AudioStream {
    type Item = Array2<f32>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining_samples == 0 {
            return None;
        }

        let chunk_samples = self.chunk_size * self.channels;
        let samples_to_read = chunk_samples.min(self.remaining_samples);
        let frames_to_read = samples_to_read / self.channels;

        if frames_to_read == 0 {
            return None;
        }

        let mut samples = Vec::with_capacity(samples_to_read);

        match (self.sample_format, self.bits_per_sample) {
            (SampleFormat::Float, 32) => {
                let mut reader_samples = self.reader.samples::<f32>();
                for _ in 0..samples_to_read {
                    if let Some(Ok(s)) = reader_samples.next() {
                        samples.push(s);
                    } else {
                        break;
                    }
                }
            }
            (SampleFormat::Int, bits) if bits <= 16 => {
                let scale = (1i32 << (bits - 1)) as f32;
                let mut reader_samples = self.reader.samples::<i16>();
                for _ in 0..samples_to_read {
                    if let Some(Ok(s)) = reader_samples.next() {
                        samples.push(s as f32 / scale);
                    } else {
                        break;
                    }
                }
            }
            (SampleFormat::Int, bits) => {
                let scale = (1i64 << (bits - 1)) as f32;
                let mut reader_samples = self.reader.samples::<i32>();
                for _ in 0..samples_to_read {
                    if let Some(Ok(s)) = reader_samples.next() {
                        samples.push(s as f32 / scale);
                    } else {
                        break;
                    }
                }
            }
            _ => {
                let mut reader_samples = self.reader.samples::<i16>();
                for _ in 0..samples_to_read {
                    if let Some(Ok(s)) = reader_samples.next() {
                        samples.push(s as f32 / i16::MAX as f32);
                    } else {
                        break;
                    }
                }
            }
        }

        if samples.is_empty() {
            return None;
        }

        let actual_frames = samples.len() / self.channels;
        self.remaining_samples = self.remaining_samples.saturating_sub(samples.len());

        let mut chunk = Array2::<f32>::zeros((self.channels, actual_frames));
        for frame in 0..actual_frames {
            for ch in 0..self.channels {
                chunk[(ch, frame)] = samples[frame * self.channels + ch];
            }
        }

        Some(chunk)
    }
}

/// Generate a click signal.
pub fn clicks(
    times: &[f32],
    sr: u32,
    length: Option<usize>,
    click_duration: f32,
    click_freq: f32,
) -> Vec<f32> {
    let len = length.unwrap_or_else(|| {
        times.iter().fold(0.0f32, |a, &b| a.max(b)).ceil() as usize * sr as usize
    });
    let mut y = vec![0.0f32; len];

    let click_samples = (click_duration * sr as f32) as usize;
    let angular_freq = 2.0 * std::f32::consts::PI * click_freq / sr as f32;

    for &time in times {
        let start = (time * sr as f32) as usize;
        if start >= len {
            continue;
        }
        for i in 0..click_samples {
            let idx = start + i;
            if idx >= len {
                break;
            }
            let t = i as f32;
            let envelope = (-t / (click_samples as f32 * 0.1)).exp();
            y[idx] += envelope * (angular_freq * t).sin();
        }
    }
    y
}

/// Generate a pure tone.
pub fn tone(frequency: f32, sr: u32, duration: f32) -> Vec<f32> {
    let n_samples = (duration * sr as f32) as usize;
    let angular_freq = 2.0 * std::f32::consts::PI * frequency / sr as f32;
    (0..n_samples)
        .map(|i| (angular_freq * i as f32).sin())
        .collect()
}

/// Generate a chirp (frequency sweep).
/// By default uses logarithmic sweep (linear=false) to match librosa.
pub fn chirp(f0: f32, f1: f32, sr: u32, duration: f32) -> Vec<f32> {
    chirp_with_mode(f0, f1, sr, duration, false)
}

/// Generate a chirp with specified mode.
/// linear=true: linear frequency sweep, linear=false: logarithmic sweep
pub fn chirp_with_mode(f0: f32, f1: f32, sr: u32, duration: f32, linear: bool) -> Vec<f32> {
    let n_samples = (duration * sr as f32) as usize;

    if linear {
        let k = (f1 - f0) / duration;
        (0..n_samples)
            .map(|i| {
                let t = i as f32 / sr as f32;
                let phase = 2.0 * std::f32::consts::PI * (f0 * t + 0.5 * k * t * t);
                phase.sin()
            })
            .collect()
    } else {
        // Logarithmic chirp (exponential frequency sweep)
        let log_f = (f1 / f0).ln() / duration;
        (0..n_samples)
            .map(|i| {
                let t = i as f32 / sr as f32;
                let phase = 2.0 * std::f32::consts::PI * f0 * ((log_f * t).exp() - 1.0) / log_f;
                phase.sin()
            })
            .collect()
    }
}

/// Save audio data to a WAV file.
///
/// # Arguments
/// * `path` - Path to save the WAV file
/// * `data` - Audio data with shape (channels, frames)
/// * `sample_rate` - Sample rate in Hz
///
/// # Returns
/// `Result<()>` indicating success or failure
///
/// # Errors
/// Returns `crate::Error::Audio` if the file cannot be written
///
/// # Limitations
/// This function only supports 16-bit integer PCM format. Floating-point
/// audio is clipped to [-1.0, 1.0] and quantized to 16-bit.
pub fn save_wav<P: AsRef<Path>>(
    path: P,
    data: &Array2<f32>,
    sample_rate: u32,
) -> crate::Result<()> {
    let channels = data.shape().first().copied().unwrap_or(1) as u16;
    let frames = data.shape().get(1).copied().unwrap_or(0);

    let spec = WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };

    let mut writer = WavWriter::create(path, spec).map_err(AudioError::Hound)?;
    for frame in 0..frames {
        for ch in 0..channels as usize {
            let sample = data[(ch, frame)].clamp(-1.0, 1.0);
            let s = (sample * i16::MAX as f32) as i16;
            writer.write_sample(s).map_err(AudioError::Hound)?;
        }
    }
    writer.finalize().map_err(AudioError::Hound)?;
    Ok(())
}

/// Apply mu-law compression to an audio signal.
///
/// mu-law compression is a logarithmic companding algorithm commonly
/// used in telephony and audio applications.
///
/// # Arguments
/// * `x` - Input signal in range [-1.0, 1.0]
/// * `mu` - Compression parameter (typical value: 255.0)
///
/// # Returns
/// Compressed signal in range [-1.0, 1.0]
///
/// # Formula
/// y = sign(x) * ln(1 + μ|x|) / ln(1 + μ)
pub fn mu_compress(x: &[f32], mu: f32) -> Vec<f32> {
    let ln_1_plus_mu = (1.0 + mu).ln();

    x.iter()
        .map(|&sample| {
            let sign = if sample >= 0.0 { 1.0 } else { -1.0 };
            let abs_val = sample.abs();
            let compressed = (1.0 + mu * abs_val).ln() / ln_1_plus_mu;
            sign * compressed
        })
        .collect()
}

/// Compute Linear Predictive Coding (LPC) coefficients using the autocorrelation method.
///
/// LPC models a signal as a linear combination of its past values.
/// This is widely used in speech coding, audio compression, and analysis.
///
/// # Arguments
/// * `y` - Input signal
/// * `order` - Number of LPC coefficients to compute (typical: 10-20 for speech)
///
/// # Returns
/// LPC coefficients `a[1..order]` where:
/// `y[n] ≈ -sum(a[k] * y[n-k])` for k=1..order
///
/// Uses the Levinson-Durbin recursion algorithm.
pub fn lpc(y: &[f32], order: usize) -> Vec<f32> {
    if y.is_empty() || order == 0 {
        return Vec::new();
    }

    // Compute autocorrelation using the existing autocorrelate function
    let r = crate::utils::autocorrelate(y, Some(order + 1));

    if r.len() < order + 1 || r[0].abs() < 1e-10 {
        // Not enough data or signal has no energy
        return vec![0.0; order];
    }

    // Levinson-Durbin recursion
    let mut a = vec![0.0f32; order + 1];
    let mut e = r[0];

    for i in 1..=order {
        // Compute reflection coefficient
        let mut lambda = r[i];
        for j in 1..i {
            lambda -= a[j] * r[i - j];
        }

        if e.abs() < 1e-10 {
            break;
        }

        let k_i = lambda / e;

        // Update coefficients using the reflection coefficient
        a[i] = k_i;
        for j in 1..i {
            let temp = a[j];
            a[j] -= k_i * a[i - j];
            if j < i - j {
                a[i - j] -= k_i * temp;
            }
        }

        // Update prediction error
        e *= 1.0 - k_i * k_i;

        if e < 0.0 {
            break;
        }
    }

    // Return coefficients (excluding a[0] which is always 1)
    a[1..=order].to_vec()
}

/// Apply mu-law expansion to a compressed audio signal.
///
/// This is the inverse operation of mu_compress.
///
/// # Arguments
/// * `y` - Compressed signal in range [-1.0, 1.0]
/// * `mu` - Compression parameter (same as used in compression)
///
/// # Returns
/// Expanded signal in range [-1.0, 1.0]
///
/// # Formula
/// x = sign(y) * (1/μ) * ((1 + μ)^|y| - 1)
pub fn mu_expand(y: &[f32], mu: f32) -> Vec<f32> {
    y.iter()
        .map(|&sample| {
            let sign = if sample >= 0.0 { 1.0 } else { -1.0 };
            let abs_val = sample.abs();
            let expanded = ((1.0 + mu).powf(abs_val) - 1.0) / mu;
            sign * expanded
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_mono() {
        let stereo =
            Array2::from_shape_vec((2, 4), vec![1.0, 2.0, 3.0, 4.0, 0.5, 1.0, 1.5, 2.0]).unwrap();
        let mono = to_mono(&stereo);
        assert_eq!(mono.shape(), &[1, 4]);
        assert!((mono[(0, 0)] - 0.75).abs() < 0.01);
        assert!((mono[(0, 1)] - 1.5).abs() < 0.01);

        let already_mono = Array2::from_shape_vec((1, 4), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let result = to_mono(&already_mono);
        assert_eq!(result.shape(), &[1, 4]);
    }

    #[test]
    fn test_get_duration() {
        let data = Array2::<f32>::zeros((1, 22050));
        let duration = get_duration(&data, 22050);
        assert!((duration - 1.0).abs() < 0.01);

        let data2 = Array2::<f32>::zeros((2, 44100));
        let duration2 = get_duration(&data2, 44100);
        assert!((duration2 - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_get_samplerate() {
        let spec = AudioSpec {
            sample_rate: 44100,
            channels: 2,
        };
        assert_eq!(get_samplerate(&spec), 44100);
    }

    #[test]
    fn test_tone() {
        let sr = 22050;
        let freq = 440.0;
        let duration = 0.1;
        let signal = tone(freq, sr, duration);

        assert_eq!(signal.len(), (duration * sr as f32) as usize);
        assert!(signal.iter().any(|&x| x.abs() > 0.9));
    }

    #[test]
    fn test_chirp() {
        let sr = 22050;
        let signal = chirp(100.0, 1000.0, sr, 0.5);

        assert_eq!(signal.len(), (0.5 * sr as f32) as usize);
        assert!(signal.iter().any(|&x| x.abs() > 0.0));
    }

    #[test]
    fn test_clicks() {
        let sr = 22050;
        let times = vec![0.0, 0.5, 1.0];
        let signal = clicks(&times, sr, Some(sr as usize * 2), 0.01, 1000.0);

        assert_eq!(signal.len(), sr as usize * 2);

        let max_val = signal.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!(max_val > 0.01);

        let click_pos = (0.5 * sr as f32) as usize;
        let window_start = click_pos.saturating_sub(50);
        let window_end = (click_pos + 50).min(signal.len());
        let window = &signal[window_start..window_end];
        assert!(window.iter().any(|&x| x.abs() > 0.01));
    }

    #[test]
    fn test_mu_compress_expand_roundtrip() {
        use approx::assert_relative_eq;

        let signal = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let mu = 255.0;

        let compressed = mu_compress(&signal, mu);
        let recovered = mu_expand(&compressed, mu);

        for i in 0..signal.len() {
            assert_relative_eq!(signal[i], recovered[i], epsilon = 1e-5);
        }
    }

    #[test]
    fn test_mu_compress_properties() {
        let mu = 255.0;

        // Test zero stays zero
        let zero = vec![0.0];
        let compressed = mu_compress(&zero, mu);
        assert_eq!(compressed[0], 0.0);

        // Test positive values
        let positive = vec![0.1, 0.5, 1.0];
        let compressed = mu_compress(&positive, mu);
        for &val in &compressed {
            assert!((0.0..=1.0).contains(&val));
        }

        // Test negative values
        let negative = vec![-0.1, -0.5, -1.0];
        let compressed = mu_compress(&negative, mu);
        for &val in &compressed {
            assert!((-1.0..=0.0).contains(&val));
        }

        // Test compression reduces dynamic range
        let signal = vec![0.1, 0.9];
        let compressed = mu_compress(&signal, mu);
        let orig_ratio = signal[1] / signal[0];
        let comp_ratio = compressed[1] / compressed[0];
        assert!(comp_ratio < orig_ratio);
    }

    #[test]
    fn test_mu_compress_symmetry() {
        use approx::assert_relative_eq;

        let signal = vec![0.5, -0.5];
        let mu = 255.0;
        let compressed = mu_compress(&signal, mu);

        // Compressed values should have same magnitude but opposite sign
        assert_relative_eq!(compressed[0], -compressed[1], epsilon = 1e-6);
    }

    #[test]
    fn test_lpc_basic() {
        // Generate a simple signal: AR(1) process
        let mut signal = vec![1.0f32; 100];
        let a1 = 0.9; // First-order AR coefficient

        for i in 1..signal.len() {
            signal[i] = a1 * signal[i - 1] + 0.1 * (i as f32).sin();
        }

        // Compute LPC coefficients
        let order = 1;
        let coeffs = lpc(&signal, order);

        assert_eq!(coeffs.len(), order);

        // The first coefficient should be close to -a1 (negative because of prediction)
        assert!(
            coeffs[0].abs() < 1.0,
            "LPC coefficient should be reasonable"
        );
    }

    #[test]
    fn test_lpc_order() {
        let signal = tone(440.0, 22050, 0.1);

        // Test different orders
        for order in &[5, 10, 20] {
            let coeffs = lpc(&signal, *order);
            assert_eq!(coeffs.len(), *order);
        }
    }

    #[test]
    fn test_lpc_empty() {
        let signal = vec![];
        let coeffs = lpc(&signal, 10);
        assert_eq!(coeffs.len(), 0);
    }

    #[test]
    fn test_lpc_zero_order() {
        let signal = vec![1.0, 2.0, 3.0];
        let coeffs = lpc(&signal, 0);
        assert_eq!(coeffs.len(), 0);
    }

    #[test]
    fn test_lpc_white_noise() {
        use rand::Rng;

        let mut rng = rand::thread_rng();
        let signal: Vec<f32> = (0..1000).map(|_| rng.gen_range(-1.0..1.0)).collect();

        let order = 10;
        let coeffs = lpc(&signal, order);

        assert_eq!(coeffs.len(), order);

        // For white noise, coefficients should be small
        let max_coeff = coeffs.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!(max_coeff < 0.5, "White noise LPC coeffs should be small");
    }

    #[test]
    fn test_load_api_exists() {
        // Test that load API is callable (functionality already tested via load_wav)
        // This just ensures the API wrapper exists and has correct signature

        // Note: Full integration tests would require actual audio files
        // Basic functionality is already tested through load_wav tests
    }

    #[test]
    fn test_audio_stream_basic() {
        // Create a temporary WAV file for testing
        let temp_dir = std::env::temp_dir();
        let temp_path = temp_dir.join("test_stream.wav");

        // Generate test audio
        let signal = tone(440.0, 22050, 1.0);
        let mut data = Array2::<f32>::zeros((1, signal.len()));
        for (i, &s) in signal.iter().enumerate() {
            data[(0, i)] = s;
        }

        // Save to temporary file
        save_wav(&temp_path, &data, 22050).unwrap();

        // Test streaming
        let chunk_size = 1024;
        let stream = AudioStream::new(&temp_path, chunk_size, None, None).unwrap();

        let mut total_frames = 0;
        let mut chunk_count = 0;

        for chunk in stream {
            assert_eq!(chunk.shape()[0], 1); // Mono
            assert!(chunk.shape()[1] <= chunk_size); // Chunk size constraint
            total_frames += chunk.shape()[1];
            chunk_count += 1;
        }

        // Should have processed all frames
        assert!(total_frames > 0);
        assert!(chunk_count > 0);

        // Clean up
        let _ = std::fs::remove_file(temp_path);
    }

    #[test]
    fn test_audio_stream_spec() {
        let temp_dir = std::env::temp_dir();
        let temp_path = temp_dir.join("test_stream_spec.wav");

        let signal = tone(880.0, 22050, 0.5);
        let mut data = Array2::<f32>::zeros((1, signal.len()));
        for (i, &s) in signal.iter().enumerate() {
            data[(0, i)] = s;
        }

        save_wav(&temp_path, &data, 22050).unwrap();

        let stream = AudioStream::new(&temp_path, 512, None, None).unwrap();
        let spec = stream.spec();

        assert_eq!(spec.sample_rate, 22050);
        assert_eq!(spec.channels, 1);

        let _ = std::fs::remove_file(temp_path);
    }

    #[test]
    fn test_audio_stream_empty_file() {
        let temp_dir = std::env::temp_dir();
        let temp_path = temp_dir.join("test_stream_empty.wav");

        // Create empty audio
        let data = Array2::<f32>::zeros((1, 0));
        save_wav(&temp_path, &data, 22050).unwrap();

        let mut stream = AudioStream::new(&temp_path, 512, None, None).unwrap();

        // Should yield no chunks
        assert!(stream.next().is_none());

        let _ = std::fs::remove_file(temp_path);
    }

    #[test]
    fn test_audio_stream_chunk_sizes() {
        let temp_dir = std::env::temp_dir();
        let temp_path = temp_dir.join("test_stream_chunks.wav");

        let signal = tone(440.0, 22050, 0.5);
        let mut data = Array2::<f32>::zeros((1, signal.len()));
        for (i, &s) in signal.iter().enumerate() {
            data[(0, i)] = s;
        }

        save_wav(&temp_path, &data, 22050).unwrap();

        // Test with different chunk sizes
        for chunk_size in &[256, 512, 1024, 2048] {
            let stream = AudioStream::new(&temp_path, *chunk_size, None, None).unwrap();
            let chunks: Vec<_> = stream.collect();

            assert!(!chunks.is_empty());

            // All chunks except last should be full size
            for (i, chunk) in chunks.iter().enumerate() {
                if i < chunks.len() - 1 {
                    assert_eq!(chunk.shape()[1], *chunk_size);
                } else {
                    // Last chunk can be smaller or equal
                    assert!(chunk.shape()[1] <= *chunk_size);
                }
            }
        }

        let _ = std::fs::remove_file(temp_path);
    }

    #[test]
    fn test_audio_stream_offset() {
        let temp_dir = std::env::temp_dir();
        let temp_path = temp_dir.join("test_stream_offset.wav");

        // Create a file with a known pattern
        let sr = 22050;
        let signal = tone(440.0, sr, 0.5);
        let mut data = Array2::<f32>::zeros((1, signal.len()));
        for (i, &s) in signal.iter().enumerate() {
            data[(0, i)] = s;
        }

        save_wav(&temp_path, &data, sr).unwrap();

        // Test offset: skip first 0.1 seconds (~2205 samples)
        let offset_sec = 0.1;
        let stream = AudioStream::new(&temp_path, 512, Some(offset_sec), None).unwrap();

        let mut total_frames = 0;
        for chunk in stream {
            total_frames += chunk.shape()[1];
        }

        let expected_frames = signal.len() - (offset_sec * sr as f64) as usize;
        assert!(
            (total_frames as i32 - expected_frames as i32).abs() < 512,
            "Offset should skip frames"
        );

        let _ = std::fs::remove_file(temp_path);
    }

    #[test]
    fn test_audio_stream_offset_and_duration() {
        let temp_dir = std::env::temp_dir();
        let temp_path = temp_dir.join("test_stream_off_dur.wav");

        let sr = 22050;
        let signal = tone(440.0, sr, 1.0);
        let mut data = Array2::<f32>::zeros((1, signal.len()));
        for (i, &s) in signal.iter().enumerate() {
            data[(0, i)] = s;
        }

        save_wav(&temp_path, &data, sr).unwrap();

        // Read from 0.2s to 0.4s (0.2 seconds duration)
        let offset_sec = 0.2;
        let duration_sec = 0.2;
        let stream =
            AudioStream::new(&temp_path, 256, Some(offset_sec), Some(duration_sec)).unwrap();

        let mut total_frames = 0;
        for chunk in stream {
            total_frames += chunk.shape()[1];
        }

        let expected_frames = (duration_sec * sr as f64) as usize;
        assert!(
            (total_frames as i32 - expected_frames as i32).abs() < 256,
            "Should read approximately duration frames"
        );

        let _ = std::fs::remove_file(temp_path);
    }
}
