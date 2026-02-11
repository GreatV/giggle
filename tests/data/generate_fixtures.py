#!/usr/bin/env python3
"""Generate reference fixtures using librosa for Rust validation."""
from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(Path(__file__).parent / "fixtures"), help="output directory")
    parser.add_argument("--sr", type=int, default=22050)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sr = args.sr
    rng = np.random.default_rng(0)

    # Signals
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sine = 0.5 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    noise = rng.normal(0.0, 0.1, size=sr).astype(np.float32)

    np.save(out_dir / "sine.npy", sine)
    np.save(out_dir / "noise.npy", noise)

    # STFT + ISTFT
    stft = librosa.stft(sine, n_fft=512, hop_length=128, win_length=512, window="hann", center=True)
    np.save(out_dir / "stft_real.npy", stft.real.astype(np.float32))
    np.save(out_dir / "stft_imag.npy", stft.imag.astype(np.float32))

    # Mel + MFCC
    mel = librosa.feature.melspectrogram(y=sine, sr=sr, n_fft=512, hop_length=128, n_mels=40)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), sr=sr, n_mfcc=13)
    np.save(out_dir / "mel.npy", mel.astype(np.float32))
    np.save(out_dir / "mfcc.npy", mfcc.astype(np.float32))

    # Basic features
    zcr = librosa.feature.zero_crossing_rate(y=sine)[0]
    rms = librosa.feature.rms(y=sine)[0]
    np.save(out_dir / "zcr.npy", zcr.astype(np.float32))
    np.save(out_dir / "rms.npy", rms.astype(np.float32))

    # Spectral contrast
    contrast = librosa.feature.spectral_contrast(y=sine, sr=sr, n_fft=512, hop_length=128, n_bands=6, fmin=200.0)
    np.save(out_dir / "contrast.npy", contrast.astype(np.float32))

    # Chroma STFT
    chroma = librosa.feature.chroma_stft(y=sine, sr=sr, n_fft=512, hop_length=128, n_chroma=12, tuning=0.0)
    np.save(out_dir / "chroma.npy", chroma.astype(np.float32))

    # New spectral features
    S = np.abs(librosa.stft(sine, n_fft=512, hop_length=128))
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr, n_fft=512, hop_length=128)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr, n_fft=512, hop_length=128)[0]
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, n_fft=512, hop_length=128, roll_percent=0.85)[0]
    flatness = librosa.feature.spectral_flatness(S=S)[0]
    np.save(out_dir / "spectral_centroid.npy", centroid.astype(np.float32))
    np.save(out_dir / "spectral_bandwidth.npy", bandwidth.astype(np.float32))
    np.save(out_dir / "spectral_rolloff.npy", rolloff.astype(np.float32))
    np.save(out_dir / "spectral_flatness.npy", flatness.astype(np.float32))

    # dB conversions
    power_spec = mel
    db_power = librosa.power_to_db(power_spec, ref=1.0, amin=1e-10, top_db=80.0)
    np.save(out_dir / "power_to_db.npy", db_power.astype(np.float32))

    amp_spec = S
    db_amp = librosa.amplitude_to_db(amp_spec, ref=1.0, amin=1e-10, top_db=80.0)
    np.save(out_dir / "amplitude_to_db.npy", db_amp.astype(np.float32))

    # Signal generators
    clicks = librosa.clicks(times=[0.0, 0.5, 1.0], sr=sr, length=sr * 2, click_duration=0.01, click_freq=1000.0)
    tone = librosa.tone(frequency=440.0, sr=sr, duration=0.5)
    chirp = librosa.chirp(fmin=100.0, fmax=1000.0, sr=sr, duration=0.5)
    np.save(out_dir / "clicks.npy", clicks.astype(np.float32))
    np.save(out_dir / "tone.npy", tone.astype(np.float32))
    np.save(out_dir / "chirp.npy", chirp.astype(np.float32))

    # Tempogram
    onset_env = librosa.onset.onset_strength(y=sine, sr=sr, n_fft=512, hop_length=128)
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=128, win_length=384)
    np.save(out_dir / "onset_strength.npy", onset_env.astype(np.float32))
    np.save(out_dir / "tempogram.npy", tempogram.astype(np.float32))

    # Autocorrelation
    acf = librosa.autocorrelate(sine[:1000])
    np.save(out_dir / "autocorrelate.npy", acf.astype(np.float32))

    # Utilities
    normalized_l2 = librosa.util.normalize(sine[:100], norm=2)
    np.save(out_dir / "normalize_l2.npy", normalized_l2.astype(np.float32))

    # Metadata for reproducibility
    (out_dir / "meta.txt").write_text(
        "\n".join(
            [
                f"sr={sr}",
                "n_fft=512",
                "hop_length=128",
                "win_length=512",
                "n_mels=40",
                "n_mfcc=13",
                "n_bands=6",
                "fmin=200.0",
                "signal=sine(440Hz,1s) + noise(random,1s)",
            ]
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
