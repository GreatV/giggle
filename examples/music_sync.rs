//! Music Synchronization with Dynamic Time Warping Example
//!
//! This example demonstrates the use of dynamic time warping (DTW) for music
//! synchronization. We align two versions of the same musical phrase played
//! at different tempos.
//!
//! Based on librosa's plot_music_sync.py example.

use giggle::feature::chroma::chroma_cqt;
use giggle::io;
use giggle::utils;
use log::info;
use ndarray::Array2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    info!("Music Synchronization with DTW");

    let sr = 22050;
    let hop_length = 1024;

    // Generate two versions of the same phrase at different tempos
    info!("Generating Test Signals");

    // Musical phrase: C - E - G - C (ascending arpeggio)
    let phrase_slow = create_phrase(1.0, sr); // 1x speed
    let phrase_fast = create_phrase(0.7, sr); // ~1.43x speed (faster)

    info!(
        "Slow version: {} samples ({:.2}s)",
        phrase_slow.len(),
        phrase_slow.len() as f32 / sr as f32
    );
    info!(
        "Fast version: {} samples ({:.2}s)",
        phrase_fast.len(),
        phrase_fast.len() as f32 / sr as f32
    );

    // Extract Chroma Features
    info!("Extracting Chroma Features");

    let chroma_slow = chroma_cqt(
        &phrase_slow,
        sr,
        hop_length,
        27.5,
        12,
        7,
        36,
        0.0,
        2,
        0.0,
        "full",
    )?;
    let chroma_fast = chroma_cqt(
        &phrase_fast,
        sr,
        hop_length,
        27.5,
        12,
        7,
        36,
        0.0,
        2,
        0.0,
        "full",
    )?;

    info!("Chroma shape (slow): {:?}", chroma_slow.shape());
    info!("Chroma shape (fast): {:?}", chroma_fast.shape());

    // Convert to standard 2D arrays for DTW
    let n_frames_slow = chroma_slow.shape()[1];
    let n_frames_fast = chroma_fast.shape()[1];

    let mut chroma_slow_2d = Array2::zeros((12, n_frames_slow));
    let mut chroma_fast_2d = Array2::zeros((12, n_frames_fast));

    for i in 0..12 {
        for j in 0..n_frames_slow {
            chroma_slow_2d[(i, j)] = chroma_slow[(i, j)];
        }
        for j in 0..n_frames_fast {
            chroma_fast_2d[(i, j)] = chroma_fast[(i, j)];
        }
    }

    // DTW Alignment
    info!("DTW Alignment");

    // Compute DTW with cosine distance (good for chroma features)
    let (distance, warp_path) = utils::dtw(&chroma_slow_2d, &chroma_fast_2d, "cosine");

    info!("Warp path length: {} points", warp_path.len());
    info!("DTW distance: {:.4}", distance);

    // Analyze Warping Path
    info!("Warping Path Analysis");

    // Show first and last few points
    info!("First 10 warp path points (slow -> fast frames):");
    for (i, (x, y)) in warp_path.iter().take(10).enumerate() {
        let time_slow = *x as f32 * hop_length as f32 / sr as f32;
        let time_fast = *y as f32 * hop_length as f32 / sr as f32;
        info!(
            "  Step {}: Slow[{}] ({:.3}s) -> Fast[{}] ({:.3}s)",
            i, x, time_slow, y, time_fast
        );
    }

    info!("\nLast 10 warp path points:");
    for (i, (x, y)) in warp_path.iter().rev().take(10).rev().enumerate() {
        let idx = warp_path.len() - 10 + i;
        let time_slow = *x as f32 * hop_length as f32 / sr as f32;
        let time_fast = *y as f32 * hop_length as f32 / sr as f32;
        info!(
            "  Step {}: Slow[{}] ({:.3}s) -> Fast[{}] ({:.3}s)",
            idx, x, time_slow, y, time_fast
        );
    }

    // Compute time stretching factors along the path
    info!("\nTime stretching factors (sampled every 10 points):");
    for i in (0..warp_path.len().saturating_sub(1)).step_by(10) {
        let (x1, y1) = warp_path[i];
        let (x2, y2) = warp_path[(i + 10).min(warp_path.len() - 1)];

        let dx = (x2 - x1) as f32;
        let dy = (y2 - y1) as f32;

        if dy > 0.0 {
            let stretch = dx / dy;
            info!("  At step {}: {:.3}x (slow time / fast time)", i, stretch);
        }
    }

    // Different Distance Metrics
    info!("Comparison of Distance Metrics");

    let metrics = ["euclidean", "manhattan", "cosine"];

    for metric in &metrics {
        let (dist, _) = utils::dtw(&chroma_slow_2d, &chroma_fast_2d, metric);
        info!("  {}: {:.4}", metric, dist);
    }

    // Applications

    Ok(())
}

/// Create a musical phrase (C - E - G - C arpeggio)
fn create_phrase(tempo_factor: f32, sr: u32) -> Vec<f32> {
    let note_duration = 0.5 * tempo_factor;
    let notes = [261.63, 329.63, 392.00, 523.25]; // C4, E4, G4, C5

    let mut phrase = Vec::new();

    for freq in &notes {
        let tone = io::tone(*freq, sr, note_duration);
        phrase.extend_from_slice(&tone);
    }

    phrase
}
