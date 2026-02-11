//! Audio Trimming and Splitting Example
//!
//! This example demonstrates audio trimming and splitting operations.

use giggle::effects::trim;
use giggle::io;
use log::info;

fn main() {
    env_logger::init();
    info!("Audio Trimming and Splitting");

    // Generate a signal with silence at the beginning and end
    let sr = 22050;
    let tone_duration = 2.0;
    let silence_duration = 1.0;

    info!("Generating signal with leading/trailing silence...");
    // Leading silence
    let leading_silence = vec![0.0f32; (sr as f32 * silence_duration) as usize];

    // Middle tone
    let tone = io::tone(440.0, sr, tone_duration);

    // Trailing silence
    let trailing_silence = vec![0.0f32; (sr as f32 * silence_duration) as usize];

    // Combine
    let mut signal = leading_silence.clone();
    signal.extend(tone);
    signal.extend(trailing_silence);

    info!("Original signal: {} samples", signal.len());

    // Trim Silence
    info!("Trim Silence");

    let (trimmed, (start, end)) = trim::trim(&signal, 20.0);

    info!("Trimmed signal: {} samples", trimmed.len());
    info!("  - Start index: {}", start);
    info!("  - End index: {}", end);
    info!("  - Removed: {} samples", signal.len() - trimmed.len());

    // Split at Silence
    info!("Split at Silence");

    // Create a signal with multiple segments
    let mut multi_segment = Vec::new();
    multi_segment.extend(io::tone(440.0, sr, 0.5));
    multi_segment.extend(vec![0.0f32; (sr as f32 * 0.3) as usize]);
    multi_segment.extend(io::tone(554.0, sr, 0.5));
    multi_segment.extend(vec![0.0f32; (sr as f32 * 0.3) as usize]);
    multi_segment.extend(io::tone(659.0, sr, 0.5));

    info!("Multi-segment signal: {} samples", multi_segment.len());

    let segments = trim::split(&multi_segment, 20.0, (sr as f32 * 0.2) as usize);

    info!("Split into {} segments:", segments.len());
    for (i, (start, end)) in segments.iter().enumerate() {
        let duration = (*end - *start) as f32 / sr as f32;
        info!(
            "  Segment {}: samples {}-{} ({:.2} seconds)",
            i + 1,
            start,
            end,
            duration
        );
    }
}
