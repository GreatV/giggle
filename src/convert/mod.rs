//! Conversion utilities for audio (time, frequency, note, MIDI).

pub(crate) const A4_HZ: f32 = 440.0;
pub(crate) const MIDI_A4: f32 = 69.0;

mod frequency;
mod intervals;
mod music_theory;
mod pitch;
mod scales;
mod timing;
mod weighting;

pub use frequency::*;
pub use intervals::*;
pub use music_theory::*;
pub use pitch::*;
pub use scales::*;
pub use timing::*;
pub use weighting::*;

#[cfg(test)]
mod tests;
