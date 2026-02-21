/**
 * @file mel_filterbank.h
 * @brief Pre-computed triangular mel filterbank for 40-channel log-mel features.
 *
 * Follows the HTK convention used by the TFLite Micro Micro Speech pipeline:
 *   - Frequency range  : 300 Hz – 8000 Hz
 *   - Filterbank count : 40
 *   - FFT bins         : 257  (512-pt FFT)
 *   - Sample rate      : 16000 Hz
 *
 * The filterbank is stored as a sparse lookup table mapping each FFT bin to
 * the (up to two) mel bands it contributes to, weighted by triangle overlap.
 * This avoids iterating over every bin×band combination on each frame.
 */

#pragma once

#include <cstdint>
#include <cstddef>

// ── Parameters (must match audio_preprocessor.h) ─────────────
static constexpr int   kMFB_NumBands    = 40;
static constexpr int   kMFB_NumBins     = 257;   // kFftSize/2 + 1
static constexpr int   kMFB_SampleRate  = 16000;
static constexpr float kMFB_LowFreq     = 300.0f;
static constexpr float kMFB_HighFreq    = 8000.0f;

// ── Per-bin contribution descriptor ──────────────────────────
struct MelBinEntry {
    int16_t band;    ///< Mel band index (-1 → unused)
    float   weight;  ///< Triangular filter weight
};

/**
 * @brief Initialise the mel filterbank lookup tables.
 *
 * Allocates two arrays of length kMFB_NumBins in PSRAM.
 * @return true on success.
 */
bool mel_filterbank_init();

/**
 * @brief Apply the filterbank to a power spectrum and accumulate mel energies.
 *
 * @param[in]  power_spectrum  kMFB_NumBins float values (|X[k]|²).
 * @param[out] mel_energies    kMFB_NumBands float values (zeroed on entry).
 */
void mel_filterbank_apply(const float* power_spectrum, float* mel_energies);

/**
 * @brief Release filterbank memory.
 */
void mel_filterbank_deinit();
