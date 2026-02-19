/**
 * @file audio_preprocessor.h
 * @brief Micro Speech–style audio front-end for MicroWakeWord
 *
 * Converts raw 16-kHz PCM audio into a stream of 40-channel INT8
 * log-mel feature slices, one slice per 10 ms stride.
 *
 * Processing chain:
 *   PCM ring-buffer → Hann window → 512-pt FFT → power spectrum →
 *   40-channel mel filterbank → log1p scaling → INT8 quantisation
 */

#pragma once

#include <cstdint>
#include <cstddef>

// ─────────────────────────────────────────────────────────────
//  Feature extraction constants
// ─────────────────────────────────────────────────────────────
static constexpr int kAudioSampleRate   = 16000;
static constexpr int kWindowMs          = 30;
static constexpr int kStrideMs_pp       = 10;

static constexpr int kWindowSamples     = (kAudioSampleRate * kWindowMs)    / 1000; // 480
static constexpr int kStrideSamples_pp  = (kAudioSampleRate * kStrideMs_pp) / 1000; // 160
static constexpr int kFftSize           = 512;   // next power-of-2 ≥ 480
static constexpr int kFftBins           = kFftSize / 2 + 1;                          // 257
static constexpr int kNumMels           = 40;

// INT8 quantisation parameters (matched to model input expectations)
static constexpr float kFeatureScale    = 1.0f / 52.0f;
static constexpr int   kFeatureZp       = -30;

// ─────────────────────────────────────────────────────────────
//  Class
// ─────────────────────────────────────────────────────────────
class AudioPreprocessor
{
public:
    AudioPreprocessor();
    ~AudioPreprocessor();

    /**
     * @brief Initialise buffers, FFT tables, and mel filterbank.
     * @return true on success.
     */
    bool Init();

    /**
     * @brief Feed new PCM samples and optionally produce one feature slice.
     *
     * @param[in]  samples      Pointer to 16-bit signed PCM samples.
     * @param[in]  num_samples  Number of samples (typically 160).
     * @param[out] features_out Buffer to receive kNumMels INT8 features.
     * @return true when a complete feature slice has been written to features_out.
     */
    bool ProcessSamples(const int16_t* samples,
                        int            num_samples,
                        int8_t*        features_out);

    /** Reset internal ring-buffer and stride counter. */
    void Reset();

private:
    // ── Ring buffer ─────────────────────────────────────────
    int16_t* ring_buf_     = nullptr;   ///< Allocated in PSRAM
    int      ring_head_    = 0;         ///< Write pointer (circular)
    int      ring_fill_    = 0;         ///< Samples currently in buffer

    // ── FFT work buffers (float, PSRAM) ─────────────────────
    float*   fft_in_       = nullptr;   ///< Windowed + zero-padded real part
    float*   fft_out_      = nullptr;   ///< Complex output [re,im, re,im, …]
    float*   window_coeff_ = nullptr;   ///< Pre-computed Hann coefficients
    float*   power_spec_   = nullptr;   ///< Power spectrum (kFftBins floats)

    // ── Mel filterbank (PSRAM) ───────────────────────────────
    float*   mel_energies_ = nullptr;   ///< Per-band accumulator (kNumMels)

    // ── Stride management ────────────────────────────────────
    int      samples_since_last_slice_ = 0;

    // ── Internal helpers ─────────────────────────────────────
    void   ComputeWindow();
    void   ApplyWindowAndFFT(int ring_offset);
    void   ComputePowerSpectrum();
    void   ApplyMelFilterbank();
    void   QuantiseToInt8(int8_t* out);
};
