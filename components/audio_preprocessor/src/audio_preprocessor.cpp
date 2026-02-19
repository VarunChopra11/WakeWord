/**
 * @file audio_preprocessor.cpp
 * @brief Micro Speech–style log-mel feature extraction for MicroWakeWord.
 *
 * All large buffers are allocated from Octal PSRAM.  The pipeline is
 * deliberately kept in a single sequential path to maximise cache locality:
 *
 *   PCM ring-buffer  →  Hann window  →  512-pt FFT  →
 *   power spectrum   →  40-band mel filterbank  →
 *   log1p            →  INT8 quantisation
 *
 * One feature slice (40 × int8) is produced every 10 ms (= 160 samples).
 */

#include "audio_preprocessor.h"
#include "fft_util.h"
#include "mel_filterbank.h"

#include <cmath>
#include <cstring>
#include <algorithm>

#include "esp_log.h"
#include "esp_heap_caps.h"

static const char* TAG = "AudioPreprocessor";

// ─────────────────────────────────────────────────────────────
//  Constructor / Destructor
// ─────────────────────────────────────────────────────────────
AudioPreprocessor::AudioPreprocessor()  = default;
AudioPreprocessor::~AudioPreprocessor()
{
    heap_caps_free(ring_buf_);
    heap_caps_free(fft_in_);
    heap_caps_free(fft_out_);
    heap_caps_free(window_coeff_);
    heap_caps_free(power_spec_);
    heap_caps_free(mel_energies_);
    fft_deinit();
    mel_filterbank_deinit();
}

// ─────────────────────────────────────────────────────────────
//  Init
// ─────────────────────────────────────────────────────────────
bool AudioPreprocessor::Init()
{
    ESP_LOGI(TAG, "Initialising audio preprocessor");
    ESP_LOGI(TAG, "  Window: %d ms (%d samples)", kWindowMs, kWindowSamples);
    ESP_LOGI(TAG, "  Stride: %d ms (%d samples)", kStrideMs_pp, kStrideSamples_pp);
    ESP_LOGI(TAG, "  FFT:    %d points → %d bins", kFftSize, kFftBins);
    ESP_LOGI(TAG, "  Mels:   %d channels", kNumMels);

    // ── Allocate ring buffer in PSRAM ────────────────────────
    ring_buf_ = static_cast<int16_t*>(
        heap_caps_calloc(kWindowSamples, sizeof(int16_t),
                         MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    if (!ring_buf_) {
        ESP_LOGE(TAG, "ring_buf_ allocation failed");
        return false;
    }

    // ── Allocate FFT buffers in PSRAM ────────────────────────
    // fft_out is complex: 2 × kFftSize floats
    fft_in_  = static_cast<float*>(
        heap_caps_calloc(kFftSize * 2, sizeof(float),
                         MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    fft_out_ = fft_in_;   // We perform in-place FFT

    window_coeff_ = static_cast<float*>(
        heap_caps_malloc(kWindowSamples * sizeof(float),
                         MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));

    power_spec_ = static_cast<float*>(
        heap_caps_calloc(kFftBins, sizeof(float),
                         MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));

    mel_energies_ = static_cast<float*>(
        heap_caps_calloc(kNumMels, sizeof(float),
                         MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));

    if (!fft_in_ || !window_coeff_ || !power_spec_ || !mel_energies_) {
        ESP_LOGE(TAG, "Buffer allocation failed");
        return false;
    }

    // ── Pre-compute Hann window coefficients ─────────────────
    ComputeWindow();

    // ── Initialise FFT (builds twiddle-factor table) ─────────
    if (!fft_init(kFftSize)) {
        ESP_LOGE(TAG, "FFT init failed");
        return false;
    }

    // ── Initialise mel filterbank ────────────────────────────
    if (!mel_filterbank_init()) {
        ESP_LOGE(TAG, "Mel filterbank init failed");
        return false;
    }

    ESP_LOGI(TAG, "AudioPreprocessor ready");
    return true;
}

// ─────────────────────────────────────────────────────────────
//  Reset
// ─────────────────────────────────────────────────────────────
void AudioPreprocessor::Reset()
{
    memset(ring_buf_, 0, kWindowSamples * sizeof(int16_t));
    ring_head_               = 0;
    ring_fill_               = 0;
    samples_since_last_slice_ = 0;
}

// ─────────────────────────────────────────────────────────────
//  ProcessSamples — main entry point
// ─────────────────────────────────────────────────────────────
bool AudioPreprocessor::ProcessSamples(const int16_t* samples,
                                       int            num_samples,
                                       int8_t*        features_out)
{
    // ── Write samples into circular ring buffer ───────────────
    for (int i = 0; i < num_samples; ++i) {
        ring_buf_[ring_head_] = samples[i];
        ring_head_            = (ring_head_ + 1) % kWindowSamples;
        if (ring_fill_ < kWindowSamples) ++ring_fill_;
    }

    samples_since_last_slice_ += num_samples;

    // ── Check whether we have enough data for a new slice ────
    if (ring_fill_ < kWindowSamples) {
        return false;   // Still buffering initial audio
    }

    if (samples_since_last_slice_ < kStrideSamples_pp) {
        return false;   // Not yet at stride boundary
    }

    // ── Produce one feature slice ────────────────────────────
    samples_since_last_slice_ = 0;

    // 1. Apply Hann window and copy into FFT input buffer
    ApplyWindowAndFFT(ring_head_);

    // 2. Execute 512-pt real FFT
    fft_real_forward(fft_in_, kFftSize);

    // 3. Compute one-sided power spectrum
    ComputePowerSpectrum();

    // 4. Apply 40-band mel filterbank
    memset(mel_energies_, 0, kNumMels * sizeof(float));
    mel_filterbank_apply(power_spec_, mel_energies_);

    // 5. Log compression + INT8 quantisation
    QuantiseToInt8(features_out);

    return true;
}

// ─────────────────────────────────────────────────────────────
//  Private helpers
// ─────────────────────────────────────────────────────────────

/** Pre-compute symmetric Hann window coefficients. */
void AudioPreprocessor::ComputeWindow()
{
    for (int i = 0; i < kWindowSamples; ++i) {
        window_coeff_[i] = 0.5f - 0.5f * cosf(
            (2.0f * static_cast<float>(M_PI) * i) /
            (static_cast<float>(kWindowSamples) - 1.0f));
    }
}

/**
 * Copy kWindowSamples from ring buffer (starting at oldest sample),
 * multiply by Hann window, and place in fft_in_ with zero-padding.
 *
 * fft_in_ layout: [re0, im0, re1, im1, …]
 * im is always 0 for real input.
 */
void AudioPreprocessor::ApplyWindowAndFFT(int ring_offset)
{
    // Zero the entire complex buffer (handles zero-padding + imaginary parts)
    memset(fft_in_, 0, kFftSize * 2 * sizeof(float));

    // Oldest sample is at ring_head_ (the write pointer which is now stale)
    // ring_head_ currently points to the location just written — the oldest
    // sample is one step ahead in the circular sense.
    int read_pos = ring_head_ % kWindowSamples;

    for (int i = 0; i < kWindowSamples; ++i) {
        float sample_f = static_cast<float>(ring_buf_[read_pos]) /
                         32768.0f;   // Normalise to [-1, +1)
        fft_in_[2 * i]     = sample_f * window_coeff_[i]; // real
        fft_in_[2 * i + 1] = 0.0f;                        // imag
        read_pos = (read_pos + 1) % kWindowSamples;
    }
    // Bins kWindowSamples…kFftSize-1 remain zero (zero-padding)
}

/** Compute magnitude-squared spectrum from complex FFT output. */
void AudioPreprocessor::ComputePowerSpectrum()
{
    for (int k = 0; k < kFftBins; ++k) {
        float re = fft_out_[2 * k];
        float im = fft_out_[2 * k + 1];
        power_spec_[k] = re * re + im * im;
    }
}

/**
 * Apply log1p compression to mel band energies and quantise to INT8.
 *
 * Compression: y = log10(1 + energy)  — avoids log(0)
 * Quantisation: int8 = clamp(round(y / scale) + zero_point, -128, 127)
 */
void AudioPreprocessor::QuantiseToInt8(int8_t* out)
{
    // Find dynamic range for per-frame normalisation (optional stabiliser)
    float max_energy = 1e-6f;  // Prevent division by zero
    for (int i = 0; i < kNumMels; ++i) {
        if (mel_energies_[i] > max_energy) max_energy = mel_energies_[i];
    }

    for (int i = 0; i < kNumMels; ++i) {
        // Log compression with a floor at kFeatureScale to avoid -∞
        float log_energy = log10f(1.0f + mel_energies_[i]);

        // Scale and shift to INT8 range (matched to model's input quantisation)
        float scaled = (log_energy / kFeatureScale) + kFeatureZp;

        // Clamp to [-128, 127]
        int rounded = static_cast<int>(scaled + 0.5f);
        if (rounded >  127) rounded =  127;
        if (rounded < -128) rounded = -128;
        out[i] = static_cast<int8_t>(rounded);
    }
}
