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
//  SetQuantizationParams
// ─────────────────────────────────────────────────────────────
void AudioPreprocessor::SetQuantizationParams(float scale, int zero_point)
{
    if (scale == 0.0f) {
        ESP_LOGW(TAG, "Quantization scale is 0 — defaulting to 1.0");
        scale = 1.0f;
    }
    input_scale_      = scale;
    input_zero_point_ = zero_point;
    ESP_LOGI(TAG, "Quantization params set: scale=%f, zero_point=%d",
             scale, zero_point);
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
        // Keep raw int16 scale — do NOT normalise to [-1,+1).
        // The model was trained with unnormalised int16 audio.
        // ln(mel_energy) of unnormalised samples falls in [0,26]
        // which matches the model's input quantisation range.
        float sample_f = static_cast<float>(ring_buf_[read_pos]);
        fft_in_[2 * i]     = sample_f * window_coeff_[i]; // real
        fft_in_[2 * i + 1] = 0.0f;                        // imag
        read_pos = (read_pos + 1) % kWindowSamples;
    }
    // Bins kWindowSamples…kFftSize-1 remain zero (zero-padding)
}

/**
 * Compute magnitude-squared spectrum from complex FFT output.
 *
 * After dsps_cplx2reC_fc32 the layout is:
 *   data[0]         = Re(X[0])     (DC, purely real)
 *   data[1]         = Re(X[N/2])   (Nyquist, purely real)
 *   data[2k..2k+1]  = Re(X[k]), Im(X[k])  for k = 1 … N/2-1
 */
void AudioPreprocessor::ComputePowerSpectrum()
{
    // DC bin (purely real)
    power_spec_[0] = fft_out_[0] * fft_out_[0];

    // Bins 1 … N/2-1
    for (int k = 1; k < kFftBins - 1; ++k) {
        float re = fft_out_[2 * k];
        float im = fft_out_[2 * k + 1];
        power_spec_[k] = re * re + im * im;
    }

    // Nyquist bin (purely real, packed at data[1])
    power_spec_[kFftBins - 1] = fft_out_[1] * fft_out_[1];
}

/**
 * Apply log compression to mel band energies and quantise to INT8.
 *
 * Compression: y = ln(max(energy, 1e-6))  — natural log, floor to avoid log(0)
 * Quantisation: int8 = clamp(round(y / scale) + zero_point, -128, 127)
 *               scale and zero_point are read from the model's input tensor.
 */
void AudioPreprocessor::QuantiseToInt8(int8_t* out)
{
    for (int i = 0; i < kNumMels; ++i) {
        // Log compression: natural log with floor to prevent log(0)
        float log_energy = logf(fmaxf(mel_energies_[i], 1e-6f));

        // TFLite INT8 quantisation: q = round(value / scale) + zero_point
        int32_t quantized = static_cast<int32_t>(
            roundf(log_energy / input_scale_)) + input_zero_point_;

        // Clamp to INT8 range [-128, 127]
        if (quantized >  127) quantized =  127;
        if (quantized < -128) quantized = -128;
        out[i] = static_cast<int8_t>(quantized);
    }
}
