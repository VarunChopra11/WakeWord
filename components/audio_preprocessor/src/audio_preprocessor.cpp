/**
 * @file audio_preprocessor.cpp
 * @brief TFLite Micro Frontend wrapper — exact ESPHome-compatible preprocessing.
 *
 * Replaces the custom log-mel pipeline with the same TFLite Micro Frontend
 * used by ESPHome's micro_wake_word component, including:
 *   - Noise reduction (spectral subtraction)
 *   - PCAN gain control (per-channel adaptive normalisation)
 *   - Log scaling with scale_shift=6
 *
 * The INT8 conversion uses the exact formula from ESPHome:
 *   int8 = clamp((frontend_uint16 * 256 / 666) - 128, -128, 127)
 */

#include "audio_preprocessor.h"

#include <cstring>
#include <algorithm>

#include "esp_log.h"

static const char* TAG = "AudioPreprocessor";

// ---------------------------------------------------------------
//  Constructor / Destructor
// ---------------------------------------------------------------
AudioPreprocessor::AudioPreprocessor() = default;

AudioPreprocessor::~AudioPreprocessor()
{
    if (initialized_) {
        FrontendFreeStateContents(&frontend_state_);
        initialized_ = false;
    }
}

// ---------------------------------------------------------------
//  Init
// ---------------------------------------------------------------
bool AudioPreprocessor::Init()
{
    ESP_LOGI(TAG, "Initialising audio preprocessor (TFLite Micro Frontend)");

    // Fill config with ESPHome-compatible settings
    // (matching esphome/components/micro_wake_word/preprocessor_settings.h)
    memset(&frontend_config_, 0, sizeof(frontend_config_));

    // Window
    frontend_config_.window.size_ms = kFeatureDurationMs;      // 30 ms
    frontend_config_.window.step_size_ms = kFeatureStepSizeMs; // 20 ms

    // Filterbank
    frontend_config_.filterbank.num_channels    = kPreprocessorFeatureSize; // 40
    frontend_config_.filterbank.lower_band_limit = 125.0f;
    frontend_config_.filterbank.upper_band_limit = 7500.0f;

    // Noise reduction
    frontend_config_.noise_reduction.smoothing_bits        = 10;
    frontend_config_.noise_reduction.even_smoothing        = 0.025f;
    frontend_config_.noise_reduction.odd_smoothing         = 0.06f;
    frontend_config_.noise_reduction.min_signal_remaining  = 0.05f;

    // PCAN gain control
    frontend_config_.pcan_gain_control.enable_pcan = 1;
    frontend_config_.pcan_gain_control.strength    = 0.95f;
    frontend_config_.pcan_gain_control.offset      = 80.0f;
    frontend_config_.pcan_gain_control.gain_bits   = 21;

    // Log scale
    frontend_config_.log_scale.enable_log  = 1;
    frontend_config_.log_scale.scale_shift = 6;

    // Populate the Frontend state
    if (!FrontendPopulateState(&frontend_config_, &frontend_state_, kAudioSampleRate)) {
        ESP_LOGE(TAG, "FrontendPopulateState() failed");
        return false;
    }

    initialized_ = true;

    ESP_LOGI(TAG, "  Window:  %d ms, stride %d ms", kFeatureDurationMs, kFeatureStepSizeMs);
    ESP_LOGI(TAG, "  Mels:    %d channels, %.0f-%.0f Hz", kPreprocessorFeatureSize,
             frontend_config_.filterbank.lower_band_limit,
             frontend_config_.filterbank.upper_band_limit);
    ESP_LOGI(TAG, "  Noise reduction: ON  (smoothing_bits=%d)",
             frontend_config_.noise_reduction.smoothing_bits);
    ESP_LOGI(TAG, "  PCAN:    ON  (strength=%.2f, offset=%.0f, gain_bits=%d)",
             frontend_config_.pcan_gain_control.strength,
             frontend_config_.pcan_gain_control.offset,
             frontend_config_.pcan_gain_control.gain_bits);
    ESP_LOGI(TAG, "  Log:     ON  (scale_shift=%d)", frontend_config_.log_scale.scale_shift);
    ESP_LOGI(TAG, "AudioPreprocessor ready");
    return true;
}

// ---------------------------------------------------------------
//  Reset
// ---------------------------------------------------------------
void AudioPreprocessor::Reset()
{
    if (initialized_) {
        FrontendReset(&frontend_state_);
    }
}

// ---------------------------------------------------------------
//  SetQuantizationParams (kept for API compat — not used)
// ---------------------------------------------------------------
void AudioPreprocessor::SetQuantizationParams(float scale, int zero_point)
{
    ESP_LOGI(TAG, "Quantization params (info only): scale=%f, zero_point=%d",
             scale, zero_point);
    // Not used — Frontend + ESPHome formula handles quantisation directly.
}

// ---------------------------------------------------------------
//  ProcessSamples
// ---------------------------------------------------------------
bool AudioPreprocessor::ProcessSamples(const int16_t* samples,
                                       int            num_samples,
                                       int8_t*        features_out)
{
    if (!initialized_ || num_samples <= 0) return false;

    size_t num_samples_read = 0;

    // Feed audio into the Frontend — it internally handles windowing,
    // stride, FFT, filterbank, noise reduction, PCAN, and log scaling.
    struct FrontendOutput frontend_output = FrontendProcessSamples(
        &frontend_state_,
        samples,
        static_cast<size_t>(num_samples),
        &num_samples_read
    );

    // If not enough samples accumulated for a feature slice, return false
    if (frontend_output.size == 0 || frontend_output.values == NULL) {
        return false;
    }

    // Convert Frontend uint16 output to INT8 using ESPHome's exact formula:
    //
    //   "The feature pipeline outputs 16-bit signed integers in roughly a
    //    0 to 670 range. In training, these are then arbitrarily divided
    //    by 25.6 to get float values in the rough range of 0.0 to 26.0."
    //
    //   input = (feature * 256) / (25.6 * 26.0) - 128
    //   Simplified: input = (feature * 256 / 666) - 128
    //
    static constexpr int32_t value_scale = 256;
    static constexpr int32_t value_div   = 666;   // 25.6 * 26.0 rounded

    size_t features_to_copy = std::min(frontend_output.size,
                                       static_cast<size_t>(kPreprocessorFeatureSize));

    for (size_t i = 0; i < features_to_copy; ++i) {
        int32_t value = ((static_cast<int32_t>(frontend_output.values[i]) * value_scale)
                         + (value_div / 2)) / value_div;
        value += INT8_MIN;  // subtract 128

        if (value > INT8_MAX) value = INT8_MAX;
        if (value < INT8_MIN) value = INT8_MIN;

        features_out[i] = static_cast<int8_t>(value);
    }

    // Zero-fill remaining channels if Frontend produced fewer than expected
    for (size_t i = features_to_copy; i < static_cast<size_t>(kPreprocessorFeatureSize); ++i) {
        features_out[i] = INT8_MIN;
    }

    return true;
}
