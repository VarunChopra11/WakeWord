/**
 * @file audio_preprocessor.h
 * @brief TFLite Micro Frontend wrapper for MicroWakeWord models.
 *
 * Uses the exact same audio preprocessing pipeline as ESPHome's
 * micro_wake_word component (kahrendt/ESPMicroSpeechFeatures):
 *
 *   PCM -> Window -> FFT -> Filterbank -> Noise Reduction ->
 *   PCAN Gain Control -> Log Scale -> INT8 quantisation
 *
 * All parameters match the ESPHome preprocessor_settings.h exactly,
 * ensuring feature compatibility with microWakeWord models.
 */

#pragma once

#include <cstdint>
#include <cstddef>

// TFLite Micro Frontend (from ESPMicroSpeechFeatures)
#include "frontend.h"
#include "frontend_util.h"

// ---------------------------------------------------------------
//  Feature extraction constants (matched to ESPHome exactly)
// ---------------------------------------------------------------
static constexpr int kAudioSampleRate        = 16000;
static constexpr int kFeatureDurationMs      = 30;   // Window
static constexpr int kFeatureStepSizeMs      = 20;   // Stride
static constexpr int kPreprocessorFeatureSize = 40;   // Mel channels

// ---------------------------------------------------------------
//  Class
// ---------------------------------------------------------------
class AudioPreprocessor
{
public:
    AudioPreprocessor();
    ~AudioPreprocessor();

    bool Init();

    /**
     * @brief Feed PCM samples; optionally produce one INT8 feature slice.
     * @return true when features_out has been filled with kPreprocessorFeatureSize values.
     */
    bool ProcessSamples(const int16_t* samples,
                        int            num_samples,
                        int8_t*        features_out);

    void Reset();

    /** Kept for API compat — not needed with Frontend pipeline. */
    void SetQuantizationParams(float scale, int zero_point);

private:
    struct FrontendConfig frontend_config_;
    struct FrontendState  frontend_state_;
    bool                  initialized_ = false;
};
