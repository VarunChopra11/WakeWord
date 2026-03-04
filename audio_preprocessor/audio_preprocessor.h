/**
 * @file audio_preprocessor.h
 * @brief Audio feature extraction for MicroWakeWord models.
 *
 * Uses the official TFLite Micro audio preprocessor model with Signal Library
 * custom ops (Window, FFT, FilterBank, PCAN, Log, etc.).
 *
 *   PCM int16 → [TFLite preprocessor model] → 40 × INT8 mel features
 *
 * The preprocessor .tflite model is embedded from
 * audio_preprocessor_int8_model_data.h (from the esp-tflite-micro micro_speech
 * example).  No custom DSP code is needed.
 */

#pragma once

#include <cstdint>
#include <cstddef>

// TFLite Micro headers
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

// ---------------------------------------------------------------
//  Feature extraction constants  (match the preprocessor model)
// ---------------------------------------------------------------
static constexpr int kAudioSampleRate         = 16000;
static constexpr int kFeatureDurationMs       = 30;    // Window size in ms
static constexpr int kFeatureStepSizeMs       = 20;    // Stride in ms
static constexpr int kPreprocessorFeatureSize = 40;    // Mel channels

// Derived constants
static constexpr int kWindowSamples  = (kAudioSampleRate * kFeatureDurationMs) / 1000;   // 480
static constexpr int kStrideSamples  = (kAudioSampleRate * kFeatureStepSizeMs) / 1000;   // 320

// ---------------------------------------------------------------
//  Op resolver type for the preprocessor model (18 ops)
// ---------------------------------------------------------------
using PreprocessorOpResolver = tflite::MicroMutableOpResolver<18>;

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
     * @return true when features_out has been filled with
     *         kPreprocessorFeatureSize values.
     */
    bool ProcessSamples(const int16_t* samples,
                        int            num_samples,
                        int8_t*        features_out);

    void Reset();

private:
    // Ring buffer for accumulating samples across calls
    int16_t* ring_buffer_       = nullptr;
    int      ring_buffer_write_ = 0;

    // TFLite preprocessor model objects
    const tflite::Model*      preproc_model_       = nullptr;
    tflite::MicroInterpreter* preproc_interpreter_  = nullptr;
    uint8_t*                  preproc_arena_         = nullptr;

    bool initialized_ = false;

    // Internal: run the preprocessor model on a full window of samples
    bool ComputeFeatures(const int16_t* window_samples, int8_t* features_out);
};
