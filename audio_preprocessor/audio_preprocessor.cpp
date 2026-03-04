/**
 * @file audio_preprocessor.cpp
 * @brief Audio feature extraction using the official TFLite Micro audio
 *        preprocessor model with Signal Library custom ops.
 *
 * The preprocessor .tflite model (from the esp-tflite-micro micro_speech
 * example) performs: Window → FFT → FilterBank → PCAN → Log → INT8 output.
 * We simply feed it 480 raw PCM int16 samples and read out 40 INT8 features.
 */

#include "audio_preprocessor.h"
#include "audio_preprocessor_int8_model_data.h"

#include <cstring>
#include <algorithm>

#include "esp_log.h"
#include "esp_heap_caps.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"

static const char* TAG = "AudioPreprocessor";

// Arena size for the preprocessor model (16 KB is sufficient)
static constexpr size_t kPreprocessorArenaSize = 16 * 1024;

// ---------------------------------------------------------------
//  Register the ops needed by the audio preprocessor model
// ---------------------------------------------------------------
static TfLiteStatus RegisterPreprocessorOps(PreprocessorOpResolver& op_resolver)
{
    TF_LITE_ENSURE_STATUS(op_resolver.AddReshape());
    TF_LITE_ENSURE_STATUS(op_resolver.AddCast());
    TF_LITE_ENSURE_STATUS(op_resolver.AddStridedSlice());
    TF_LITE_ENSURE_STATUS(op_resolver.AddConcatenation());
    TF_LITE_ENSURE_STATUS(op_resolver.AddMul());
    TF_LITE_ENSURE_STATUS(op_resolver.AddAdd());
    TF_LITE_ENSURE_STATUS(op_resolver.AddDiv());
    TF_LITE_ENSURE_STATUS(op_resolver.AddMinimum());
    TF_LITE_ENSURE_STATUS(op_resolver.AddMaximum());
    TF_LITE_ENSURE_STATUS(op_resolver.AddWindow());
    TF_LITE_ENSURE_STATUS(op_resolver.AddFftAutoScale());
    TF_LITE_ENSURE_STATUS(op_resolver.AddRfft());
    TF_LITE_ENSURE_STATUS(op_resolver.AddEnergy());
    TF_LITE_ENSURE_STATUS(op_resolver.AddFilterBank());
    TF_LITE_ENSURE_STATUS(op_resolver.AddFilterBankSquareRoot());
    TF_LITE_ENSURE_STATUS(op_resolver.AddFilterBankSpectralSubtraction());
    TF_LITE_ENSURE_STATUS(op_resolver.AddPCAN());
    TF_LITE_ENSURE_STATUS(op_resolver.AddFilterBankLog());
    return kTfLiteOk;
}

// ---------------------------------------------------------------
//  Constructor / Destructor
// ---------------------------------------------------------------
AudioPreprocessor::AudioPreprocessor() = default;

AudioPreprocessor::~AudioPreprocessor()
{
    if (initialized_) {
        heap_caps_free(ring_buffer_);
        heap_caps_free(preproc_arena_);
        // interpreter was placement-constructed, no delete needed
        initialized_ = false;
    }
}

// ---------------------------------------------------------------
//  Init
// ---------------------------------------------------------------
bool AudioPreprocessor::Init()
{
    ESP_LOGI(TAG, "Initialising audio preprocessor (official TFLite model)");

    // ── Load the preprocessor .tflite model ──────────────────
    preproc_model_ = tflite::GetModel(g_audio_preprocessor_int8_tflite);
    if (preproc_model_->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Preprocessor model schema version %lu != expected %d",
                 preproc_model_->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }

    // ── Allocate tensor arena in PSRAM ───────────────────────
    preproc_arena_ = static_cast<uint8_t*>(
        heap_caps_malloc(kPreprocessorArenaSize,
                         MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    if (!preproc_arena_) {
        ESP_LOGE(TAG, "Failed to allocate preprocessor arena (%u bytes)",
                 (unsigned)kPreprocessorArenaSize);
        return false;
    }

    // ── Register ops & build interpreter ─────────────────────
    static PreprocessorOpResolver op_resolver;
    if (RegisterPreprocessorOps(op_resolver) != kTfLiteOk) {
        ESP_LOGE(TAG, "Failed to register preprocessor ops");
        return false;
    }

    static tflite::MicroInterpreter static_interpreter(
        preproc_model_, op_resolver, preproc_arena_, kPreprocessorArenaSize);
    preproc_interpreter_ = &static_interpreter;

    if (preproc_interpreter_->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "Preprocessor AllocateTensors() failed");
        return false;
    }

    ESP_LOGI(TAG, "Preprocessor model arena used: %u / %u bytes",
             (unsigned)preproc_interpreter_->arena_used_bytes(),
             (unsigned)kPreprocessorArenaSize);

    // ── Allocate ring buffer for sample accumulation ─────────
    ring_buffer_ = static_cast<int16_t*>(
        heap_caps_calloc(kWindowSamples, sizeof(int16_t),
                         MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    if (!ring_buffer_) {
        ESP_LOGE(TAG, "Failed to allocate ring buffer");
        return false;
    }
    ring_buffer_write_ = 0;

    initialized_ = true;

    ESP_LOGI(TAG, "  Window:  %d ms (%d samples)", kFeatureDurationMs, kWindowSamples);
    ESP_LOGI(TAG, "  Stride:  %d ms (%d samples)", kFeatureStepSizeMs, kStrideSamples);
    ESP_LOGI(TAG, "  Output:  %d INT8 mel features per slice", kPreprocessorFeatureSize);
    ESP_LOGI(TAG, "AudioPreprocessor ready");
    return true;
}

// ---------------------------------------------------------------
//  Reset
// ---------------------------------------------------------------
void AudioPreprocessor::Reset()
{
    if (initialized_) {
        ring_buffer_write_ = 0;
        memset(ring_buffer_, 0, kWindowSamples * sizeof(int16_t));
    }
}

// ---------------------------------------------------------------
//  ProcessSamples
// ---------------------------------------------------------------
bool AudioPreprocessor::ProcessSamples(const int16_t* samples,
                                       int            num_samples,
                                       int8_t*        features_out)
{
    if (!initialized_ || num_samples <= 0) return false;

    // Accumulate samples into ring buffer
    for (int i = 0; i < num_samples; ++i) {
        ring_buffer_[ring_buffer_write_++] = samples[i];

        // When we have enough samples for a full window, compute features
        if (ring_buffer_write_ >= kWindowSamples) {
            bool result = ComputeFeatures(ring_buffer_, features_out);

            // Shift buffer by stride amount (keep overlap for next window)
            int samples_to_keep = kWindowSamples - kStrideSamples;
            memmove(ring_buffer_, ring_buffer_ + kStrideSamples,
                    samples_to_keep * sizeof(int16_t));
            ring_buffer_write_ = samples_to_keep;

            return result;
        }
    }

    return false;  // Not enough samples yet
}

// ---------------------------------------------------------------
//  ComputeFeatures — run the TFLite preprocessor model
// ---------------------------------------------------------------
bool AudioPreprocessor::ComputeFeatures(const int16_t* window_samples,
                                        int8_t*        features_out)
{
    // Copy PCM data into the preprocessor model's input tensor
    TfLiteTensor* input = preproc_interpreter_->input(0);
    std::copy_n(window_samples, kWindowSamples,
                tflite::GetTensorData<int16_t>(input));

    // Run the preprocessor model
    if (preproc_interpreter_->Invoke() != kTfLiteOk) {
        ESP_LOGE(TAG, "Preprocessor model invocation failed");
        return false;
    }

    // Copy INT8 features from the output tensor
    TfLiteTensor* output = preproc_interpreter_->output(0);
    std::copy_n(tflite::GetTensorData<int8_t>(output),
                kPreprocessorFeatureSize,
                features_out);

    return true;
}
