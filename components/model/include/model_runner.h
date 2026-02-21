/**
 * @file model_runner.h
 * @brief TFLite Micro interpreter wrapper for the stateful MicroWakeWord model.
 *
 * This class owns:
 *   - A 150 KB tensor arena allocated in Octal PSRAM
 *   - A MicroAllocator backed by that arena
 *   - A MicroResourceVariables instance managing 22 VAR_HANDLE state slots
 *   - A MicroInterpreter built with the above allocator + resource variables
 *
 * Critical design rule:
 *   Invoke() is called repeatedly WITHOUT ever re-initialising the interpreter.
 *   The model's VAR_HANDLE ops read and write their states through the
 *   MicroResourceVariables on every call, accumulating temporal context across
 *   audio frames.  Resetting the interpreter destroys this state.
 */

#pragma once

#include <cstdint>
#include <cstddef>

// TFLite Micro headers — provided by espressif/esp-tflite-micro
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_resource_variable.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/schema/schema_generated.h"

// ─────────────────────────────────────────────────────────────
//  Configuration
// ─────────────────────────────────────────────────────────────
static constexpr size_t kTensorArenaSize      = 150 * 1024;  // 150 KB in PSRAM
static constexpr int    kNumResourceVariables = 22;           // VAR_HANDLE slots

class ModelRunner
{
public:
    ModelRunner();
    ~ModelRunner();

    /**
     * @brief Allocate arena, build interpreter, and call AllocateTensors().
     * @return true on success.
     */
    bool Init();

    /**
     * @brief Execute one inference step.
     *
     * The caller must have already written 40 INT8 feature values into the
     * buffer returned by GetInputBuffer() before calling this function.
     *
     * @return true if kTfLiteOk.
     */
    bool Invoke();

    /**
     * @brief Return a pointer to the model's input tensor data.
     *
     * Shape: [1, 1, 40], dtype: INT8.
     * The caller writes 40 bytes here before each Invoke().
     */
    int8_t* GetInputBuffer();

    /**
     * @brief Read the scalar wake-word probability from the output tensor.
     *
     * Shape: [1, 1], dtype: UINT8.
     * Returns 0 on any error.
     */
    uint8_t GetOutputProbability();

private:
    // ── Tensor arena (PSRAM) ─────────────────────────────────
    uint8_t* tensor_arena_ = nullptr;

    // ── TFLite Micro objects ─────────────────────────────────
    tflite::MicroAllocator*       allocator_          = nullptr;
    tflite::MicroResourceVariables* resource_variables_ = nullptr;
    tflite::MicroInterpreter*     interpreter_        = nullptr;
    const tflite::Model*          model_              = nullptr;

    // ── Op resolver — ops used by MicroWakeWord ──────────────
    // We use a fixed-size resolver; tune the template param to the op count.
    using OpResolver = tflite::MicroMutableOpResolver<24>;
    OpResolver* op_resolver_ = nullptr;

    void RegisterOps();
};
