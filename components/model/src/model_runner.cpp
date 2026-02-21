/**
 * @file model_runner.cpp
 * @brief Stateful MicroWakeWord TFLite Micro interpreter.
 *
 * Boot sequence:
 *   1. Heap-allocate 150 KB tensor arena in Octal PSRAM.
 *   2. Create MicroAllocator from that arena.
 *   3. Create MicroResourceVariables (22 slots) using the SAME allocator.
 *   4. Build MicroInterpreter with both the allocator and resource variables.
 *   5. Call AllocateTensors() — this initialises all 11 VAR_HANDLE state
 *      buffers (via Subgraph#1 of the model).
 *
 * IMPORTANT: The interpreter is NEVER reset after step 5.  Internal streaming
 * state (READ_VARIABLE / ASSIGN_VARIABLE ops) must persist across all Invoke()
 * calls.  Destroying or re-constructing the interpreter resets the state and
 * breaks wake word detection.
 */

#include "model_runner.h"
#include "model_data.h"   // g_model_data[], g_model_data_len

#include "esp_log.h"
#include "esp_heap_caps.h"

// Required TFLite Micro ops for MicroWakeWord
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_log.h"

static const char* TAG = "ModelRunner";

// ─────────────────────────────────────────────────────────────
//  Constructor / Destructor
// ─────────────────────────────────────────────────────────────
ModelRunner::ModelRunner()  = default;

ModelRunner::~ModelRunner()
{
    // interpreter_ is placement-constructed inside PSRAM — must be explicitly
    // destructed before the arena is freed.
    if (interpreter_) {
        interpreter_->~MicroInterpreter();
        interpreter_ = nullptr;
    }
    if (op_resolver_) {
        delete op_resolver_;
        op_resolver_ = nullptr;
    }
    heap_caps_free(tensor_arena_);
    tensor_arena_ = nullptr;
}

// ─────────────────────────────────────────────────────────────
//  Init
// ─────────────────────────────────────────────────────────────
bool ModelRunner::Init()
{
    ESP_LOGI(TAG, "Initialising ModelRunner");
    ESP_LOGI(TAG, "  Arena:              %zu KB in PSRAM", kTensorArenaSize / 1024);
    ESP_LOGI(TAG, "  Resource variables: %d", kNumResourceVariables);

    // ── 1. Allocate tensor arena in Octal PSRAM ───────────────
    tensor_arena_ = static_cast<uint8_t*>(
        heap_caps_aligned_alloc(16,                  // 16-byte alignment
                                kTensorArenaSize,
                                MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));

    if (!tensor_arena_) {
        ESP_LOGE(TAG, "Failed to allocate %zu KB tensor arena in PSRAM",
                 kTensorArenaSize / 1024);
        return false;
    }
    ESP_LOGI(TAG, "  Arena @ %p (PSRAM)", static_cast<void*>(tensor_arena_));

    // ── 2. Verify model flatbuffer ────────────────────────────
    model_ = tflite::GetModel(g_model_data);
    if (model_->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model schema version mismatch: expected %d, got %lu",
                 TFLITE_SCHEMA_VERSION,
                 static_cast<unsigned long>(model_->version()));
        return false;
    }

    // ── 3. Build op resolver ──────────────────────────────────
    op_resolver_ = new (std::nothrow) OpResolver();
    if (!op_resolver_) {
        ESP_LOGE(TAG, "Op resolver allocation failed");
        return false;
    }
    RegisterOps();

    // ── 4. Create MicroAllocator ──────────────────────────────
    //
    //   Note: MicroAllocator::Create() consumes part of the tensor arena.
    //   We pass the raw arena directly so that both the allocator metadata
    //   AND the resource variable storage come from the same PSRAM region.
    allocator_ = tflite::MicroAllocator::Create(tensor_arena_,
                                                 kTensorArenaSize);
    if (!allocator_) {
        ESP_LOGE(TAG, "MicroAllocator::Create() failed");
        return false;
    }

    // ── 5. Create MicroResourceVariables ─────────────────────
    //
    //   CRITICAL: Must be created BEFORE MicroInterpreter so that the
    //   interpreter can register VAR_HANDLE ops against these slots.
    //   The number of slots (22) must be >= the number of VAR_HANDLE ops
    //   in the model's main subgraph.
    resource_variables_ = tflite::MicroResourceVariables::Create(
        allocator_, kNumResourceVariables);

    if (!resource_variables_) {
        ESP_LOGE(TAG, "MicroResourceVariables::Create() failed");
        return false;
    }
    ESP_LOGI(TAG, "  MicroResourceVariables created (%d slots)",
             kNumResourceVariables);

    // ── 6. Build MicroInterpreter ─────────────────────────────
    //
    //   The interpreter is heap-constructed so that we have explicit
    //   lifetime control.  We deliberately use placement-new into a
    //   region allocated via new so that the vtable lives in IRAM.
    interpreter_ = new (std::nothrow) tflite::MicroInterpreter(
        model_,
        *op_resolver_,
        allocator_,
        resource_variables_
    );

    if (!interpreter_) {
        ESP_LOGE(TAG, "MicroInterpreter allocation failed");
        return false;
    }

    // ── 7. AllocateTensors ────────────────────────────────────
    //
    //   This step runs Subgraph#1 (state initialisation), zeroing all
    //   stream_11/states1 … stream_21/states1 buffers.  It must succeed
    //   exactly once; calling it again would reset the streaming state.
    TfLiteStatus status = interpreter_->AllocateTensors();
    if (status != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors() failed");
        return false;
    }

    // ── 8. Sanity-check tensor shapes ────────────────────────
    TfLiteTensor* input  = interpreter_->input(0);
    TfLiteTensor* output = interpreter_->output(0);

    if (!input || !output) {
        ESP_LOGE(TAG, "Input or output tensor is null");
        return false;
    }

    ESP_LOGI(TAG, "  Input  tensor: shape=[%d,%d,%d] dtype=%d",
             input->dims->data[0], input->dims->data[1], input->dims->data[2],
             input->type);

    ESP_LOGI(TAG, "  Output tensor: shape=[%d,%d] dtype=%d",
             output->dims->data[0], output->dims->data[1],
             output->type);

    // Validate expected shape [1,1,40]
    if (input->dims->size != 3 ||
        input->dims->data[0] != 1 ||
        input->dims->data[1] != 1 ||
        input->dims->data[2] != 40) {
        ESP_LOGE(TAG, "Unexpected input tensor shape");
        return false;
    }

    if (input->type != kTfLiteInt8) {
        ESP_LOGE(TAG, "Unexpected input dtype (expected INT8)");
        return false;
    }

    ESP_LOGI(TAG, "ModelRunner ready — streaming state active");
    return true;
}

// ─────────────────────────────────────────────────────────────
//  Invoke
// ─────────────────────────────────────────────────────────────
bool ModelRunner::Invoke()
{
    TfLiteStatus status = interpreter_->Invoke();
    if (status != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke() returned error: %d", static_cast<int>(status));
        return false;
    }
    return true;
}

// ─────────────────────────────────────────────────────────────
//  Tensor accessors
// ─────────────────────────────────────────────────────────────
int8_t* ModelRunner::GetInputBuffer()
{
    TfLiteTensor* input = interpreter_->input(0);
    if (!input) return nullptr;
    return input->data.int8;
}

uint8_t ModelRunner::GetOutputProbability()
{
    TfLiteTensor* output = interpreter_->output(0);
    if (!output) return 0;
    if (output->type != kTfLiteUInt8) {
        ESP_LOGW(TAG, "Output dtype is not UINT8");
        return 0;
    }
    return output->data.uint8[0];
}

// ─────────────────────────────────────────────────────────────
//  Op Registration
// ─────────────────────────────────────────────────────────────
void ModelRunner::RegisterOps()
{
    // Ops required by the MicroWakeWord streaming model.
    // Derived from the model analysis: CONV_2D, STRIDED_SLICE, CONCATENATION,
    // FULLY_CONNECTED, LOGISTIC, QUANTIZE, VAR_HANDLE, READ_VARIABLE,
    // ASSIGN_VARIABLE.
    op_resolver_->AddConv2D();
    op_resolver_->AddDepthwiseConv2D();
    op_resolver_->AddFullyConnected();
    op_resolver_->AddLogistic();
    op_resolver_->AddQuantize();
    op_resolver_->AddDequantize();
    op_resolver_->AddReshape();
    op_resolver_->AddSoftmax();
    op_resolver_->AddStridedSlice();
    op_resolver_->AddConcatenation();
    op_resolver_->AddMean();
    op_resolver_->AddPad();
    op_resolver_->AddRelu();
    op_resolver_->AddCallOnce();
    op_resolver_->AddVarHandle();
    op_resolver_->AddReadVariable();
    op_resolver_->AddAssignVariable();
    op_resolver_->AddMul();
    op_resolver_->AddAdd();
    op_resolver_->AddSub();
}
