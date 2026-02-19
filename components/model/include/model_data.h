/**
 * @file model_data.h
 * @brief Declares the MicroWakeWord TFLite flatbuffer as a C array.
 *
 * The actual model bytes are defined in model_data.cc, which is generated
 * by running:
 *
 *   xxd -i your_model.tflite > model_data.cc
 *
 * Then rename the generated array and length to match the names below.
 *
 * The array MUST be aligned to at least 8 bytes; the attribute is set in
 * model_data.cc.  On ESP32-S3, placing it in internal flash (default) is fine
 * since the MMU maps flash into the address space as read-only memory.
 */

#pragma once

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

extern const uint8_t g_model_data[];
extern const size_t  g_model_data_len;

#ifdef __cplusplus
}
#endif
