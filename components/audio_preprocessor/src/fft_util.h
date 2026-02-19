/**
 * @file fft_util.h
 * @brief Thin wrapper around esp-dsp radix-2 FFT for real-valued audio.
 *
 * Performs an in-place real FFT using the split-radix algorithm provided by
 * the espressif/esp-dsp component, which uses Xtensa SIMD intrinsics on the
 * ESP32-S3 for maximum throughput.
 */

#pragma once

#include <cstddef>

/**
 * @brief Initialise FFT tables for a given transform size.
 *
 * Must be called once before any calls to fft_real_forward().
 *
 * @param fft_size  Power-of-2 FFT length (e.g. 512).
 * @return true on success.
 */
bool fft_init(int fft_size);

/**
 * @brief Perform a real-to-complex FFT in-place.
 *
 * @param[in,out] data   Interleaved [re, im, re, im, …] buffer of length
 *                       2 × fft_size floats.  On entry the imaginary parts
 *                       must be zero-filled.  On return the buffer contains
 *                       the complex frequency-domain result.
 * @param fft_size       Must match the value passed to fft_init().
 */
void fft_real_forward(float* data, int fft_size);

/**
 * @brief Release FFT table memory.
 */
void fft_deinit();
