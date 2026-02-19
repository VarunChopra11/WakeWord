/**
 * @file fft_util.cpp
 * @brief Real FFT wrapper using espressif/esp-dsp.
 *
 * esp-dsp provides dsps_fft2r_fc32 which performs a complex FFT on
 * interleaved [re, im] float data.  For a real input signal we zero-fill
 * the imaginary channel, execute the full complex transform, and interpret
 * the first (N/2 + 1) complex bins as our one-sided spectrum.
 */

#include "fft_util.h"

#include <cstdlib>
#include <cstring>
#include <cmath>

#include "esp_log.h"
#include "esp_heap_caps.h"

// esp-dsp headers — provided by espressif/esp-dsp component
#include "dsps_fft2r.h"
#include "dsps_wind.h"

static const char* TAG = "fft_util";

// Pointer to the twiddled-factor table allocated by dsps_fft2r_init_fc32
static bool s_initialized = false;
static int  s_fft_size    = 0;

bool fft_init(int fft_size)
{
    if (s_initialized) {
        if (s_fft_size == fft_size) return true;
        fft_deinit();
    }

    esp_err_t ret = dsps_fft2r_init_fc32(nullptr, fft_size);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "dsps_fft2r_init_fc32 failed: %s", esp_err_to_name(ret));
        return false;
    }

    s_fft_size    = fft_size;
    s_initialized = true;
    ESP_LOGI(TAG, "FFT initialised: N=%d", fft_size);
    return true;
}

void fft_real_forward(float* data, int fft_size)
{
    // data layout: [re0, im0, re1, im1, …] — im is pre-zeroed by caller
    // 1. Bit-reverse permutation + DIF butterfly
    dsps_fft2r_fc32(data, fft_size);
    // 2. Reorder the output into natural frequency order
    dsps_bit_rev_fc32(data, fft_size);
    // 3. Convert to complex conjugate pairs (standard C2C FFT convention)
    dsps_cplx2reC_fc32(data, fft_size);
}

void fft_deinit()
{
    if (s_initialized) {
        dsps_fft2r_deinit_fc32();
        s_initialized = false;
        s_fft_size    = 0;
    }
}
