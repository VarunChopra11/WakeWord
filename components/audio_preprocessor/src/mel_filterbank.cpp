/**
 * @file mel_filterbank.cpp
 * @brief Pre-computed triangular mel filterbank — 40 bands, 20–7600 Hz.
 *
 * Algorithm:
 *   1. Convert frequency limits to mel scale (HTK formula).
 *   2. Distribute kMFB_NumBands + 2 centre points linearly in mel space.
 *   3. Convert centre points back to Hz, then to FFT bin indices.
 *   4. For each FFT bin, compute which (lower, upper) triangle it falls in
 *      and store the weight in a flat lookup array.
 *
 * The sparse table means mel_filterbank_apply() runs in O(kMFB_NumBins)
 * rather than O(kMFB_NumBins × kMFB_NumBands).
 */

#include "mel_filterbank.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <new>

#include "esp_log.h"
#include "esp_heap_caps.h"

static const char* TAG = "mel_fb";

// Two contributions per bin (a bin can straddle the edge of two triangles)
static constexpr int kMaxContribPerBin = 2;

struct BinContrib {
    int   band;
    float weight;
};

static BinContrib* s_contrib     = nullptr; // [kMFB_NumBins][kMaxContribPerBin]
static int*        s_contrib_cnt = nullptr; // [kMFB_NumBins]

// ── Mel / Hz conversion (HTK) ────────────────────────────────
static inline float hz_to_mel(float hz)
{
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

static inline float mel_to_hz(float mel)
{
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

// ── Public API ────────────────────────────────────────────────
bool mel_filterbank_init()
{
    // Allocate in PSRAM
    s_contrib = static_cast<BinContrib*>(
        heap_caps_calloc(kMFB_NumBins * kMaxContribPerBin,
                         sizeof(BinContrib),
                         MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));

    s_contrib_cnt = static_cast<int*>(
        heap_caps_calloc(kMFB_NumBins, sizeof(int),
                         MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));

    if (!s_contrib || !s_contrib_cnt) {
        ESP_LOGE(TAG, "Failed to allocate filterbank tables in PSRAM");
        return false;
    }

    // ── Build centre-frequency array (kMFB_NumBands + 2 points) ──
    const int   num_centres = kMFB_NumBands + 2;
    float       centres_hz[num_centres];
    {
        float mel_lo = hz_to_mel(kMFB_LowFreq);
        float mel_hi = hz_to_mel(kMFB_HighFreq);
        float mel_step = (mel_hi - mel_lo) / (kMFB_NumBands + 1);
        for (int i = 0; i < num_centres; ++i) {
            centres_hz[i] = mel_to_hz(mel_lo + i * mel_step);
        }
    }

    // ── Convert centre frequencies to FFT bin indices ─────────
    // bin_index = round(freq / (sample_rate / fft_size))
    const float hz_per_bin = static_cast<float>(kMFB_SampleRate) /
                             static_cast<float>((kMFB_NumBins - 1) * 2);

    float centre_bins[num_centres];
    for (int i = 0; i < num_centres; ++i) {
        centre_bins[i] = centres_hz[i] / hz_per_bin;
    }

    // ── Fill contribution table ───────────────────────────────
    for (int bin = 0; bin < kMFB_NumBins; ++bin) {
        float f = static_cast<float>(bin);
        int cnt = 0;

        for (int band = 0; band < kMFB_NumBands; ++band) {
            float left   = centre_bins[band];
            float centre = centre_bins[band + 1];
            float right  = centre_bins[band + 2];

            float w = 0.0f;
            if (f >= left && f < centre) {
                w = (f - left) / (centre - left);
            } else if (f >= centre && f <= right) {
                w = (right - f) / (right - centre);
            }

            if (w > 0.0f && cnt < kMaxContribPerBin) {
                int idx = bin * kMaxContribPerBin + cnt;
                s_contrib[idx].band   = band;
                s_contrib[idx].weight = w;
                ++cnt;
            }
        }
        s_contrib_cnt[bin] = cnt;
    }

    ESP_LOGI(TAG, "Mel filterbank initialised: %d bands, %.0f–%.0f Hz",
             kMFB_NumBands, kMFB_LowFreq, kMFB_HighFreq);
    return true;
}

void mel_filterbank_apply(const float* power_spectrum, float* mel_energies)
{
    // mel_energies is assumed to be zero on entry
    for (int bin = 0; bin < kMFB_NumBins; ++bin) {
        float power = power_spectrum[bin];
        if (power == 0.0f) continue;

        int cnt = s_contrib_cnt[bin];
        const BinContrib* c = &s_contrib[bin * kMaxContribPerBin];
        for (int i = 0; i < cnt; ++i) {
            mel_energies[c[i].band] += power * c[i].weight;
        }
    }
}

void mel_filterbank_deinit()
{
    heap_caps_free(s_contrib);
    heap_caps_free(s_contrib_cnt);
    s_contrib     = nullptr;
    s_contrib_cnt = nullptr;
}
