/**
 * @file audio_task.cpp
 * @brief Core 1 — Real-time audio capture, wake word inference, VAD, and playback.
 *
 * This file refactors the original main.cpp inference_task into the full
 * Hellum Hub audio pipeline while preserving the existing AudioPreprocessor
 * and ModelRunner logic completely untouched.
 *
 * Responsibilities:
 *   1. I2S full-duplex init (RX: dual INMP441 stereo, TX: MAX98357A mono)
 *   2. Continuous stereo capture → beamforming → Ring Buffer write
 *   3. IDLE:             Run wake word pipeline
 *   4. STREAMING:        Run energy-based VAD, signal silence timeout
 *   5. RESPONSE:         Read TTS chunks from queue → I2S TX playback
 *   6. POST_RESPONSE:    Reset, return to IDLE
 *
 * The beamforming strategy: Simple delay-and-sum of L/R mic channels.
 * Both INMP441s share the same I2S bus (stereo), so we capture both
 * simultaneously. Averaging L+R provides ~3dB SNR improvement.
 */

#include "audio_task.h"
#include "led_manager.h"
#include "ring_buffer.h"
#include "audio_preprocessor.h"
#include "model_runner.h"

#include <cstring>
#include <cstdlib>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "driver/i2s_std.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_task_wdt.h"

static const char* TAG = "AudioTask";

// ═════════════════════════════════════════════════════════════════
//  Configuration
// ═════════════════════════════════════════════════════════════════

// Wake word detection threshold (~78% confidence)
static constexpr uint8_t kDetectThreshold = 200;

// VAD: energy-based voice activity detection
// Samples with RMS energy below this are "silence"
static constexpr int32_t kVadEnergyThreshold = 500;

// I2S read size = one stride worth of stereo samples
// kStrideSamples (320) defined in audio_preprocessor.h
// Stereo → 2 channels × 320 samples × 4 bytes (32-bit I2S) = 2560 bytes
static constexpr size_t kI2sReadSamples = 320;  // Per channel
static constexpr size_t kI2sReadBytes   = kI2sReadSamples * 2 * sizeof(int32_t);

// ═════════════════════════════════════════════════════════════════
//  I2S Handles
// ═════════════════════════════════════════════════════════════════
static i2s_chan_handle_t s_rx_handle = nullptr;
static i2s_chan_handle_t s_tx_handle = nullptr;

// ═════════════════════════════════════════════════════════════════
//  I2S Initialisation — Full-Duplex on shared bus
// ═════════════════════════════════════════════════════════════════

/**
 * @brief Initialise I2S_NUM_0 in full-duplex mode.
 *
 * RX (mics):  32-bit stereo, SCK=41, WS=42, DIN=2
 * TX (amp):   16-bit mono,   SCK=41, WS=42, DOUT=1
 *
 * Both share the same clock lines. The ESP32-S3 I2S peripheral
 * supports this via the i2s_new_channel API with both tx and rx.
 */
static bool i2s_init_full_duplex()
{
    // Channel config — full duplex: both TX and RX on I2S_NUM_0
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(
        I2S_NUM_0, I2S_ROLE_MASTER);
    chan_cfg.auto_clear = true;

    esp_err_t err = i2s_new_channel(&chan_cfg, &s_tx_handle, &s_rx_handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "i2s_new_channel failed: %s", esp_err_to_name(err));
        return false;
    }

    // ── RX config (microphones — 32-bit stereo, Philips standard) ──
    i2s_std_config_t rx_cfg = {
        .clk_cfg  = I2S_STD_CLK_DEFAULT_CONFIG(kSampleRate),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(
                        I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = static_cast<gpio_num_t>(kPinI2S_SCK),
            .ws   = static_cast<gpio_num_t>(kPinI2S_WS),
            .dout = I2S_GPIO_UNUSED,
            .din  = static_cast<gpio_num_t>(kPinI2S_DIN),
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv   = false,
            },
        },
    };

    err = i2s_channel_init_std_mode(s_rx_handle, &rx_cfg);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "i2s_channel_init_std_mode(RX) failed: %s",
                 esp_err_to_name(err));
        return false;
    }

    // ── TX config (amplifier — 16-bit mono, Philips standard) ──
    i2s_std_config_t tx_cfg = {
        .clk_cfg  = I2S_STD_CLK_DEFAULT_CONFIG(kSampleRate),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(
                        I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = static_cast<gpio_num_t>(kPinI2S_SCK),
            .ws   = static_cast<gpio_num_t>(kPinI2S_WS),
            .dout = static_cast<gpio_num_t>(kPinI2S_DOUT),
            .din  = I2S_GPIO_UNUSED,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv   = false,
            },
        },
    };

    err = i2s_channel_init_std_mode(s_tx_handle, &tx_cfg);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "i2s_channel_init_std_mode(TX) failed: %s",
                 esp_err_to_name(err));
        return false;
    }

    // Enable RX immediately; TX enabled on-demand during playback
    ESP_ERROR_CHECK(i2s_channel_enable(s_rx_handle));

    ESP_LOGI(TAG, "I2S full-duplex initialised: RX(stereo 32-bit) + TX(mono 16-bit)");
    return true;
}

// ═════════════════════════════════════════════════════════════════
//  Beamforming — average L/R channels for SNR improvement
// ═════════════════════════════════════════════════════════════════

/**
 * @brief Extract and beamform stereo I2S data into mono PCM.
 *
 * INMP441 outputs 24-bit data left-justified in 32-bit words.
 * L channel (Mic 1): even indices, R channel (Mic 2): odd indices.
 * We average both channels for a ~3dB noise floor reduction.
 *
 * @param raw_stereo  Input: interleaved 32-bit stereo samples.
 * @param frames      Number of stereo frames.
 * @param mono_out    Output: 16-bit mono PCM (beamformed).
 * @return Peak absolute amplitude for diagnostics.
 */
static int32_t beamform_stereo_to_mono(const int32_t* raw_stereo,
                                        int frames,
                                        int16_t* mono_out)
{
    int32_t peak = 0;

    for (int i = 0; i < frames; ++i) {
        // Extract 16-bit from 32-bit I2S (upper 16 bits = MSB alignment)
        int16_t left  = static_cast<int16_t>(raw_stereo[i * 2]     >> 16);
        int16_t right = static_cast<int16_t>(raw_stereo[i * 2 + 1] >> 16);

        // Delay-and-sum beamforming: simple average of both mics
        // Division by 2 prevents overflow; the SNR gain comes from
        // uncorrelated noise averaging while correlated speech adds
        int16_t beamformed = static_cast<int16_t>((static_cast<int32_t>(left) +
                                                    static_cast<int32_t>(right)) / 2);

        mono_out[i] = beamformed;

        int32_t abs_val = beamformed < 0 ? -beamformed : beamformed;
        if (abs_val > peak) peak = abs_val;
    }

    return peak;
}

// ═════════════════════════════════════════════════════════════════
//  Energy-based VAD
// ═════════════════════════════════════════════════════════════════

/**
 * @brief Compute RMS energy of a PCM buffer.
 * @return Approximate RMS (integer, not true sqrt — uses mean absolute value).
 */
static int32_t compute_energy(const int16_t* pcm, int count)
{
    int64_t sum = 0;
    for (int i = 0; i < count; ++i) {
        int32_t s = pcm[i];
        sum += (s < 0) ? -s : s;
    }
    return static_cast<int32_t>(sum / count);
}

// ═════════════════════════════════════════════════════════════════
//  Main Audio Task
// ═════════════════════════════════════════════════════════════════

static void audio_task_entry(void* arg)
{
    auto* ctx = static_cast<HellumContext*>(arg);

    esp_task_wdt_add(nullptr);

    // ── Allocate working buffers ─────────────────────────────
    //    Using stack-static for deterministic memory
    static int32_t raw_stereo[kI2sReadSamples * 2];   // 32-bit stereo from I2S
    static int16_t mono_pcm[kI2sReadSamples];          // Beamformed mono
    static int8_t  features[kPreprocessorFeatureSize];  // Mel features

    int debug_counter     = 0;
    int64_t silence_start = 0;
    bool    silence_active = false;

    ESP_LOGI(TAG, "Audio task started on Core %d", xPortGetCoreID());

    while (true) {
        const HellumState state = ctx->state.load();

        // ── 1. Read raw I2S stereo data ──────────────────────
        size_t bytes_read = 0;
        esp_err_t err = i2s_channel_read(
            s_rx_handle, raw_stereo, kI2sReadBytes, &bytes_read, portMAX_DELAY);

        esp_task_wdt_reset();

        if (err != ESP_OK || bytes_read == 0) {
            vTaskDelay(1);
            continue;
        }

        const int frames_read = bytes_read / (sizeof(int32_t) * 2);

        // ── 2. Beamform stereo → mono ────────────────────────
        int32_t peak = beamform_stereo_to_mono(raw_stereo, frames_read, mono_pcm);

        // ── 3. Always write to ring buffer (continuous recording) ─
        ctx->ring_buffer->Write(mono_pcm, frames_read);

        // ── 4. Update LED based on state ─────────────────────
        led::Update(state);

        // ── State-dependent processing ───────────────────────
        switch (state) {

        // ─────────────────────────────────────────────────────
        //  IDLE — Wake word detection pipeline
        // ─────────────────────────────────────────────────────
        case HellumState::IDLE: {
            // Feed mono PCM into the AudioPreprocessor
            bool slice_ready = ctx->preprocessor->ProcessSamples(
                mono_pcm, frames_read, features);

            if (!slice_ready) break;

            // Copy features into model input and run inference
            int8_t* input_ptr = ctx->model->GetInputBuffer();
            memcpy(input_ptr, features,
                   kPreprocessorFeatureSize * sizeof(int8_t));

            esp_task_wdt_reset();
            ctx->model->Invoke();
            esp_task_wdt_reset();

            uint8_t prob = ctx->model->GetOutputProbability();

            // Diagnostic logs every ~500ms (25 × 20ms)
            if (++debug_counter >= 25) {
                debug_counter = 0;
                const char* level =
                    (peak < 500)   ? "SILENCE" :
                    (peak < 10000) ? "WEAK"    :
                    (peak < 28000) ? "GOOD"    : "CLIPPING";
                ESP_LOGI(TAG, "Vol:%5ld Prob:%3u feat:%d,%d,%d,%d [%s]",
                         peak, prob,
                         features[0], features[1], features[2], features[3],
                         level);
            }

            // ── Wake word detected! ──────────────────────────
            if (prob > kDetectThreshold) {
                ESP_LOGI(TAG, ">>> WAKE WORD DETECTED (prob=%u) <<<", prob);

                ctx->state.store(HellumState::WAKE_DETECTED);
                xEventGroupSetBits(ctx->event_group, BIT_WAKE_DETECTED);

                // Reset silence tracker for upcoming STREAMING state
                silence_active = false;
            }
            break;
        }

        // ─────────────────────────────────────────────────────
        //  WAKE_DETECTED — Brief transition state
        //  Network task handles pre-roll; audio just keeps recording
        // ─────────────────────────────────────────────────────
        case HellumState::WAKE_DETECTED:
            // Ring buffer continues to be written.
            // Network task reads pre-roll and transitions to STREAMING.
            break;

        // ─────────────────────────────────────────────────────
        //  STREAMING — Live audio, running VAD for silence cutoff
        // ─────────────────────────────────────────────────────
        case HellumState::STREAMING: {
            int32_t energy = compute_energy(mono_pcm, frames_read);
            int64_t now_us = esp_timer_get_time();

            if (energy < kVadEnergyThreshold) {
                // Silence detected
                if (!silence_active) {
                    silence_active = true;
                    silence_start  = now_us;
                } else if ((now_us - silence_start) >=
                           (kVadSilenceTimeoutMs * 1000LL)) {
                    // Silence exceeded threshold → stop streaming
                    ESP_LOGI(TAG, "VAD: %dms silence → stopping stream",
                             kVadSilenceTimeoutMs);
                    ctx->state.store(HellumState::SERVER_PROCESSING);
                    xEventGroupSetBits(ctx->event_group, BIT_STREAM_STOP);
                    silence_active = false;
                }
            } else {
                // Voice activity — reset silence counter
                silence_active = false;
            }
            break;
        }

        // ─────────────────────────────────────────────────────
        //  SERVER_PROCESSING — Waiting for server response
        //  Audio task just idles and updates LED
        // ─────────────────────────────────────────────────────
        case HellumState::SERVER_PROCESSING:
            // Nothing to do — network task handles the wait
            break;

        // ─────────────────────────────────────────────────────
        //  RESPONSE — TTS playback via I2S TX
        // ─────────────────────────────────────────────────────
        case HellumState::RESPONSE: {
            // Try to receive a TTS chunk from the queue (non-blocking)
            TtsChunk chunk;
            if (xQueueReceive(ctx->tts_queue, &chunk, 0) == pdTRUE) {
                // Enable TX if not already enabled
                static bool tx_enabled = false;
                if (!tx_enabled) {
                    i2s_channel_enable(s_tx_handle);
                    tx_enabled = true;
                }

                // Write PCM to amplifier
                size_t bytes_written = 0;
                i2s_channel_write(s_tx_handle,
                                  chunk.samples,
                                  chunk.length * sizeof(int16_t),
                                  &bytes_written,
                                  portMAX_DELAY);

                // Check if this was the last chunk
                if (chunk.is_last) {
                    // Small delay to let the DAC flush
                    vTaskDelay(pdMS_TO_TICKS(100));
                    i2s_channel_disable(s_tx_handle);
                    tx_enabled = false;

                    ESP_LOGI(TAG, "TTS playback complete");
                    ctx->state.store(HellumState::POST_RESPONSE);
                }
            }
            break;
        }

        // ─────────────────────────────────────────────────────
        //  POST_RESPONSE — Reset and return to IDLE
        // ─────────────────────────────────────────────────────
        case HellumState::POST_RESPONSE: {
            ESP_LOGI(TAG, "Post-response cleanup → returning to IDLE");

            // Short visual feedback (LED handles the pattern)
            vTaskDelay(pdMS_TO_TICKS(500));

            // Clear any stale event bits
            xEventGroupClearBits(ctx->event_group, BIT_ALL);

            // Return to listening mode
            ctx->state.store(HellumState::IDLE);
            debug_counter = 0;
            break;
        }

        }  // switch
    }  // while
}

// ═════════════════════════════════════════════════════════════════
//  Public entry point
// ═════════════════════════════════════════════════════════════════
void audio_task_start(HellumContext* ctx)
{
    // Initialise I2S full-duplex
    if (!i2s_init_full_duplex()) {
        ESP_LOGE(TAG, "I2S init failed — cannot start audio task");
        return;
    }

    // Create the task pinned to Core 1
    xTaskCreatePinnedToCore(
        audio_task_entry,
        "audio_task",
        16384,          // 16KB stack (matches original inference task)
        ctx,
        5,              // High priority for real-time audio
        nullptr,
        1               // Core 1 (App CPU)
    );

    ESP_LOGI(TAG, "Audio task created (Core 1, priority 5)");
}
