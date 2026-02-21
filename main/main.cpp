/**
 * @file main.cpp
 * @brief Wake Word Detection Firmware — ESP32-S3-N16R8
 *
 * Streaming pipeline:
 *   I2S (INMP441) → AudioPreprocessor → MicroWakeWord → LED trigger
 *
 * Every 10 ms:
 *   1. Read 160 PCM samples from I2S
 *   2. Generate one [1,1,40] INT8 feature slice
 *   3. Run TFLite Micro inference
 *   4. If output > 200 (~80%), toggle wake-word LED for 2 s
 */

#include <cstdio>
#include <cstring>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "driver/i2s_std.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_task_wdt.h"

#include "audio_preprocessor.h"
#include "model_runner.h"

// ─────────────────────────────────────────────────────────────
//  Hardware configuration
// ─────────────────────────────────────────────────────────────
static constexpr int    kI2S_SCK   = 41;
static constexpr int    kI2S_WS    = 42;
static constexpr int    kI2S_SD    = 2;

static constexpr gpio_num_t kLED_R  = GPIO_NUM_48;
static constexpr gpio_num_t kLED_G  = GPIO_NUM_47;
static constexpr gpio_num_t kLED_B  = GPIO_NUM_21;

// ─────────────────────────────────────────────────────────────
//  Audio / model configuration
// ─────────────────────────────────────────────────────────────
static constexpr int kSampleRate        = 16000;
static constexpr int kStrideMs          = 10;
static constexpr int kStrideSamples     = (kSampleRate * kStrideMs) / 1000;  // 160
static constexpr int kNumMelChannels    = 40;

static constexpr uint8_t kDetectThresh  = 200;   // ~78.4 % confidence
static constexpr int64_t kLedOnUs       = 2000000LL; // 2 seconds in µs

static const char* TAG = "WWD";

// ─────────────────────────────────────────────────────────────
//  Globals
// ─────────────────────────────────────────────────────────────
static i2s_chan_handle_t  s_i2s_rx_handle = nullptr;
static AudioPreprocessor* s_preprocessor  = nullptr;
static ModelRunner*       s_model         = nullptr;

static int64_t  s_led_off_time_us = 0;
static bool     s_led_active      = false;

// ─────────────────────────────────────────────────────────────
//  LED helpers
// ─────────────────────────────────────────────────────────────
static void led_init()
{
    gpio_config_t cfg = {};
    cfg.pin_bit_mask = (1ULL << kLED_R) | (1ULL << kLED_G) | (1ULL << kLED_B);
    cfg.mode         = GPIO_MODE_OUTPUT;
    cfg.pull_up_en   = GPIO_PULLUP_DISABLE;
    cfg.pull_down_en = GPIO_PULLDOWN_DISABLE;
    cfg.intr_type    = GPIO_INTR_DISABLE;
    gpio_config(&cfg);

    // All off (common-cathode → LOW = off)
    gpio_set_level(kLED_R, 0);
    gpio_set_level(kLED_G, 0);
    gpio_set_level(kLED_B, 0);
}

static inline void led_wake_on()
{
    gpio_set_level(kLED_G, 1);   // Green = wake word detected
    gpio_set_level(kLED_R, 0);
    gpio_set_level(kLED_B, 0);
    s_led_off_time_us = esp_timer_get_time() + kLedOnUs;
    s_led_active      = true;
}

static inline void led_off()
{
    gpio_set_level(kLED_R, 0);
    gpio_set_level(kLED_G, 0);
    gpio_set_level(kLED_B, 0);
    s_led_active = false;
}

static void led_tick()
{
    if (s_led_active && esp_timer_get_time() >= s_led_off_time_us) {
        led_off();
    }
}

// ─────────────────────────────────────────────────────────────
//  I2S initialisation — INMP441 @ 16 kHz mono
// ─────────────────────────────────────────────────────────────
static void i2s_init()
{
    // Channel configuration
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(
        I2S_NUM_0,
        I2S_ROLE_MASTER
    );
    chan_cfg.auto_clear = true;

    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, nullptr, &s_i2s_rx_handle));

    // Standard mode (Philips / I2S)
    i2s_std_config_t std_cfg = {
        .clk_cfg  = I2S_STD_CLK_DEFAULT_CONFIG(kSampleRate),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT,
                                                     I2S_SLOT_MODE_STEREO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = static_cast<gpio_num_t>(kI2S_SCK),
            .ws   = static_cast<gpio_num_t>(kI2S_WS),
            .dout = I2S_GPIO_UNUSED,
            .din  = static_cast<gpio_num_t>(kI2S_SD),
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv   = false,
            },
        },
    };

    ESP_ERROR_CHECK(i2s_channel_init_std_mode(s_i2s_rx_handle, &std_cfg));
    ESP_ERROR_CHECK(i2s_channel_enable(s_i2s_rx_handle));

    ESP_LOGI(TAG, "I2S initialised: %d Hz, Philips, 32-bit, Stereo", kSampleRate);
}

// ─────────────────────────────────────────────────────────────
//  Main streaming task
// ─────────────────────────────────────────────────────────────
static void inference_task(void* /*arg*/)
{
    esp_task_wdt_add(NULL);

    // Buffers
    static int32_t  raw_buf[kStrideSamples * 2]; // 32-bit Stereo
    static int16_t  pcm_buf[kStrideSamples];     // 16-bit Mono
    static int8_t   features[kNumMelChannels];

    // Filter state
    static float dc_offset = 0.0f;
    const float  dc_alpha  = 0.95f; 

    // ── TUNING PARAMETER ──────────────────────────────────────────
    const int32_t kGainShift = 12; // Adjusted for optimal volume
    // ──────────────────────────────────────────────────────────────

    static int debug_timer = 0;

    ESP_LOGI(TAG, "Streaming task started (Gain Tuned to >> 12)");

    while (true) {
        size_t bytes_read = 0;
        
        // 1. Read Raw I2S
        esp_err_t err = i2s_channel_read(
            s_i2s_rx_handle, raw_buf, sizeof(raw_buf), &bytes_read, portMAX_DELAY
        );

        esp_task_wdt_reset();

        if (err != ESP_OK || bytes_read == 0) {
            vTaskDelay(1);
            continue;
        }

        // 2. Process Audio
        int frames_read = bytes_read / (sizeof(int32_t) * 2);
        int32_t signal_max_ac = 0; 

        for (int i = 0; i < frames_read; i++) {
            // LEFT Channel: raw_buf[i * 2]
            // If you get NO signal, try RIGHT: raw_buf[i * 2 + 1]
            int32_t raw = raw_buf[i * 2]; 

            // DC Removal
            dc_offset = (dc_alpha * dc_offset) + ((1.0f - dc_alpha) * (float)raw);
            float centered = (float)raw - dc_offset;

            // Apply Gain
            int32_t amplified = (int32_t)centered >> kGainShift;

            // Clamp
            if (amplified > 32767)  amplified = 32767;
            if (amplified < -32768) amplified = -32768;

            pcm_buf[i] = (int16_t)amplified;

            if (abs(amplified) > signal_max_ac) signal_max_ac = abs(amplified);
        }

        // 3. Generate Features
        bool slice_ready = s_preprocessor->ProcessSamples(
            pcm_buf, frames_read, features
        );

        if (!slice_ready) {
            vTaskDelay(1);
            continue;
        }

        // 4. Inference
        int8_t* input_ptr = s_model->GetInputBuffer();
        memcpy(input_ptr, features, kNumMelChannels * sizeof(int8_t));

        esp_task_wdt_reset();
        s_model->Invoke();
        esp_task_wdt_reset();

        uint8_t prob = s_model->GetOutputProbability();

        // 5. Intelligent Debug Logs
        debug_timer++;
        if (debug_timer >= 50) { // Every ~500ms
            debug_timer = 0;
            
            const char* status = "UNKNOWN";
            if (signal_max_ac < 500)        status = "SILENCE (OK)";
            else if (signal_max_ac < 10000) status = "WEAK SIGNAL (Speak Louder)";
            else if (signal_max_ac < 28000) status = "GOOD LEVEL (Target)";
            else                            status = "CLIPPING (Too Loud)";
            
            ESP_LOGI(TAG, "Vol: %5ld | Prob: %3u | %s", signal_max_ac, prob, status);
        }

        if (prob > kDetectThresh) {
            ESP_LOGI(TAG, ">>> ALEXA DETECTED (Prob: %d) <<<", prob);
            led_wake_on();
        }

        led_tick();
        vTaskDelay(1);
    }
}

// ─────────────────────────────────────────────────────────────
//  Application entry point
// ─────────────────────────────────────────────────────────────
extern "C" void app_main()
{
    ESP_LOGI(TAG, "=== Wake Word Detection Booting ===");
    ESP_LOGI(TAG, "Chip: ESP32-S3-N16R8 | Flash: 16MB | PSRAM: 8MB Octal");

    // ── LEDs ──────────────────────────────────────────────────
    led_init();

    // ── Audio Preprocessor ───────────────────────────────────
    s_preprocessor = new AudioPreprocessor();
    if (!s_preprocessor->Init()) {
        ESP_LOGE(TAG, "AudioPreprocessor init failed — halting");
        esp_restart();
    }

    // ── Model ────────────────────────────────────────────────
    s_model = new ModelRunner();
    if (!s_model->Init()) {
        ESP_LOGE(TAG, "ModelRunner init failed — halting");
        esp_restart();
    }

    // ── I2S ──────────────────────────────────────────────────
    i2s_init();

    ESP_LOGI(TAG, "All systems ready. Starting inference task...");

    // ── Streaming task (pinned to Core 1) ────────────────────
    xTaskCreatePinnedToCore(
        inference_task,
        "inference",
        8192,          // Stack in internal SRAM
        nullptr,
        5,             // Priority
        nullptr,
        1              // Core 1 (App CPU)
    );
}
