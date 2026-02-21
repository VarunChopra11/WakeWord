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
        .slot_cfg = I2S_STD_MSB_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT,
                                                     I2S_SLOT_MODE_MONO),
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

    ESP_LOGI(TAG, "I2S initialised: %d Hz, mono, 16-bit", kSampleRate);
}

// ─────────────────────────────────────────────────────────────
//  Main streaming task
// ─────────────────────────────────────────────────────────────
static void inference_task(void* /*arg*/)
{
    // ── Subscribe this task to the Task Watchdog Timer ────────
    //
    // IDLE1 monitoring is disabled via sdkconfig (the inference task
    // permanently hogs Core 1 at priority 5, starving IDLE1).  Instead
    // this task feeds the WDT itself before and after every Invoke().
    esp_task_wdt_add(NULL);

    // PCM read buffer: 160 samples × 2 bytes = 320 bytes
    static int16_t  pcm_buf[kStrideSamples];
    static int8_t   features[kNumMelChannels];

    ESP_LOGI(TAG, "Streaming task started (stride=%d samples)", kStrideSamples);

    while (true) {
        // ── 1. Read 160 PCM samples from I2S ──────────────────
        size_t bytes_read = 0;
        esp_err_t err = i2s_channel_read(
            s_i2s_rx_handle,
            pcm_buf,
            sizeof(pcm_buf),
            &bytes_read,
            portMAX_DELAY
        );

        // Feed WDT after the potentially long blocking I2S read
        esp_task_wdt_reset();

        if (err != ESP_OK || bytes_read == 0) {
            ESP_LOGW(TAG, "I2S read error: %s", esp_err_to_name(err));
            vTaskDelay(1);
            continue;
        }

        int samples_read = static_cast<int>(bytes_read / sizeof(int16_t));

        // ── 2. Generate one [1,1,40] feature slice ────────────
        bool slice_ready = s_preprocessor->ProcessSamples(
            pcm_buf, samples_read, features
        );

        if (!slice_ready) {
            // Not enough audio buffered yet — yield and loop back
            vTaskDelay(1);
            continue;
        }

        // ── 3. Copy features into model input tensor ──────────
        int8_t* input_ptr = s_model->GetInputBuffer();
        memcpy(input_ptr, features, kNumMelChannels * sizeof(int8_t));

        // ── 4. Run inference ───────────────────────────────────
        //   Feed WDT immediately before the long Invoke() so the
        //   full TWDT timeout window (30 s) is available.
        esp_task_wdt_reset();

        if (!s_model->Invoke()) {
            ESP_LOGE(TAG, "Inference failed!");
            vTaskDelay(1);
            continue;
        }

        // Feed WDT immediately after Invoke() completes
        esp_task_wdt_reset();

        // ── 5. Read output probability [0–255] ────────────────
        uint8_t probability = s_model->GetOutputProbability();

        ESP_LOGD(TAG, "Probability: %u", probability);

        // ── 6. Wake word detection check ──────────────────────
        if (probability > kDetectThresh) {
            ESP_LOGI(TAG, ">>> WAKE WORD DETECTED  (prob=%u/255) <<<",
                     probability);
            led_wake_on();
        }

        // ── 7. LED timeout management ─────────────────────────
        led_tick();

        // ── 8. Yield to let lower-priority tasks run ──────────
        //   vTaskDelay(1) guarantees a context switch regardless
        //   of priority, unlike taskYIELD() which only yields to
        //   equal-or-higher priority tasks (never to IDLE at 0).
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
