/**
 * @file main.cpp
 * @brief Hellum Hub — Application entry point.
 *
 * Boot sequence:
 *   1. NVS Flash init
 *   2. Wi-Fi STA connect (blocking with retry)
 *   3. Allocate shared HellumContext (FreeRTOS primitives, ring buffer)
 *   4. Init AudioPreprocessor and ModelRunner (existing, untouched logic)
 *   5. Init LED manager
 *   6. Start Audio Task on Core 1 (I2S, wake word, VAD, playback)
 *   7. Start Network Task on Core 0 (WebSocket, Opus, streaming)
 *   8. Main task idles (health monitoring)
 *
 * Hardware: ESP32-S3-N16R8 (dual-core 240MHz, 16MB Flash, 8MB PSRAM)
 */

#include <cstring>

#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/queue.h"
#include "freertos/task.h"

#include "esp_event.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_system.h"
#include "esp_wifi.h"
#include "nvs_flash.h"

#include "audio_preprocessor.h"
#include "audio_task.h"
#include "hellum_types.h"
#include "led_manager.h"
#include "model_runner.h"
#include "network_task.h"
#include "ring_buffer.h"

static const char *TAG = "HellumHub";

// ═════════════════════════════════════════════════════════════════
//  Wi-Fi Configuration (override via menuconfig / Kconfig)
// ═════════════════════════════════════════════════════════════════
#ifndef CONFIG_HELLUM_WIFI_SSID
#define CONFIG_HELLUM_WIFI_SSID "47jio4g"
#endif

#ifndef CONFIG_HELLUM_WIFI_PASS
#define CONFIG_HELLUM_WIFI_PASS "varun1100"
#endif

#ifndef CONFIG_HELLUM_WS_URI
#define CONFIG_HELLUM_WS_URI "wss://sampleserver-cp2h.onrender.com/ws"
#endif

// ═════════════════════════════════════════════════════════════════
//  Shared Context — single global instance, lives for app lifetime
// ═════════════════════════════════════════════════════════════════
static HellumContext s_ctx;

// ═════════════════════════════════════════════════════════════════
//  Wi-Fi Event Handler
// ═════════════════════════════════════════════════════════════════
static int s_wifi_retry_count = 0;
static constexpr int kMaxWifiRetries = 10;

static void wifi_event_handler(void *arg, esp_event_base_t event_base,
                               int32_t event_id, void *event_data) {
  if (event_base == WIFI_EVENT) {
    switch (event_id) {
    case WIFI_EVENT_STA_START:
      esp_wifi_connect();
      break;

    case WIFI_EVENT_STA_DISCONNECTED:
      if (s_wifi_retry_count < kMaxWifiRetries) {
        ++s_wifi_retry_count;
        ESP_LOGW(TAG, "Wi-Fi disconnected — retry %d/%d", s_wifi_retry_count,
                 kMaxWifiRetries);
        esp_wifi_connect();
      } else {
        ESP_LOGE(TAG, "Wi-Fi max retries exceeded");
        xEventGroupClearBits(s_ctx.event_group, BIT_WIFI_CONNECTED);
      }
      break;

    default:
      break;
    }
  } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
    auto *event = static_cast<ip_event_got_ip_t *>(event_data);
    ESP_LOGI(TAG, "Wi-Fi connected — IP: " IPSTR, IP2STR(&event->ip_info.ip));
    s_wifi_retry_count = 0;
    xEventGroupSetBits(s_ctx.event_group, BIT_WIFI_CONNECTED);
  }
}

// ═════════════════════════════════════════════════════════════════
//  Wi-Fi STA Init
// ═════════════════════════════════════════════════════════════════
static void wifi_init_sta() {
  ESP_ERROR_CHECK(esp_netif_init());
  ESP_ERROR_CHECK(esp_event_loop_create_default());
  esp_netif_create_default_wifi_sta();

  wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
  ESP_ERROR_CHECK(esp_wifi_init(&cfg));

  // Register event handlers
  ESP_ERROR_CHECK(esp_event_handler_instance_register(
      WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, nullptr, nullptr));
  ESP_ERROR_CHECK(esp_event_handler_instance_register(
      IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, nullptr, nullptr));

  // Configure STA
  wifi_config_t wifi_cfg = {};
  strncpy(reinterpret_cast<char *>(wifi_cfg.sta.ssid), CONFIG_HELLUM_WIFI_SSID,
          sizeof(wifi_cfg.sta.ssid) - 1);
  strncpy(reinterpret_cast<char *>(wifi_cfg.sta.password),
          CONFIG_HELLUM_WIFI_PASS, sizeof(wifi_cfg.sta.password) - 1);
  wifi_cfg.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;

  ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
  ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_cfg));
  ESP_ERROR_CHECK(esp_wifi_start());

  ESP_LOGI(TAG, "Wi-Fi STA initialised — connecting to \"%s\"...",
           CONFIG_HELLUM_WIFI_SSID);
}

// ═════════════════════════════════════════════════════════════════
//  Application Entry Point
// ═════════════════════════════════════════════════════════════════
extern "C" void app_main() {
  ESP_LOGI(TAG, "╔══════════════════════════════════════════╗");
  ESP_LOGI(TAG, "║        Hellum Hub — Booting...           ║");
  ESP_LOGI(TAG, "║  ESP32-S3-N16R8 | 16MB Flash | 8MB PSRAM║");
  ESP_LOGI(TAG, "╚══════════════════════════════════════════╝");

  // ── 1. NVS Flash ─────────────────────────────────────────
  esp_err_t ret = nvs_flash_init();
  if (ret == ESP_ERR_NVS_NO_FREE_PAGES ||
      ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
    ESP_ERROR_CHECK(nvs_flash_erase());
    ret = nvs_flash_init();
  }
  ESP_ERROR_CHECK(ret);
  ESP_LOGI(TAG, "[1/7] NVS Flash initialised");

  // ── 2. FreeRTOS Primitives ───────────────────────────────
  s_ctx.event_group = xEventGroupCreate();
  s_ctx.tts_queue = xQueueCreate(kTtsQueueDepth, sizeof(TtsChunk));
  if (!s_ctx.event_group || !s_ctx.tts_queue) {
    ESP_LOGE(TAG, "Failed to create FreeRTOS primitives — halting");
    esp_restart();
  }
  ESP_LOGI(TAG, "[2/7] FreeRTOS primitives created");

  // ── 3. Ring Buffer (PSRAM) ───────────────────────────────
  s_ctx.ring_buffer = new RingBuffer(kRingBufferSamples);
  if (!s_ctx.ring_buffer->Init()) {
    ESP_LOGE(TAG, "Ring buffer init failed — halting");
    esp_restart();
  }
  ESP_LOGI(TAG, "[3/7] Ring buffer: %d samples (%ds) in PSRAM",
           kRingBufferSamples, kRingBufferDurationSec);

  // ── 4. Audio Preprocessor (existing, untouched) ──────────
  s_ctx.preprocessor = new AudioPreprocessor();
  if (!s_ctx.preprocessor->Init()) {
    ESP_LOGE(TAG, "AudioPreprocessor init failed — halting");
    esp_restart();
  }
  ESP_LOGI(TAG, "[4/7] AudioPreprocessor ready");

  // ── 5. Model Runner (existing, untouched) ────────────────
  s_ctx.model = new ModelRunner();
  if (!s_ctx.model->Init()) {
    ESP_LOGE(TAG, "ModelRunner init failed — halting");
    esp_restart();
  }
  ESP_LOGI(TAG, "[5/7] ModelRunner ready (streaming state active)");

  // ── 6. LED Manager ───────────────────────────────────────
  led::Init();
  ESP_LOGI(TAG, "[6/7] LED manager initialised");

  // ── 7. Wi-Fi ─────────────────────────────────────────────
  s_ctx.wifi_ssid = CONFIG_HELLUM_WIFI_SSID;
  s_ctx.wifi_pass = CONFIG_HELLUM_WIFI_PASS;
  s_ctx.ws_uri = CONFIG_HELLUM_WS_URI;
  wifi_init_sta();
  ESP_LOGI(TAG, "[7/7] Wi-Fi STA started");

  // ═════════════════════════════════════════════════════════
  //  Launch dual-core tasks
  // ═════════════════════════════════════════════════════════

  ESP_LOGI(TAG, "═══ Starting Dual-Core Pipeline ═══");

  // Core 1: Audio capture, wake word, VAD, playback
  audio_task_start(&s_ctx);

  // Core 0: Network, WebSocket, streaming
  network_task_start(&s_ctx);

  ESP_LOGI(TAG, "╔══════════════════════════════════════════╗");
  ESP_LOGI(TAG, "║      Hellum Hub — All Systems Ready      ║");
  ESP_LOGI(TAG, "║  Audio: Core 1 | Network: Core 0         ║");
  ESP_LOGI(TAG, "║  State: IDLE — Listening for wake word    ║");
  ESP_LOGI(TAG, "╚══════════════════════════════════════════╝");

  // ── Main task: health monitor ────────────────────────────
  // The main task now just monitors heap health and feeds WDT.
  // In production, add OTA update checks, BLE provisioning, etc.
  while (true) {
    vTaskDelay(pdMS_TO_TICKS(30000)); // Every 30 seconds

    ESP_LOGI(TAG, "Health: Free heap=%u, Min free=%u, PSRAM free=%u",
             esp_get_free_heap_size(), esp_get_minimum_free_heap_size(),
             heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
  }
}
