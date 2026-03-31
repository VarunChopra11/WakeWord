/**
 * @file network_task.cpp
 * @brief Core 0 — Network connectivity, WebSocket streaming, Opus codec, TTS handling.
 *
 * State-driven behaviour:
 *   IDLE:              WebSocket stays connected (dormant). Waits for BIT_WAKE_DETECTED.
 *   WAKE_DETECTED:     Reads pre-roll from Ring Buffer, Opus-encodes, sends via WS.
 *                      Transitions to STREAMING.
 *   STREAMING:         Reads live audio from Ring Buffer in real-time, encodes, sends.
 *                      On BIT_STREAM_STOP: sends {"event":"stop"} and → SERVER_PROCESSING.
 *   SERVER_PROCESSING: Waits for incoming WS binary frames (TTS audio).
 *   RESPONSE:          Decodes incoming Opus TTS frames → pushes TtsChunks to queue.
 *   POST_RESPONSE:     Handled by audio task (network task returns to waiting).
 *
 * Opus encoding:
 *   - Mono 16kHz, 20ms frames (320 samples)
 *   - CELT mode for low-latency
 *   - ~13 kbps target bitrate
 *
 * WebSocket protocol:
 *   - Binary frames → Opus-encoded audio
 *   - Text frames   → JSON control messages
 */

#include "network_task.h"
#include "ring_buffer.h"

#include <cstring>
#include <cstdlib>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"

#include "esp_log.h"
#include "esp_timer.h"
#include "esp_websocket_client.h"
#include "cJSON.h"

static const char* TAG = "NetTask";

// ═════════════════════════════════════════════════════════════════
//  Opus Codec Stubs
//
//  If the esp_opus component is available, we use real Opus.
//  Otherwise, we fall back to streaming raw PCM over WebSocket.
//  The WebSocket frame format is the same either way — the server
//  must be configured to decode accordingly.
// ═════════════════════════════════════════════════════════════════

// Opus is optional — controlled by build flag
#if defined(CONFIG_HELLUM_USE_OPUS) && CONFIG_HELLUM_USE_OPUS
    #include "opus.h"
    static OpusEncoder* s_opus_enc = nullptr;
    static OpusDecoder* s_opus_dec = nullptr;
    static constexpr bool kUseOpus = true;
#else
    static constexpr bool kUseOpus = false;
#endif

// Opus output buffer (max Opus frame is ~1275 bytes, but typical is ~60-80)
static constexpr size_t kOpusMaxBytes = 512;
static uint8_t s_opus_out[kOpusMaxBytes];

// ═════════════════════════════════════════════════════════════════
//  WebSocket State
// ═════════════════════════════════════════════════════════════════
static esp_websocket_client_handle_t s_ws_client = nullptr;
static HellumContext* s_ctx = nullptr;  // For use in WS callbacks

// ═════════════════════════════════════════════════════════════════
//  Opus Init / Cleanup
// ═════════════════════════════════════════════════════════════════
#if defined(CONFIG_HELLUM_USE_OPUS) && CONFIG_HELLUM_USE_OPUS
static bool opus_init()
{
    int err;
    s_opus_enc = opus_encoder_create(kSampleRate, 1, OPUS_APPLICATION_VOIP, &err);
    if (err != OPUS_OK || !s_opus_enc) {
        ESP_LOGE(TAG, "Opus encoder create failed: %d", err);
        return false;
    }
    // Low-latency settings
    opus_encoder_ctl(s_opus_enc, OPUS_SET_BITRATE(16000));
    opus_encoder_ctl(s_opus_enc, OPUS_SET_COMPLEXITY(2));  // Low CPU
    opus_encoder_ctl(s_opus_enc, OPUS_SET_SIGNAL(OPUS_SIGNAL_VOICE));

    s_opus_dec = opus_decoder_create(kSampleRate, 1, &err);
    if (err != OPUS_OK || !s_opus_dec) {
        ESP_LOGE(TAG, "Opus decoder create failed: %d", err);
        return false;
    }

    ESP_LOGI(TAG, "Opus encoder/decoder initialised (16kHz mono, 16kbps VOIP)");
    return true;
}
#endif

/**
 * @brief Encode a 20ms PCM frame.
 * @return Number of bytes in s_opus_out, or 0 on error/passthrough.
 */
static int encode_frame(const int16_t* pcm, int samples)
{
#if defined(CONFIG_HELLUM_USE_OPUS) && CONFIG_HELLUM_USE_OPUS
    int bytes = opus_encode(s_opus_enc, pcm, samples, s_opus_out, kOpusMaxBytes);
    if (bytes < 0) {
        ESP_LOGE(TAG, "Opus encode error: %d", bytes);
        return 0;
    }
    return bytes;
#else
    // Raw PCM fallback — copy directly
    int byte_count = samples * sizeof(int16_t);
    if (byte_count > (int)kOpusMaxBytes) byte_count = kOpusMaxBytes;
    memcpy(s_opus_out, pcm, byte_count);
    return byte_count;
#endif
}

/**
 * @brief Decode an incoming Opus frame into PCM.
 * @return Number of decoded samples, or 0 on error.
 */
static int decode_frame(const uint8_t* data, int data_len,
                         int16_t* pcm_out, int max_samples)
{
#if defined(CONFIG_HELLUM_USE_OPUS) && CONFIG_HELLUM_USE_OPUS
    int samples = opus_decode(s_opus_dec, data, data_len, pcm_out, max_samples, 0);
    if (samples < 0) {
        ESP_LOGE(TAG, "Opus decode error: %d", samples);
        return 0;
    }
    return samples;
#else
    // Raw PCM fallback
    int samples_available = data_len / sizeof(int16_t);
    if (samples_available > max_samples) samples_available = max_samples;
    memcpy(pcm_out, data, samples_available * sizeof(int16_t));
    return samples_available;
#endif
}

// ═════════════════════════════════════════════════════════════════
//  WebSocket Event Handler
// ═════════════════════════════════════════════════════════════════
static void ws_event_handler(void* arg, esp_event_base_t event_base,
                              int32_t event_id, void* event_data)
{
    auto* data = static_cast<esp_websocket_event_data_t*>(event_data);

    switch (event_id) {
        case WEBSOCKET_EVENT_CONNECTED:
            ESP_LOGI(TAG, "WebSocket connected");
            xEventGroupSetBits(s_ctx->event_group, BIT_WS_CONNECTED);
            break;

        case WEBSOCKET_EVENT_DISCONNECTED:
            ESP_LOGW(TAG, "WebSocket disconnected");
            xEventGroupClearBits(s_ctx->event_group, BIT_WS_CONNECTED);
            break;

        case WEBSOCKET_EVENT_DATA:
            // ── Handle incoming data based on opcode ──
            if (data->op_code == 0x01) {
                // Text frame → JSON control message
                // Check for end-of-stream marker
                cJSON* json = cJSON_ParseWithLength(
                    static_cast<const char*>(data->data_ptr), data->data_len);
                if (json) {
                    cJSON* event = cJSON_GetObjectItem(json, "event");
                    if (event && cJSON_IsString(event)) {
                        if (strcmp(event->valuestring, "response_start") == 0) {
                            ESP_LOGI(TAG, "Server: response_start");
                            s_ctx->state.store(HellumState::RESPONSE);
                        } else if (strcmp(event->valuestring, "response_end") == 0) {
                            ESP_LOGI(TAG, "Server: response_end");
                            // Push a final chunk marker
                            TtsChunk final_chunk = {};
                            final_chunk.length  = 0;
                            final_chunk.is_last = true;
                            xQueueSend(s_ctx->tts_queue, &final_chunk, portMAX_DELAY);
                        }
                    }
                    cJSON_Delete(json);
                }
            } else if (data->op_code == 0x02) {
                // Binary frame → TTS audio data
                HellumState current = s_ctx->state.load();
                if (current == HellumState::RESPONSE ||
                    current == HellumState::SERVER_PROCESSING) {

                    // Ensure we're in RESPONSE state
                    if (current == HellumState::SERVER_PROCESSING) {
                        s_ctx->state.store(HellumState::RESPONSE);
                    }

                    // Decode the incoming audio
                    TtsChunk chunk = {};
                    chunk.length = decode_frame(
                        reinterpret_cast<const uint8_t*>(data->data_ptr),
                        data->data_len,
                        chunk.samples,
                        kOpusFrameSamples);
                    chunk.is_last = false;

                    if (chunk.length > 0) {
                        // Non-blocking push — drop if queue full (prevents stall)
                        if (xQueueSend(s_ctx->tts_queue, &chunk, 0) != pdTRUE) {
                            ESP_LOGW(TAG, "TTS queue full — dropping chunk");
                        }
                    }
                }
            }
            break;

        case WEBSOCKET_EVENT_ERROR:
            ESP_LOGE(TAG, "WebSocket error");
            break;

        default:
            break;
    }
}

// ═════════════════════════════════════════════════════════════════
//  WebSocket Init
// ═════════════════════════════════════════════════════════════════

// Use the default CA bundle for mbedTLS (enabled via menuconfig)
#include "esp_crt_bundle.h"

static bool ws_init(HellumContext* ctx)
{
    esp_websocket_client_config_t ws_cfg = {};
    ws_cfg.uri = ctx->ws_uri;
    ws_cfg.buffer_size = 2048;
    ws_cfg.task_stack  = 4096;
    
    // Enable SSL/TLS with the default certificate bundle
    ws_cfg.crt_bundle_attach = esp_crt_bundle_attach;

    s_ws_client = esp_websocket_client_init(&ws_cfg);
    if (!s_ws_client) {
        ESP_LOGE(TAG, "WebSocket client init failed");
        return false;
    }

    esp_websocket_register_events(s_ws_client,
                                   WEBSOCKET_EVENT_ANY,
                                   ws_event_handler,
                                   nullptr);

    esp_err_t err = esp_websocket_client_start(s_ws_client);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "WebSocket start failed: %s", esp_err_to_name(err));
        return false;
    }

    ESP_LOGI(TAG, "WebSocket client started → %s", ctx->ws_uri);
    return true;
}

// ═════════════════════════════════════════════════════════════════
//  Send JSON control message
// ═════════════════════════════════════════════════════════════════
static void ws_send_json(const char* event_name)
{
    if (!esp_websocket_client_is_connected(s_ws_client)) return;

    cJSON* root = cJSON_CreateObject();
    cJSON_AddStringToObject(root, "event", event_name);
    char* json_str = cJSON_PrintUnformatted(root);

    esp_websocket_client_send_text(s_ws_client, json_str, strlen(json_str),
                                    portMAX_DELAY);

    ESP_LOGI(TAG, "Sent: %s", json_str);
    free(json_str);
    cJSON_Delete(root);
}

// ═════════════════════════════════════════════════════════════════
//  Stream audio from Ring Buffer to WebSocket
// ═════════════════════════════════════════════════════════════════

/**
 * @brief Send pre-roll + live audio until BIT_STREAM_STOP.
 *
 * 1. Read pre-roll from ring buffer (wake word duration + 500ms margin)
 * 2. Encode and send in 20ms Opus frames
 * 3. Transition to live streaming
 * 4. Continue until VAD triggers stream stop
 */
static void stream_audio_to_server(HellumContext* ctx)
{
    RingBuffer* rb = ctx->ring_buffer;

    // ── Send "wake" event to server ──────────────────────────
    ws_send_json("wake");

    // ── Pre-roll: read from buffer's past ────────────────────
    const size_t preroll_samples = kPrerollSamples;
    size_t read_idx = rb->GetPrerollStart(preroll_samples);

    ESP_LOGI(TAG, "Streaming pre-roll: %zu samples from idx %zu",
             preroll_samples, read_idx);

    // Pre-roll buffer (one Opus frame at a time)
    int16_t frame_buf[kOpusFrameSamples];
    size_t preroll_sent = 0;

    while (preroll_sent < preroll_samples) {
        size_t remaining = preroll_samples - preroll_sent;
        size_t to_read = (remaining > kOpusFrameSamples) ?
                          kOpusFrameSamples : remaining;

        size_t got = rb->ReadFrom(read_idx, frame_buf, to_read);
        if (got == 0) {
            vTaskDelay(1);  // Wait for more data
            continue;
        }

        // Pad short frame with zeros
        if (got < kOpusFrameSamples) {
            memset(frame_buf + got, 0,
                   (kOpusFrameSamples - got) * sizeof(int16_t));
        }

        int encoded_bytes = encode_frame(frame_buf, kOpusFrameSamples);
        if (encoded_bytes > 0 && esp_websocket_client_is_connected(s_ws_client)) {
            esp_websocket_client_send_bin(s_ws_client,
                                           reinterpret_cast<const char*>(s_opus_out),
                                           encoded_bytes,
                                           portMAX_DELAY);
        }

        read_idx     += got;
        preroll_sent += got;
    }

    ESP_LOGI(TAG, "Pre-roll complete. Starting live stream...");
    ctx->state.store(HellumState::STREAMING);

    // ── Live streaming: follow the ring buffer's write head ──
    while (true) {
        // Check if VAD has triggered stop
        EventBits_t bits = xEventGroupWaitBits(
            ctx->event_group, BIT_STREAM_STOP, pdTRUE, pdFALSE, 0);

        if (bits & BIT_STREAM_STOP) {
            ESP_LOGI(TAG, "Stream stop signal received");
            break;
        }

        // Read next frame from ring buffer
        size_t got = rb->ReadFrom(read_idx, frame_buf, kOpusFrameSamples);
        if (got < kOpusFrameSamples) {
            // Not enough new data yet — wait a bit
            vTaskDelay(pdMS_TO_TICKS(5));
            continue;
        }

        int encoded_bytes = encode_frame(frame_buf, kOpusFrameSamples);
        if (encoded_bytes > 0 && esp_websocket_client_is_connected(s_ws_client)) {
            esp_websocket_client_send_bin(s_ws_client,
                                           reinterpret_cast<const char*>(s_opus_out),
                                           encoded_bytes,
                                           portMAX_DELAY);
        }

        read_idx += got;
    }

    // ── Send stop event ──────────────────────────────────────
    ws_send_json("stop");
    ESP_LOGI(TAG, "Live stream ended. Waiting for server response...");
}

// ═════════════════════════════════════════════════════════════════
//  Main Network Task Loop
// ═════════════════════════════════════════════════════════════════
static void network_task_entry(void* arg)
{
    auto* ctx = static_cast<HellumContext*>(arg);
    s_ctx = ctx;

    // ── Initialise Opus codec ────────────────────────────────
#if defined(CONFIG_HELLUM_USE_OPUS) && CONFIG_HELLUM_USE_OPUS
    if (!opus_init()) {
        ESP_LOGE(TAG, "Opus init failed — will use raw PCM");
    }
#endif

    // ── Wait for Wi-Fi ───────────────────────────────────────
    ESP_LOGI(TAG, "Waiting for Wi-Fi connection...");
    xEventGroupWaitBits(ctx->event_group, BIT_WIFI_CONNECTED,
                        pdFALSE, pdTRUE, portMAX_DELAY);

    // ── Start WebSocket client ───────────────────────────────
    if (!ws_init(ctx)) {
        ESP_LOGE(TAG, "WebSocket init failed — retrying in 5s");
        vTaskDelay(pdMS_TO_TICKS(5000));
        // In production, implement exponential backoff retry here
    }

    // ── Wait for WebSocket connection ────────────────────────
    ESP_LOGI(TAG, "Waiting for WebSocket connection...");
    xEventGroupWaitBits(ctx->event_group, BIT_WS_CONNECTED,
                        pdFALSE, pdTRUE, portMAX_DELAY);
    ESP_LOGI(TAG, "Network task ready. Waiting for wake word...");

    // ── Main event loop ──────────────────────────────────────
    while (true) {
        // Wait for wake word detection (blocks until signalled)
        EventBits_t bits = xEventGroupWaitBits(
            ctx->event_group, BIT_WAKE_DETECTED,
            pdTRUE,    // Clear on exit
            pdFALSE,   // Any bit
            portMAX_DELAY);

        if (bits & BIT_WAKE_DETECTED) {
            ESP_LOGI(TAG, "Wake detected — beginning pre-roll stream");

            if (esp_websocket_client_is_connected(s_ws_client)) {
                // Stream pre-roll + live audio
                stream_audio_to_server(ctx);
            } else {
                ESP_LOGW(TAG, "WebSocket not connected — skipping stream");
                ctx->state.store(HellumState::POST_RESPONSE);
            }
        }
    }
}

// ═════════════════════════════════════════════════════════════════
//  Public entry point
// ═════════════════════════════════════════════════════════════════
void network_task_start(HellumContext* ctx)
{
    xTaskCreatePinnedToCore(
        network_task_entry,
        "net_task",
        8192,           // 8KB stack
        ctx,
        4,              // Priority 4 (below audio task)
        nullptr,
        0               // Core 0 (Protocol CPU)
    );

    ESP_LOGI(TAG, "Network task created (Core 0, priority 4)");
}
