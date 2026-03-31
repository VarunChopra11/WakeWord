/**
 * @file hellum_types.h
 * @brief Shared type definitions, state machine, and inter-task communication
 *        primitives for the Hellum Hub firmware.
 *
 * This header is the single source of truth for:
 *   - Hardware GPIO assignments
 *   - System state machine enum & FreeRTOS event-group bits
 *   - Shared context struct used by both Core 0 and Core 1 tasks
 */

#pragma once

#include <cstdint>
#include <atomic>

#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/queue.h"

// Forward declarations
class AudioPreprocessor;
class ModelRunner;
class RingBuffer;

// ═════════════════════════════════════════════════════════════════
//  Hardware Pin Definitions
// ═════════════════════════════════════════════════════════════════

// I2S Bus (shared between dual INMP441 mics and MAX98357A amplifier)
static constexpr int kPinI2S_SCK  = 41;   // BCLK — shared clock
static constexpr int kPinI2S_WS   = 42;   // LRCLK / Word Select — shared
static constexpr int kPinI2S_DIN  = 2;    // Data In from mics (via mute switch)
static constexpr int kPinI2S_DOUT = 1;    // Data Out to MAX98357A amplifier

// RGB LED (common-anode: LOW = on, HIGH = off)
static constexpr int kPinLED_R = 48;
static constexpr int kPinLED_G = 47;
static constexpr int kPinLED_B = 21;

// ═════════════════════════════════════════════════════════════════
//  Audio Configuration
// ═════════════════════════════════════════════════════════════════

static constexpr int kSampleRate       = 16000;   // Hz
static constexpr int kBitsPerSample    = 16;
static constexpr int kBytesPerSample   = 2;

// Ring Buffer: 3 seconds of continuous audio in PSRAM
static constexpr int kRingBufferDurationSec = 3;
static constexpr int kRingBufferSamples     = kSampleRate * kRingBufferDurationSec; // 48000
static constexpr int kRingBufferBytes       = kRingBufferSamples * kBytesPerSample; // 96000

// Pre-roll: wake word duration + margin sent before live stream begins
static constexpr int kWakeWordDurationMs  = 800;   // Typical wake word length
static constexpr int kPrerollMarginMs     = 500;   // Extra context before wake word
static constexpr int kPrerollTotalMs      = kWakeWordDurationMs + kPrerollMarginMs;
static constexpr int kPrerollSamples      = (kSampleRate * kPrerollTotalMs) / 1000; // 20800

// VAD silence threshold for stream cutoff
static constexpr int kVadSilenceTimeoutMs = 800;

// Opus frame duration
static constexpr int kOpusFrameMs      = 20;
static constexpr int kOpusFrameSamples = (kSampleRate * kOpusFrameMs) / 1000; // 320

// TTS playback queue depth (number of PCM frame buffers)
static constexpr int kTtsQueueDepth = 20;

// ═════════════════════════════════════════════════════════════════
//  State Machine
// ═════════════════════════════════════════════════════════════════

/**
 * @brief Hellum Hub operational states.
 *
 * State flow:
 *   IDLE → WAKE_DETECTED → STREAMING → SERVER_PROCESSING → RESPONSE → POST_RESPONSE → IDLE
 */
enum class HellumState : uint8_t {
    IDLE              = 0,  // Listening for wake word
    WAKE_DETECTED     = 1,  // Wake word fired — pre-roll being sent
    STREAMING         = 2,  // Live audio streaming to server
    SERVER_PROCESSING = 3,  // Waiting for server response
    RESPONSE          = 4,  // Playing TTS response from server
    POST_RESPONSE     = 5,  // Cleanup, reset, return to IDLE
};

// ═════════════════════════════════════════════════════════════════
//  FreeRTOS Event Group Bits
//
//  Used for lock-free inter-core signaling between the audio task
//  (Core 1) and the network task (Core 0).
// ═════════════════════════════════════════════════════════════════

static constexpr EventBits_t BIT_WAKE_DETECTED    = (1 << 0);  // Audio → Network: wake word fired
static constexpr EventBits_t BIT_PREROLL_DONE     = (1 << 1);  // Network → Audio: pre-roll sent, go live
static constexpr EventBits_t BIT_STREAM_STOP      = (1 << 2);  // Audio → Network: VAD silence detected
static constexpr EventBits_t BIT_SERVER_DONE      = (1 << 3);  // Network → Audio: server response complete
static constexpr EventBits_t BIT_RESPONSE_CHUNK   = (1 << 4);  // Network → Audio: TTS chunk in queue
static constexpr EventBits_t BIT_RESPONSE_DONE    = (1 << 5);  // Network → Audio: end of TTS stream
static constexpr EventBits_t BIT_WIFI_CONNECTED   = (1 << 6);  // Main → All: Wi-Fi up
static constexpr EventBits_t BIT_WS_CONNECTED     = (1 << 7);  // Network: WebSocket connected

static constexpr EventBits_t BIT_ALL              = 0xFF;

// ═════════════════════════════════════════════════════════════════
//  Shared Context
//
//  Allocated once in main(), then passed by pointer to both tasks.
//  All members are either atomic, behind FreeRTOS primitives, or
//  are written once during init and then read-only.
// ═════════════════════════════════════════════════════════════════

/**
 * @brief PCM audio chunk for TTS playback queue.
 *
 * Network task decodes Opus → fills this → pushes to tts_queue.
 * Audio task pops this → writes to I2S TX.
 */
struct TtsChunk {
    int16_t  samples[kOpusFrameSamples];    // 320 samples = 20ms
    uint16_t length;                         // Actual sample count (may be < 320 at end)
    bool     is_last;                        // Marks end of TTS stream
};

struct HellumContext {
    // ── State ──────────────────────────────────────────────────
    std::atomic<HellumState> state{HellumState::IDLE};

    // ── FreeRTOS Primitives ───────────────────────────────────
    EventGroupHandle_t event_group = nullptr;
    QueueHandle_t      tts_queue   = nullptr;  // Queue of TtsChunk*

    // ── Shared Objects (init once, then read-only pointers) ───
    RingBuffer*        ring_buffer     = nullptr;
    AudioPreprocessor* preprocessor    = nullptr;
    ModelRunner*       model           = nullptr;

    // ── Configuration (set during init) ───────────────────────
    const char* ws_uri     = nullptr;  // WebSocket server URI
    const char* wifi_ssid  = nullptr;
    const char* wifi_pass  = nullptr;
};
