# Hellum Hub — ESP32-S3 Dual-Core AI Voice Assistant Firmware

Dual-core voice assistant firmware for the **ESP32-S3-N16R8**, combining on-device TFLite Micro wake word detection with real-time audio streaming, WebSocket networking, and speaker playback.

## Architecture

```
┌─── Core 1 (Real-Time Audio & AI) ──────────────────────────────────────────┐
│  I2S RX (Stereo) → Beamform → Ring Buffer Write                           │
│                   → AudioPreprocessor → ModelRunner → Wake Detect          │
│                   → VAD (during STREAMING)                                 │
│  I2S TX ← TTS PCM Queue (during RESPONSE)                                 │
└────────────────────────────────────────────────────────────────────────────-┘

┌─── Core 0 (Network & Connectivity) ───────────────────────────────────────-┐
│  Wi-Fi STA → WebSocket Client (persistent)                                 │
│  Ring Buffer Read → Opus Encode → WS Binary Frames (STREAMING)             │
│  WS Binary Frames → Opus Decode → TTS PCM Queue (RESPONSE)                │
└────────────────────────────────────────────────────────────────────────────-┘
```

### State Machine

```
IDLE → WAKE_DETECTED → STREAMING → SERVER_PROCESSING → RESPONSE → POST_RESPONSE → IDLE
```

| State | Core 1 | Core 0 | LED |
|---|---|---|---|
| IDLE | Wake word inference | WebSocket dormant | Off |
| WAKE_DETECTED | Continue recording | Send pre-roll audio | Solid green |
| STREAMING | VAD silence detection | Live Opus streaming | Pulsing blue |
| SERVER_PROCESSING | Idle | Wait for response | Cyan blink |
| RESPONSE | I2S TX playback | Decode TTS chunks | Solid blue |
| POST_RESPONSE | Reset → IDLE | — | Green flash |

## Hardware

| Component | Details |
|---|---|
| MCU | ESP32-S3-N16R8 (dual-core 240 MHz, 16 MB Flash, 8 MB Octal PSRAM) |
| Microphones | 2× INMP441 I2S MEMS (stereo: L/R channel separation for beamforming) |
| Amplifier | MAX98357A I2S DAC (12dB gain, 5V supply) |
| LED | Common-cathode RGB LED |
| Privacy | SPDT mute switch on mic data line |

### Pin Mapping

| Signal | GPIO | Notes |
|---|---|---|
| I2S SCK (BCLK) | 41 | Shared: mics + amp |
| I2S WS (LRCLK) | 42 | Shared: mics + amp |
| I2S SD (Mics In) | 2 | Via mute switch |
| I2S DIN (Amp Out) | 1 | To MAX98357A |
| LED Red | 48 | |
| LED Green | 47 | |
| LED Blue | 21 | |

## Project Structure

```
WakeWord/
├── CMakeLists.txt              # Root project — registers components
├── partitions.csv              # 5MB app + 11MB storage
├── sdkconfig.defaults          # PSRAM, Wi-Fi, networking config
├── alexa.tflite                # Source wake word model
│
├── main/
│   ├── CMakeLists.txt
│   ├── idf_component.yml       # esp-tflite-micro + esp_websocket_client
│   ├── main.cpp                # Entry: NVS, Wi-Fi, dual-core task launch
│   ├── hellum_types.h          # State machine, event bits, shared context
│   ├── audio_task.h/cpp        # Core 1: I2S, beamforming, WW, VAD, playback
│   ├── network_task.h/cpp      # Core 0: WebSocket, Opus, streaming
│   ├── led_manager.h/cpp       # State-aware RGB LED patterns
│
├── ring_buffer/                # PSRAM circular buffer (3s, 96KB)
│   ├── CMakeLists.txt
│   ├── ring_buffer.h/cpp
│
├── audio_preprocessor/         # TFLite Micro feature extraction (UNCHANGED)
│   ├── audio_preprocessor.h/cpp
│   └── audio_preprocessor_int8_model_data.h
│
├── model/                      # TFLite Micro wake word model (UNCHANGED)
│   ├── model_runner.h/cpp
│   ├── model_data.h/cc
│
└── tools/
    └── convert_model.sh
```

## Prerequisites

- **ESP-IDF v5.1+** ([installation guide](https://docs.espressif.com/projects/esp-idf/en/stable/esp32s3/get-started/))
- Python 3.8+

## Getting Started

### 1. Configure Wi-Fi

Edit `sdkconfig.defaults` or use `idf.py menuconfig`:
```
CONFIG_ESP_WIFI_SSID="YourSSID"
CONFIG_ESP_WIFI_PASSWORD="YourPassword"
```

Also set the WebSocket server URI in `main.cpp`:
```cpp
#define CONFIG_HELLUM_WS_URI "ws://your-server:8080/ws"
```

### 2. Build

```bash
. $IDF_PATH/export.sh
idf.py set-target esp32s3
idf.py build
```

### 3. Flash & Monitor

```bash
idf.py flash monitor
```

## Key Configuration

### Detection Threshold

In `audio_task.cpp`:
```cpp
static constexpr uint8_t kDetectThreshold = 200;  // ~78% confidence
```

### Audio Parameters

| Parameter | Value |
|---|---|
| Sample rate | 16,000 Hz |
| Window size | 30 ms (480 samples) |
| Stride | 20 ms (320 samples) |
| Mel channels | 40 |
| Ring buffer | 3 seconds (96KB, PSRAM) |
| Pre-roll | 1.3s (800ms wake word + 500ms margin) |
| VAD timeout | 800ms silence |

### Memory Layout

| Resource | Size | Location |
|---|---|---|
| Ring buffer | 96 KB | PSRAM |
| Preprocessor arena | 16 KB | PSRAM |
| Model tensor arena | 150 KB | PSRAM |
| Audio task stack | 16 KB | Internal SRAM |
| Network task stack | 8 KB | Internal SRAM |
| TTS queue | ~13 KB | Internal SRAM |

### WebSocket Protocol

| Direction | Frame Type | Content |
|---|---|---|
| Client → Server | Text | `{"event": "wake"}` — wake word detected |
| Client → Server | Binary | Opus-encoded 20ms audio frames |
| Client → Server | Text | `{"event": "stop"}` — VAD silence cutoff |
| Server → Client | Text | `{"event": "response_start"}` |
| Server → Client | Binary | Opus-encoded TTS audio |
| Server → Client | Text | `{"event": "response_end"}` |

## License

This project is part of the Hellum firmware suite.
