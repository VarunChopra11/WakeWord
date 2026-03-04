# Wake Word Detection — ESP32-S3

On-device wake word detection firmware for the **ESP32-S3-N16R8**, using TensorFlow Lite Micro to run a streaming MicroWakeWord model entirely on the edge. The system continuously listens via an INMP441 I2S MEMS microphone and triggers an RGB LED when the wake word (e.g. "Alexa") is detected.

## Architecture

```
INMP441 Mic ──I2S──▸ AudioPreprocessor ──▸ MicroWakeWord Model ──▸ LED Trigger
                     (TFLite Micro)         (TFLite Micro)
                     480 PCM → 40 INT8      [1,1,40] → probability
                     mel features            (UINT8, 0–255)
```

Every **20 ms** stride:

1. Read 320 PCM samples (16-bit mono, 16 kHz) from the I2S peripheral
2. Accumulate into a 480-sample window and generate **40 INT8 mel-frequency features** using the TFLite audio preprocessor model
3. Feed features into the **stateful streaming MicroWakeWord** model (22 VAR_HANDLE state slots maintain temporal context across frames)
4. If output probability > **~78% (200/255)**, light the green LED for 2 seconds

## Hardware

| Component | Details |
|---|---|
| MCU | ESP32-S3-N16R8 (dual-core 240 MHz, 16 MB Flash, 8 MB Octal PSRAM) |
| Microphone | INMP441 I2S MEMS (16 kHz, 32-bit stereo, left channel used) |
| LED | Common-cathode RGB LED |

### Pin Mapping

| Signal | GPIO |
|---|---|
| I2S SCK (BCLK) | 41 |
| I2S WS (LRCLK) | 42 |
| I2S SD (Data In) | 2 |
| LED Red | 48 |
| LED Green | 47 |
| LED Blue | 21 |

## Project Structure

```
WakeWord/
├── CMakeLists.txt              # Root project CMake — registers component dirs
├── partitions.csv              # Custom partition table (5 MB app, ~11 MB storage)
├── sdkconfig.defaults          # ESP-IDF Kconfig defaults
├── alexa.tflite                # Source wake word model (TFLite flatbuffer)
│
├── main/
│   ├── CMakeLists.txt
│   ├── idf_component.yml       # Declares esp-tflite-micro dependency
│   └── main.cpp                # Entry point — I2S init, streaming inference task
│
├── audio_preprocessor/         # ESP-IDF component
│   ├── CMakeLists.txt
│   ├── audio_preprocessor.h    # Feature extraction constants & class
│   ├── audio_preprocessor.cpp  # Ring buffer + TFLite preprocessor model runner
│   └── audio_preprocessor_int8_model_data.h  # Embedded preprocessor .tflite
│
├── model/                      # ESP-IDF component
│   ├── CMakeLists.txt
│   ├── model_data.h            # Declares g_model_data[] / g_model_data_len
│   ├── model_data.cc           # Auto-generated C array of the .tflite model
│   ├── model_runner.h          # Stateful interpreter wrapper
│   └── model_runner.cpp        # Arena, allocator, resource variables, inference
│
├── tools/
│   └── convert_model.sh        # Converts .tflite → model_data.cc via xxd
│
└── managed_components/         # Auto-fetched by IDF Component Manager
    ├── espressif__esp-nn/
    └── espressif__esp-tflite-micro/
```

## Prerequisites

- **ESP-IDF v5.1+** ([installation guide](https://docs.espressif.com/projects/esp-idf/en/stable/esp32s3/get-started/))
- Python 3.8+ (for ESP-IDF tools)
- `xxd` (for model conversion; usually pre-installed on Linux/macOS)

## Getting Started

### 1. Set up ESP-IDF environment

```bash
. $IDF_PATH/export.sh
```

### 2. Build

```bash
idf.py set-target esp32s3
idf.py build
```

### 3. Flash & Monitor

```bash
idf.py flash monitor
```

Press `Ctrl+]` to exit the serial monitor.

## Replacing the Wake Word Model

To use a different MicroWakeWord `.tflite` model:

```bash
./tools/convert_model.sh path/to/your_model.tflite
```

This generates `model/model_data.cc` with the model embedded as a C array. Then rebuild:

```bash
idf.py build
```

> **Note:** The new model must have a compatible input shape `[1, 1, 40]` (INT8) and output shape `[1, 1]` (UINT8). You may also need to adjust the op resolver in `model_runner.cpp` if the new model uses different TFLite ops.

## Key Configuration

### Detection Threshold

In `main.cpp`:

```cpp
static constexpr uint8_t kDetectThresh = 200;  // ~78% confidence
```

Lower values increase sensitivity (more detections, more false positives). Higher values require stronger confidence.

### Audio Parameters

Defined in `audio_preprocessor.h`:

| Parameter | Value | Description |
|---|---|---|
| Sample rate | 16,000 Hz | I2S capture rate |
| Window size | 30 ms (480 samples) | FFT window for feature extraction |
| Stride | 20 ms (320 samples) | Step between successive feature slices |
| Mel channels | 40 | INT8 features per slice |

### Memory Layout

| Resource | Size | Location |
|---|---|---|
| Preprocessor arena | 16 KB | PSRAM |
| Model tensor arena | 150 KB | PSRAM (16-byte aligned) |
| Inference task stack | 16 KB | Internal SRAM |
| Model resource variables | 22 slots | Inside tensor arena |

### Partition Table

| Name | Type | Size |
|---|---|---|
| nvs | data | 24 KB |
| phy_init | data | 4 KB |
| factory | app | 5 MB |
| storage | data (SPIFFS) | ~11 MB |

## Serial Monitor Output

Diagnostic logs are printed every ~500 ms:

```
I (12345) WWD: Vol:  8234 | Prob:  42 | feat[0..3]: -12,5,-3,8 | GOOD LEVEL (Target)
I (12845) WWD: Vol: 15200 | Prob: 215 | feat[0..3]: 20,18,12,9 | GOOD LEVEL (Target)
I (12845) WWD: >>> ALEXA DETECTED (Prob: 215) <<<
```

| Field | Meaning |
|---|---|
| Vol | Peak absolute PCM amplitude (0–32767) |
| Prob | Wake word probability (0–255; threshold = 200) |
| feat[0..3] | First 4 mel features (sanity check) |
| Status | SILENCE / WEAK SIGNAL / GOOD LEVEL / CLIPPING |

## License

This project is part of the Hellum firmware suite.
