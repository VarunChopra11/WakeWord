# Wake Word Detection Firmware
## ESP32-S3-N16R8 · ESP-IDF · MicroWakeWord

---

## Hardware

| Component       | Details                                      |
|-----------------|----------------------------------------------|
| MCU             | ESP32-S3-N16R8 (Xtensa LX7 dual-core 240 MHz) |
| Flash           | 16 MB (QIO 80 MHz)                           |
| PSRAM           | 8 MB Octal SPI                               |
| Microphones     | 2 × INMP441 (I2S digital, 16 kHz)           |
| LED             | Common-cathode RGB (GPIO 47/48/21)           |

### Pin Assignments

| Signal      | GPIO |
|-------------|------|
| I2S SCK     | 41   |
| I2S WS      | 42   |
| I2S SD (Mic 1, Left — L/R → GND) | 2 |
| LED Red     | 48   |
| LED Green   | 47   |
| LED Blue    | 21   |

> **Microphone orientation**
> Mic 1 (Left channel): L/R pin → GND  
> Mic 2 (Right channel): L/R pin → 3.3 V  
> Only the left channel is read; both clocks are shared.

---

## Project Structure

```
wake_word_detection/
├── CMakeLists.txt                  — Top-level build entry
├── partitions.csv                  — 5 MB factory app + 11 MB storage
├── sdkconfig.defaults              — Octal PSRAM, 240 MHz, O3, 16 MB flash
│
├── main/
│   ├── CMakeLists.txt
│   ├── idf_component.yml           — Declares esp-tflite-micro dependency
│   └── main.cpp                    — I2S init, streaming loop, LED logic
│
└── components/
    ├── audio_preprocessor/
    │   ├── CMakeLists.txt
    │   ├── include/
    │   │   └── audio_preprocessor.h
    │   └── src/
    │       ├── audio_preprocessor.cpp   — Pipeline orchestration
    │       ├── fft_util.{h,cpp}         — esp-dsp FFT wrapper
    │       └── mel_filterbank.{h,cpp}   — 40-band triangular mel filterbank
    │
    └── model/
        ├── CMakeLists.txt
        ├── include/
        │   ├── model_runner.h           — Interpreter lifecycle API
        │   └── model_data.h             — Model array declaration
        └── src/
            ├── model_runner.cpp         — MicroInterpreter + ResourceVariables
            └── model_data.cc            — TFLite flatbuffer (replace placeholder)
```

---

## Prerequisites

### 1. Install ESP-IDF v5.1 or later

```bash
git clone --recursive https://github.com/espressif/esp-idf.git ~/esp/esp-idf
cd ~/esp/esp-idf
git checkout v5.2.1     # Tested version
./install.sh esp32s3
source export.sh
```

### 2. Install ESP-IDF Component Manager dependencies

The `main/idf_component.yml` declares:
- `espressif/esp-tflite-micro >= 1.3.1`
- `espressif/esp-dsp >= 1.0.0`

These are downloaded automatically on first build.

---

## Adding Your Model

1. Place your `.tflite` file in the project root:
   ```bash
   cp ~/models/your_wakeword.tflite ./wakeword.tflite
   ```

2. Run the provided helper script:
   ```bash
   chmod +x tools/convert_model.sh
   ./tools/convert_model.sh wakeword.tflite
   ```
   This generates `components/model/src/model_data.cc` with the correct
   array name, alignment attribute, and length constant.

3. **Or** do it manually:
   ```bash
   xxd -i wakeword.tflite | sed \
     's/unsigned char .*/const uint8_t g_model_data[] __attribute__((aligned(8))) = {/' | \
     sed 's/unsigned int .*/const size_t g_model_data_len = sizeof(g_model_data);/' \
     > components/model/src/model_data.cc

   # Prepend the required header include
   sed -i '1s/^/#include "model_data.h"\n\n/' components/model/src/model_data.cc
   ```

### Verifying Model Compatibility

Your `.tflite` must match these specifications (from model analysis):

| Property       | Expected Value                        |
|----------------|---------------------------------------|
| Total size     | ~115,400 bytes                        |
| Signature      | `serving_default`                     |
| Input name     | `input_audio`                         |
| Input shape    | `[1, 1, 40]`                          |
| Input dtype    | `INT8`                                |
| Output name    | `dense_1`                             |
| Output shape   | `[1, 1]`                              |
| Output dtype   | `UINT8`                               |
| Subgraphs      | 2 (main inference + state init)       |
| VAR_HANDLE ops | 11 state buffers (stream_11–stream_21)|
| Resource vars  | 22 slots allocated                    |

---

## Building

```bash
cd wake_word_detection

# Set target
idf.py set-target esp32s3

# Build (component manager downloads dependencies automatically)
idf.py build
```

### First build output

Expected flash usage:
```
Project Memory Type Usage Summary:
  DRAM .data size:     ~12 KB
  DRAM .bss  size:     ~6  KB
  Flash code size:     ~1.2 MB   (TFLite Micro runtime)
  Flash rodata size:   ~115 KB   (model weights)
  Total binary size:   ~1.4 MB   (well within 5 MB factory partition)
```

---

## Flashing

```bash
# Replace /dev/ttyUSB0 with your actual port
idf.py -p /dev/ttyUSB0 flash monitor
```

On macOS use `/dev/cu.usbserial-*`.

---

## Runtime Behaviour

### Boot sequence (logged over UART at 115200 baud)

```
I (xxx) WWD: === Wake Word Detection Booting ===
I (xxx) WWD: Chip: ESP32-S3-N16R8 | Flash: 16MB | PSRAM: 8MB Octal
I (xxx) AudioPreprocessor: Initialising audio preprocessor
I (xxx) AudioPreprocessor:   Window: 30 ms (480 samples)
I (xxx) AudioPreprocessor:   Stride: 10 ms (160 samples)
I (xxx) AudioPreprocessor:   FFT:    512 points → 257 bins
I (xxx) AudioPreprocessor:   Mels:   40 channels
I (xxx) mel_fb: Mel filterbank initialised: 40 bands, 20–7600 Hz
I (xxx) ModelRunner: Initialising ModelRunner
I (xxx) ModelRunner:   Arena:              150 KB in PSRAM
I (xxx) ModelRunner:   Resource variables: 22
I (xxx) ModelRunner:   MicroResourceVariables created (22 slots)
I (xxx) ModelRunner:   Input  tensor: shape=[1,1,40] dtype=9
I (xxx) ModelRunner:   Output tensor: shape=[1,1] dtype=3
I (xxx) ModelRunner: ModelRunner ready — streaming state active
I (xxx) WWD: I2S initialised: 16000 Hz, mono, 16-bit
I (xxx) WWD: All systems ready. Starting inference task...
I (xxx) WWD: Streaming task started (stride=160 samples)
```

### Wake word detection

When a wake word is detected (output probability > 200/255 ≈ 78%):

```
I (xxx) WWD: >>> WAKE WORD DETECTED  (prob=217/255) <<<
```

The **green LED** (GPIO 47) turns on for **2 seconds**.

---

## Tuning

### Detection threshold

In `main/main.cpp`:
```cpp
static constexpr uint8_t kDetectThresh = 200;  // 200/255 ≈ 78.4%
```
Increase to reduce false positives; decrease for higher sensitivity.

### Feature quantisation

In `components/audio_preprocessor/include/audio_preprocessor.h`:
```cpp
static constexpr float kFeatureScale = 1.0f / 52.0f;
static constexpr int   kFeatureZp    = -30;
```
These must match the `input_quantization` parameters baked into your `.tflite`
model.  Retrieve them with:
```bash
python3 - <<'EOF'
import flatbuffers, tflite.Model as M
data = open("wakeword.tflite","rb").read()
model = M.Model.GetRootAsModel(bytearray(data), 0)
sg = model.Subgraphs(0)
t  = sg.Tensors(0)
print("scale    =", t.Quantization().Scale(0))
print("zero_pt  =", t.Quantization().ZeroPoint(0))
EOF
```

### Resource variable count

If your model has more or fewer VAR_HANDLE ops, update:
```cpp
// components/model/include/model_runner.h
static constexpr int kNumResourceVariables = 22;
```

### Tensor arena size

If `AllocateTensors()` fails with OOM, increase:
```cpp
static constexpr size_t kTensorArenaSize = 150 * 1024;  // in model_runner.h
```
Maximum safe value with 8 MB PSRAM is approximately 6 MB (leaving room for
heap, stack, and driver buffers).

---

## Memory Map

| Region        | Location | Size    | Contents                         |
|---------------|----------|---------|----------------------------------|
| Tensor arena  | PSRAM    | 150 KB  | TFLite tensors + state buffers   |
| Ring buffer   | PSRAM    | ~1 KB   | PCM audio (480 × int16)          |
| FFT buffers   | PSRAM    | ~5 KB   | Windowed + complex FFT data      |
| Mel table     | PSRAM    | ~10 KB  | Filterbank lookup entries        |
| Model weights | Flash    | ~115 KB | Quantised kernel data (read-only)|
| Stack         | IRAM     | 8 KB    | inference_task stack             |

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Crash at `Invoke()` | Missing VAR_HANDLE op registration | Ensure `AddVarHandle()`, `AddReadVariable()`, `AddAssignVariable()` are called |
| `AllocateTensors()` returns error | Arena too small | Increase `kTensorArenaSize` |
| No audio / all-zero features | I2S wiring | Check SCK/WS/SD pins; verify INMP441 L/R strapping |
| Always detecting wake word | Threshold too low | Increase `kDetectThresh` toward 230–240 |
| Never detecting wake word | Feature mismatch | Verify `kFeatureScale` / `kFeatureZp` against model quantisation params |
| PSRAM not detected | sdkconfig mismatch | Verify `CONFIG_SPIRAM_MODE_OCT=y` and correct flash mode |

---

## License

Proprietary — All rights reserved.
