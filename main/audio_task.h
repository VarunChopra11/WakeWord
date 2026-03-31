/**
 * @file audio_task.h
 * @brief Core 1 real-time audio task — I2S capture, wake word, VAD, playback.
 */

#pragma once

#include "hellum_types.h"

/**
 * @brief Start the audio pipeline task on Core 1.
 *
 * Initialises I2S in full-duplex mode (RX stereo mics + TX amplifier),
 * then enters the main processing loop. Never returns.
 *
 * @param ctx  Pointer to the shared HellumContext (must outlive the task).
 */
void audio_task_start(HellumContext* ctx);
