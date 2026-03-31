/**
 * @file network_task.h
 * @brief Core 0 network task — Wi-Fi, WebSocket, Opus streaming, TTS decode.
 */

#pragma once

#include "hellum_types.h"

/**
 * @brief Start the network task on Core 0.
 *
 * Maintains WebSocket connection and handles all network I/O:
 * streaming encoded audio to the server and receiving TTS responses.
 *
 * @param ctx  Pointer to the shared HellumContext.
 */
void network_task_start(HellumContext* ctx);
