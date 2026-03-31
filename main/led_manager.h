/**
 * @file led_manager.h
 * @brief State-aware RGB LED controller for Hellum Hub.
 *
 * Each HellumState maps to a distinct visual pattern so the user always
 * knows the device's current mode at a glance.
 */

#pragma once

#include "hellum_types.h"

namespace led {

/** @brief Configure GPIO pins for the RGB LED. Call once at boot. */
void Init();

/**
 * @brief Update the LED pattern based on the current system state.
 *
 * Should be called periodically from the audio task loop (~every 20ms)
 * to drive animations (pulsing, blinking).
 */
void Update(HellumState state);

/** @brief Force all LEDs off immediately. */
void Off();

}  // namespace led
