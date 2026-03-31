/**
 * @file led_manager.cpp
 * @brief RGB LED patterns driven by Hellum Hub state machine.
 *
 * Pattern mapping:
 *   IDLE              → All off (low power)
 *   WAKE_DETECTED     → Solid green (acknowledged)
 *   STREAMING         → Pulsing blue (listening)
 *   SERVER_PROCESSING → Slow cyan blink (thinking)
 *   RESPONSE          → Solid blue (speaking)
 *   POST_RESPONSE     → Brief green flash, then off
 */

#include "led_manager.h"

#include "driver/gpio.h"
#include "esp_timer.h"

namespace led {

// ─────────────────────────────────────────────────────────────
//  Internal state for animation timing
// ─────────────────────────────────────────────────────────────
static HellumState s_last_state    = HellumState::IDLE;
static int64_t     s_state_enter_us = 0;
static bool        s_blink_on       = false;

static inline void SetRGB(int r, int g, int b)
{
    gpio_set_level(static_cast<gpio_num_t>(kPinLED_R), r);
    gpio_set_level(static_cast<gpio_num_t>(kPinLED_G), g);
    gpio_set_level(static_cast<gpio_num_t>(kPinLED_B), b);
}

// ─────────────────────────────────────────────────────────────
//  Public API
// ─────────────────────────────────────────────────────────────
void Init()
{
    gpio_config_t cfg = {};
    cfg.pin_bit_mask = (1ULL << kPinLED_R) |
                       (1ULL << kPinLED_G) |
                       (1ULL << kPinLED_B);
    cfg.mode         = GPIO_MODE_OUTPUT;
    cfg.pull_up_en   = GPIO_PULLUP_DISABLE;
    cfg.pull_down_en = GPIO_PULLDOWN_DISABLE;
    cfg.intr_type    = GPIO_INTR_DISABLE;
    gpio_config(&cfg);

    SetRGB(0, 0, 0);
}

void Off()
{
    SetRGB(0, 0, 0);
}

void Update(HellumState state)
{
    const int64_t now_us = esp_timer_get_time();

    // Detect state transitions → reset animation timers
    if (state != s_last_state) {
        s_last_state     = state;
        s_state_enter_us = now_us;
        s_blink_on       = true;
    }

    const int64_t elapsed_us = now_us - s_state_enter_us;

    switch (state) {
        case HellumState::IDLE:
            SetRGB(0, 0, 0);
            break;

        case HellumState::WAKE_DETECTED:
            // Solid green
            SetRGB(0, 1, 0);
            break;

        case HellumState::STREAMING:
            // Pulsing blue — toggle every 300ms
            if ((elapsed_us / 300000) % 2 == 0) {
                SetRGB(0, 0, 1);
            } else {
                SetRGB(0, 0, 0);
            }
            break;

        case HellumState::SERVER_PROCESSING:
            // Slow cyan blink — toggle every 500ms
            if ((elapsed_us / 500000) % 2 == 0) {
                SetRGB(0, 1, 1);
            } else {
                SetRGB(0, 0, 0);
            }
            break;

        case HellumState::RESPONSE:
            // Solid blue — playing TTS
            SetRGB(0, 0, 1);
            break;

        case HellumState::POST_RESPONSE:
            // Brief green flash for 500ms, then off
            if (elapsed_us < 500000) {
                SetRGB(0, 1, 0);
            } else {
                SetRGB(0, 0, 0);
            }
            break;
    }
}

}  // namespace led
