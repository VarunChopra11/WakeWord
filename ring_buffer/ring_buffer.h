/**
 * @file ring_buffer.h
 * @brief Thread-safe circular PCM buffer in PSRAM for the "Time Machine" pre-roll trick.
 *
 * The buffer continuously stores the last N seconds of 16kHz 16-bit mono audio.
 * Core 1 writes every 20ms stride; Core 0 reads upon wake word detection,
 * starting from a calculated pre-roll offset.
 *
 * Concurrency model:
 *   - Write (Core 1): atomic write-index update + spinlock for data copy
 *   - Read  (Core 0): spinlock-protected snapshot of write index, then memcpy
 *   - Spinlock chosen over mutex for minimal latency (~10 CPU cycles)
 */

#pragma once

#include <cstdint>
#include <cstddef>

#include "freertos/FreeRTOS.h"
#include "freertos/portmacro.h"

class RingBuffer {
public:
    /**
     * @brief Construct a ring buffer.
     * @param capacity_samples Total capacity in samples (e.g., 48000 = 3s at 16kHz)
     */
    explicit RingBuffer(size_t capacity_samples);
    ~RingBuffer();

    /**
     * @brief Allocate the backing buffer in PSRAM.
     * @return true on success.
     */
    bool Init();

    /**
     * @brief Write PCM samples into the buffer (called from Core 1).
     *
     * Overwrites oldest data when the buffer wraps. This is intentional —
     * we always want the most recent N seconds.
     *
     * @param data    Pointer to int16_t PCM samples.
     * @param count   Number of samples to write.
     */
    void Write(const int16_t* data, size_t count);

    /**
     * @brief Read samples from a position relative to the current write head.
     *
     * Used by Core 0 to read "past" audio for the pre-roll trick.
     *
     * @param samples_before_head  How many samples before the write head to start.
     * @param out                  Output buffer (caller-allocated).
     * @param count                Number of samples to read.
     * @return                     Actual samples read (may be < count if buffer not full).
     */
    size_t ReadBefore(size_t samples_before_head, int16_t* out, size_t count);

    /**
     * @brief Read samples starting from an absolute index (for sequential streaming).
     *
     * Used to continue reading after pre-roll, tracking the read position.
     *
     * @param read_idx  Absolute read index (wraps internally).
     * @param out       Output buffer.
     * @param count     Desired sample count.
     * @return          Actual samples read (limited by available new data).
     */
    size_t ReadFrom(size_t read_idx, int16_t* out, size_t count);

    /**
     * @brief Get the current absolute write position.
     *
     * This monotonically increases (never wraps); use modulo capacity_ to
     * compute the physical index.
     */
    size_t GetWritePosition() const;

    /**
     * @brief Compute the absolute read index for the pre-roll start point.
     *
     * @param preroll_samples  Total pre-roll length in samples.
     * @return Absolute index to begin reading from.
     */
    size_t GetPrerollStart(size_t preroll_samples) const;

    /** @brief Total capacity in samples. */
    size_t Capacity() const { return capacity_; }

private:
    int16_t*       buffer_   = nullptr;   // PSRAM-backed storage
    const size_t   capacity_;             // Capacity in samples
    size_t         write_pos_ = 0;        // Monotonic absolute write position
    size_t         total_written_ = 0;    // Total samples ever written
    portMUX_TYPE   spinlock_;             // ISR-safe spinlock for dual-core access
};
