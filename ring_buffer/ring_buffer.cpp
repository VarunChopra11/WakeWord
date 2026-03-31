/**
 * @file ring_buffer.cpp
 * @brief PSRAM-backed circular PCM buffer for the Hellum Hub "Time Machine".
 *
 * Memory layout:
 *   buffer_[0 .. capacity_-1] — continuous int16_t array in Octal PSRAM
 *
 * Write semantics:
 *   write_pos_ is a monotonically increasing absolute counter. The physical
 *   index is (write_pos_ % capacity_). When write_pos_ exceeds capacity_,
 *   older data is silently overwritten — this is by design.
 *
 * Read semantics:
 *   ReadBefore() computes an absolute start index relative to write_pos_ and
 *   copies data out with proper wrap-around handling.
 */

#include "ring_buffer.h"

#include <cstring>
#include <algorithm>

#include "esp_log.h"
#include "esp_heap_caps.h"

static const char* TAG = "RingBuffer";

// ─────────────────────────────────────────────────────────────
//  Construction / Destruction
// ─────────────────────────────────────────────────────────────
RingBuffer::RingBuffer(size_t capacity_samples)
    : capacity_(capacity_samples)
{
    portMUX_INITIALIZE(&spinlock_);
}

RingBuffer::~RingBuffer()
{
    if (buffer_) {
        heap_caps_free(buffer_);
        buffer_ = nullptr;
    }
}

// ─────────────────────────────────────────────────────────────
//  Init — allocate in PSRAM
// ─────────────────────────────────────────────────────────────
bool RingBuffer::Init()
{
    const size_t alloc_bytes = capacity_ * sizeof(int16_t);

    buffer_ = static_cast<int16_t*>(
        heap_caps_calloc(capacity_, sizeof(int16_t),
                         MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));

    if (!buffer_) {
        ESP_LOGE(TAG, "Failed to allocate %zu bytes in PSRAM", alloc_bytes);
        return false;
    }

    write_pos_     = 0;
    total_written_ = 0;

    ESP_LOGI(TAG, "Allocated %zu bytes (%zu samples, %.1fs) in PSRAM @ %p",
             alloc_bytes, capacity_,
             static_cast<float>(capacity_) / 16000.0f,
             static_cast<void*>(buffer_));

    return true;
}

// ─────────────────────────────────────────────────────────────
//  Write — called from Core 1 audio task
// ─────────────────────────────────────────────────────────────
void RingBuffer::Write(const int16_t* data, size_t count)
{
    if (!buffer_ || count == 0) return;

    taskENTER_CRITICAL(&spinlock_);

    for (size_t i = 0; i < count; ++i) {
        buffer_[write_pos_ % capacity_] = data[i];
        ++write_pos_;
    }
    total_written_ += count;

    taskEXIT_CRITICAL(&spinlock_);
}

// ─────────────────────────────────────────────────────────────
//  ReadBefore — read N samples before the current write head
// ─────────────────────────────────────────────────────────────
size_t RingBuffer::ReadBefore(size_t samples_before_head,
                               int16_t* out,
                               size_t count)
{
    if (!buffer_ || count == 0 || !out) return 0;

    taskENTER_CRITICAL(&spinlock_);

    // Clamp to available data
    const size_t available = std::min(total_written_, capacity_);
    if (samples_before_head > available) {
        samples_before_head = available;
    }

    // Compute physical start index
    // write_pos_ points to the NEXT write location
    size_t start_abs = 0;
    if (write_pos_ >= samples_before_head) {
        start_abs = write_pos_ - samples_before_head;
    }
    // else: buffer hasn't filled enough, start from 0

    const size_t actual_count = std::min(count, samples_before_head);
    const size_t snap_write   = write_pos_;

    taskEXIT_CRITICAL(&spinlock_);

    // Copy with wrap-around (outside critical section — data is stable enough
    // for audio; worst case we get a slightly stale sample at the boundary)
    for (size_t i = 0; i < actual_count; ++i) {
        out[i] = buffer_[(start_abs + i) % capacity_];
    }

    return actual_count;
}

// ─────────────────────────────────────────────────────────────
//  ReadFrom — sequential read from an absolute index
// ─────────────────────────────────────────────────────────────
size_t RingBuffer::ReadFrom(size_t read_idx, int16_t* out, size_t count)
{
    if (!buffer_ || count == 0 || !out) return 0;

    taskENTER_CRITICAL(&spinlock_);
    const size_t current_write = write_pos_;
    taskEXIT_CRITICAL(&spinlock_);

    // Don't read ahead of write position
    if (read_idx >= current_write) return 0;

    const size_t available = current_write - read_idx;
    const size_t actual    = std::min(count, available);

    for (size_t i = 0; i < actual; ++i) {
        out[i] = buffer_[(read_idx + i) % capacity_];
    }

    return actual;
}

// ─────────────────────────────────────────────────────────────
//  Position accessors
// ─────────────────────────────────────────────────────────────
size_t RingBuffer::GetWritePosition() const
{
    return write_pos_;  // Atomic-width read on 32-bit Xtensa
}

size_t RingBuffer::GetPrerollStart(size_t preroll_samples) const
{
    const size_t wp = write_pos_;
    if (wp >= preroll_samples) {
        return wp - preroll_samples;
    }
    return 0;  // Buffer hasn't filled enough; start from beginning
}
