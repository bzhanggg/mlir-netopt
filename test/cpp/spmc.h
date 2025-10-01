#ifndef SPMC_H_
#define SPMC_H_

#include <vector>
#include <atomic>
#include <optional>

namespace cpp {
namespace parallel {

template<class T>
class Spmc {
public:
explicit Spmc<T>(size_t size)
: buf{static_cast<int>(size)}
, write_idx_{0}
, read_idx_{0}
, capacity{size}
{
}

// Producer
bool push_back(const T& val) {
    size_t curr_write = write_idx_.load(std::memory_order_relaxed);
    size_t next_write = (curr_write + 1) % capacity;

    // check if queue is full
    if (next_write == read_idx_.load(std::memory_order_acquire)) {
        return false;
    }

    buf[curr_write] = val;
    write_idx_.store(next_write, std::memory_order_release);
    return true;
}

// Consumer
std::optional<T> pop_front() {
    while (true) {
        size_t curr_read = read_idx_.load(std::memory_order_relaxed);
    
        // check if queue is empty
        if (curr_read == write_idx_.load(std::memory_order_acquire)) {
            return std::nullopt;
        }

        T val = buf[curr_read];
        size_t next_read = (curr_read + 1) % capacity;

        if (read_idx_.compare_exchange_weak(curr_read, next_read, std::memory_order_release, std::memory_order_relaxed)) {
            return std::optional<T>(val);
        }
    }
}

operator bool() const {
    return read_idx_.load(std::memory_order_acquire) == write_idx_.load(std::memory_order_acquire);
}

size_t size() const {
    size_t write = write_idx_.load(std::memory_order_acquire);
    size_t read = read_idx_.load(std::memory_order_acquire);
    if (write > read) {
        return write - read;
    }
    return capacity - (read - write);
}

private:
    std::vector<T> buf;
    std::atomic<size_t> write_idx_;
    std::atomic<size_t> read_idx_;
    const size_t capacity;
};

}
}



#endif // INCLUDE_SPMC_H_