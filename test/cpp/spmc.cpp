#include "spmc.h"

#include <iostream>
#include <thread>

using namespace cpp::parallel;

namespace {
    static Spmc<int> queue(1024);
    static std::atomic<bool> running{true};

    void producerJob() {
        for (int i = 0; running; ++i) {
            while (!queue.push_back(i) && running) {
                std::this_thread::yield();  // back off if queue is full
            }
        }
    }

    void consumerJob() {
        while (running) {
            const std::optional<int> val = queue.pop_front();
            if (val.has_value()) {
                std::cout << std::to_string(val.value()) << '\n';
            } else {
                std::this_thread::yield();  // back off if queue is empty
            }
        }
    }
}

int main() {
    std::thread producer(producerJob);

    const int num_consumers = 4;
    std::vector<std::thread> consumers;
    consumers.reserve(num_consumers);

    for (int i = 0; i < num_consumers; ++i) {
        consumers.emplace_back(consumerJob);
    }

    std::this_thread::sleep_for(std::chrono::seconds(5));

    // shutdown
    running = false;
    // join threads
    producer.join();
    for (auto& consumer : consumers) {
        consumer.join();
    }

    return 0;
}
