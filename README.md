# mlir-netopt

NetOpt is a domain-specific language, targeting a MLIR-based intermediate representation, designed for concurrent network packet processing.

## Concurrency

The fundamental building block of NetOpt is the single-producer, multiple-consumer (spmc) queue. The spmc mimics a network interface card (NIC) spraying packets in a round-robin fashion across multiple cores/threads of a CPU to be processed. Each core/thread of the CPU is treated as a consumer in this model. The API exposes 3 operations to the user:

```mlir
spmc.queue.create<T>(void) -> spmc.queue<T>
spmc.queue.push_back(T val) -> void
spmc.queue.pop_front(spmc.queue<T> q) -> T val
```
