# mlir-netopt

NetOpt is a domain-specific language, targeting a MLIR-based intermediate representation, designed for concurrent network packet processing.

## Concurrency

The fundamental building block of NetOpt is the single-producer, multiple-consumer (spmc) queue. The spmc mimics a network interface card (NIC) spraying packets in a round-robin fashion across multiple cores/threads of a CPU to be processed. Each core/thread of the CPU is treated as a consumer in this model. The API exposes 3 operations to the user:

```mlir
spmc.create() {element=T, capacity=N} : () -> !spmc.queue<T,N>
spmc.push_back(%q, %val) : (!spmc.queue<T,N>, T) -> ()
spmc.pop_front(%q) : (!spmc.queue<T,N>) -> T
```
