import Foundation

/// A bounded, order-preserving hand-off between one producer and one consumer.
///
/// The indexing pipeline uses this to overlap encoding (producer) with
/// index-building + metadata writes (consumer): the encoder fills a batch and
/// `enqueue`s it; the indexer `dequeue`s FIFO. `capacity` caps how many batches
/// can be in flight, so `enqueue` suspends when the buffer is full — that
/// backpressure keeps peak memory to ~`capacity` batches regardless of corpus
/// size.
///
/// Cancellation is handled cooperatively without `Task` cancellation: on error
/// either side calls `fail`, which wakes the other so neither suspends forever.
actor BoundedBatchQueue<Element: Sendable> {
    private let capacity: Int
    private var buffer: [Element] = []
    private var finished = false
    private(set) var failure: Error?

    private var dequeueWaiters: [CheckedContinuation<Element?, Never>] = []
    private var enqueueWaiters:
        [(element: Element, continuation: CheckedContinuation<Bool, Never>)] =
            []

    init(capacity: Int) {
        self.capacity = max(1, capacity)
    }

    /// Append an element, suspending while the buffer is full.
    /// Returns `false` if the queue was already finished/failed (element dropped).
    func enqueue(_ element: Element) async -> Bool {
        if finished { return false }

        // A consumer is parked waiting — hand the element straight over.
        if !dequeueWaiters.isEmpty {
            let waiter = dequeueWaiters.removeFirst()
            waiter.resume(returning: element)
            return true
        }

        if buffer.count < capacity {
            buffer.append(element)
            return true
        }

        // Full: park until a dequeue frees a slot (or the queue fails).
        return await withCheckedContinuation { (continuation: CheckedContinuation<Bool, Never>) in
            enqueueWaiters.append((element, continuation))
        }
    }

    /// Remove the oldest element, suspending while the buffer is empty.
    /// Returns `nil` once the queue is finished/failed and drained.
    func dequeue() async -> Element? {
        if !buffer.isEmpty {
            let element = buffer.removeFirst()
            admitWaitingProducer()
            return element
        }

        // Buffer empty but a producer is parked (buffer was full at capacity 0
        // edge / direct hand-off): take its element and let it proceed.
        if !enqueueWaiters.isEmpty {
            let waiter = enqueueWaiters.removeFirst()
            waiter.continuation.resume(returning: true)
            return waiter.element
        }

        if finished { return nil }

        return await withCheckedContinuation {
            (continuation: CheckedContinuation<Element?, Never>) in
            dequeueWaiters.append(continuation)
        }
    }

    /// Signal that no more elements will be enqueued. Buffered elements remain
    /// available to `dequeue`; once drained, `dequeue` returns `nil`.
    func finish() {
        guard !finished else { return }
        finished = true
        let waiters = dequeueWaiters
        dequeueWaiters.removeAll()
        for waiter in waiters { waiter.resume(returning: nil) }
    }

    /// Abort the queue with an error, discarding buffered elements and waking
    /// both sides so neither suspends forever.
    func fail(_ error: Error) {
        if failure == nil { failure = error }
        finished = true
        buffer.removeAll()

        let dequeues = dequeueWaiters
        dequeueWaiters.removeAll()
        for waiter in dequeues { waiter.resume(returning: nil) }

        let enqueues = enqueueWaiters
        enqueueWaiters.removeAll()
        for waiter in enqueues { waiter.continuation.resume(returning: false) }
    }

    /// Admit a parked producer into a freed buffer slot, preserving FIFO order.
    private func admitWaitingProducer() {
        guard buffer.count < capacity, !enqueueWaiters.isEmpty else { return }
        let waiter = enqueueWaiters.removeFirst()
        buffer.append(waiter.element)
        waiter.continuation.resume(returning: true)
    }
}
