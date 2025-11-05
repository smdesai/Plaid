import Foundation

public struct SeededGenerator: RandomNumberGenerator {
    private var state: UInt64

    public init(seed: UInt64) {
        var value = seed
        if value == 0 {
            value = 0x853C_49E6_748F_EA9B
        }
        state = value
    }

    public init() {
        let seed =
            UInt64(Date().timeIntervalSince1970) ^ UInt64.random(in: UInt64.min ... UInt64.max)
        self.init(seed: seed)
    }

    public mutating func next() -> UInt64 {
        state = state &* 636_413_622_384_679_3005 &+ 1
        return state
    }
}
