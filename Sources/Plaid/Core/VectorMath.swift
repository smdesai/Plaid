import Foundation

enum VectorMath {
    static func normalize(_ vector: [Float], epsilon: Float = 1e-12) -> [Float] {
        var norm: Float = 0
        for value in vector {
            norm += value * value
        }
        norm = sqrt(norm)
        guard norm > epsilon else {
            return vector
        }
        return vector.map { $0 / norm }
    }
}
