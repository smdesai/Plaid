import Foundation

enum VectorMath {
    static func averageAndNormalize(_ matrix: [[Float]], embeddingDim: Int) throws -> [Float] {
        guard !matrix.isEmpty else {
            throw PlaidError.emptyEmbeddingSet
        }

        var accumulator = [Float](repeating: 0, count: embeddingDim)
        for row in matrix {
            guard row.count == embeddingDim else {
                throw PlaidError.invalidEmbeddingDimensions(
                    expected: embeddingDim, actual: row.count)
            }
            for (idx, value) in row.enumerated() {
                accumulator[idx] += value
            }
        }

        let count = Float(matrix.count)
        var averaged = accumulator.map { $0 / count }
        averaged = normalize(averaged)
        return averaged
    }

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

    static func dot(_ lhs: [Float], _ rhs: [Float]) -> Float {
        zip(lhs, rhs).reduce(0) { $0 + $1.0 * $1.1 }
    }
}
