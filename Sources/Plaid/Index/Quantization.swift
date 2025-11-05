import Foundation
import MLX

enum Quantization {
    static func compressIntoCodes(
        embeddings: MLXArray,
        centroids: MLXArray,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        precondition(embeddings.shape.count == 2, "Embeddings must be rank-2")
        precondition(centroids.shape.count == 2, "Centroids must be rank-2")
        precondition(embeddings.shape[1] == centroids.shape[1], "Dimension mismatch")
        let centroidCount = centroids.shape[0]
        precondition(centroidCount > 0, "Centroids must contain at least one row")

        let embeddingsT = embeddings.transposed(1, 0, stream: stream)
        let scores = centroids.matmul(embeddingsT, stream: stream)
        let rawCodes =
            scores
            .argMax(axis: 0, stream: stream)
            .asType(.int32, stream: stream)
        let zero = MLXArray([Int32(0)], [1]).asType(.int32, stream: stream)
        let maxIndex = MLXArray([Int32(centroidCount - 1)], [1]).asType(.int32, stream: stream)
        let clamped = minimum(maximum(rawCodes, zero, stream: stream), maxIndex, stream: stream)
        return clamped
    }

    static func packBits(_ bits: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
        let shape = bits.shape
        precondition(shape.count == 2, "Expecting rank-2 bit tensor")
        let rows = shape[0]
        let cols = shape[1]
        precondition(cols % 8 == 0, "Number of columns must be divisible by 8")

        let uint8Bits = bits.asType(.uint8, stream: stream)
        let bitValues = uint8Bits.asArray(UInt8.self)
        var packed: [UInt8] = []
        packed.reserveCapacity(rows * cols / 8)

        let bytesPerRow = cols / 8
        for row in 0 ..< rows {
            let rowOffset = row * cols
            for chunk in 0 ..< bytesPerRow {
                var byte: UInt8 = 0
                let bitOffset = rowOffset + chunk * 8
                for bitIndex in 0 ..< 8 {
                    let bit = bitValues[bitOffset + bitIndex] & 0x1
                    byte |= bit << (7 - bitIndex)
                }
                packed.append(byte)
            }
        }

        return MLXArray(packed, [rows, bytesPerRow])
    }
}
