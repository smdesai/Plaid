import CoreML
import Foundation

/// A batch whose CPU-side inputs (tokenized ids → `MLMultiArray`s) have already
/// been built, ready to hand to the model for inference.
///
/// Splitting "build the inputs" (CPU) from "run the model" (GPU) lets the encode
/// loop prepare the *next* batch while the *current* one runs on the GPU — the
/// two phases are otherwise serial, and a device trace showed the GPU idle ~47%
/// of encode wall-clock waiting on this CPU prep. The stored payload is only
/// read during inference and each instance is handed off to exactly one consumer,
/// so it is safe to move across threads (`@unchecked Sendable`).
public struct PreparedColbertBatch: @unchecked Sendable {
    let batchProvider: MLArrayBatchProvider
    let attentionMasks: [[Int]]

    var count: Int { attentionMasks.count }
}

extension MLMultiArray {
    /// Read the leading `limit` token rows of a `[.., tokens, dim]` embedding
    /// tensor into `[[Float]]`.
    ///
    /// This replaces the previous per-element `array[i].floatValue` read, which
    /// boxed every value through `NSNumber` (~`tokens × dim` bridged calls). It
    /// reads the backing buffer directly using the **same** contiguous
    /// row-major indexing (`token * dim + d`) the boxed path used, so the
    /// produced floats are identical — only faster. Unexpected element types
    /// fall back to the boxed read so behavior is preserved.
    static func colbertRowMajorFloats(from array: MLMultiArray, limit: Int) -> [[Float]] {
        let shape = array.shape.map { $0.intValue }
        guard shape.count >= 2 else { return [] }
        let embeddingDim = shape[shape.count - 1]
        let tokenCount = shape[shape.count - 2]
        let totalTokens = min(tokenCount, limit)
        guard totalTokens > 0, embeddingDim > 0 else { return [] }

        var vectors = [[Float]](
            repeating: [Float](repeating: 0, count: embeddingDim),
            count: totalTokens
        )

        func fill<T>(_ pointer: UnsafePointer<T>, _ convert: (T) -> Float) {
            for token in 0 ..< totalTokens {
                let base = token * embeddingDim
                for dim in 0 ..< embeddingDim {
                    vectors[token][dim] = convert(pointer[base + dim])
                }
            }
        }

        switch array.dataType {
        case .float32:
            array.withUnsafeBytes { raw in
                if let base = raw.bindMemory(to: Float32.self).baseAddress {
                    fill(base) { Float($0) }
                }
            }
        case .float16:
            array.withUnsafeBytes { raw in
                if let base = raw.bindMemory(to: Float16.self).baseAddress {
                    fill(base) { Float($0) }
                }
            }
        case .double:
            array.withUnsafeBytes { raw in
                if let base = raw.bindMemory(to: Double.self).baseAddress {
                    fill(base) { Float($0) }
                }
            }
        default:
            // Integer element types (or any future type): fall back to the
            // boxed read so the produced floats match the previous behavior.
            for token in 0 ..< totalTokens {
                for dim in 0 ..< embeddingDim {
                    vectors[token][dim] = array[token * embeddingDim + dim].floatValue
                }
            }
        }

        return vectors
    }

    /// Build a `[1, values.count]` int32 `MLMultiArray` for a single model-input row.
    ///
    /// Fills the backing buffer directly instead of assigning an `NSNumber` per
    /// element (the previous per-index subscript path boxed every token — one
    /// bridged allocation per value, thousands per encode batch). Token ids and
    /// attention-mask values are small, in-range integers, so
    /// `Int32(truncatingIfNeeded:)` yields the same stored bits that
    /// `NSNumber(value:).int32Value` did — identical array contents, cheaper to
    /// build. Now that inference can run on the ANE, this input prep is a larger
    /// share of the per-batch CPU cost, so removing the boxing matters more.
    static func makeInt32Batch(values: [Int]) throws -> MLMultiArray {
        let array = try MLMultiArray(
            shape: [1, NSNumber(value: values.count)], dataType: .int32)
        array.withUnsafeMutableBytes { raw, _ in
            guard let base = raw.bindMemory(to: Int32.self).baseAddress else { return }
            for (index, value) in values.enumerated() {
                base[index] = Int32(truncatingIfNeeded: value)
            }
        }
        return array
    }
}
