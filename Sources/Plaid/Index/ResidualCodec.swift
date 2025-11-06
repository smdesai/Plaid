import Foundation
import MLX

struct ResidualCodec {
    let nbits: Int
    let embeddingDim: Int
    let numCentroids: Int

    let centroids: MLXArray
    let avgResidual: MLXArray
    let bucketCutoffs: MLXArray?
    let bucketWeights: MLXArray?

    let bitHelper: MLXArray
    let byteReversedBitsMap: MLXArray
    let bucketWeightIndicesLookup: MLXArray?

    init(
        nbits: Int,
        centroids: MLXArray,
        avgResidual: MLXArray,
        bucketCutoffs: MLXArray?,
        bucketWeights: MLXArray?,
        stream: StreamOrDevice = .default
    ) throws {
        precondition(nbits > 0 && nbits <= 8, "nbits must be in 1...8")
        precondition(centroids.shape.count == 2, "centroids must be rank-2")
        precondition(avgResidual.shape.count == 1, "avgResidual must be rank-1")
        precondition(centroids.shape[1] == avgResidual.shape[0], "dimension mismatch")

        if let cutoffs = bucketCutoffs {
            precondition(cutoffs.shape.count == 2, "bucketCutoffs must be rank-2")
            precondition(cutoffs.shape[1] == avgResidual.shape[0], "cutoff dims mismatch")
            precondition(cutoffs.shape[0] == (1 << nbits) - 1, "cutoff quantiles mismatch")
        }
        if let weights = bucketWeights {
            precondition(weights.shape.count == 2, "bucketWeights must be rank-2")
            precondition(weights.shape[1] == avgResidual.shape[0], "weight dims mismatch")
            precondition(weights.shape[0] == (1 << nbits), "weight quantiles mismatch")
        }

        self.nbits = nbits
        self.embeddingDim = centroids.shape[1]
        self.numCentroids = centroids.shape[0]
        self.centroids = centroids
        self.avgResidual = avgResidual
        self.bucketCutoffs = bucketCutoffs
        self.bucketWeights = bucketWeights

        self.bitHelper = ResidualCodec.makeBitHelper(nbits: nbits, stream: stream)
        self.byteReversedBitsMap = ResidualCodec.makeByteReversedBitsMap(
            nbits: nbits, stream: stream)
        if let weights = bucketWeights {
            self.bucketWeightIndicesLookup = try ResidualCodec.makeBucketWeightLookup(
                nbits: nbits,
                bucketWeights: weights,
                stream: stream
            )
        } else {
            self.bucketWeightIndicesLookup = nil
        }
    }

    var keysPerByte: Int { 8 / nbits }
    var numBuckets: Int { 1 << nbits }

    private static func makeBitHelper(nbits: Int, stream: StreamOrDevice) -> MLXArray {
        let values = (0 ..< nbits).map { Int32($0) }
        return MLXArray(values, [nbits]).asType(.int32, stream: stream)
    }

    private static func makeByteReversedBitsMap(nbits: Int, stream: StreamOrDevice) -> MLXArray {
        let mask = (1 << nbits) - 1
        var reversed: [Int32] = Array(repeating: 0, count: 256)
        for byte in 0 ..< 256 {
            var reversedByte: Int32 = 0
            var bitPos = 8
            while bitPos >= nbits {
                let segment = (byte >> (bitPos - nbits)) & mask
                var reversedSegment = 0
                for k in 0 ..< nbits {
                    if (segment & (1 << k)) != 0 {
                        reversedSegment |= 1 << (nbits - 1 - k)
                    }
                }
                reversedByte |= Int32(reversedSegment)
                if bitPos > nbits {
                    reversedByte <<= Int32(nbits)
                }
                bitPos -= nbits
            }
            reversed[byte] = reversedByte & Int32(0xFF)
        }
        return MLXArray(reversed, [256]).asType(.int32, stream: stream)
    }

    private static func makeBucketWeightLookup(
        nbits: Int,
        bucketWeights: MLXArray,
        stream: StreamOrDevice
    ) throws -> MLXArray {
        let keysPerByte = 8 / nbits
        precondition(keysPerByte > 0, "nbits must divide 8")
        let numBuckets = bucketWeights.shape[0]
        var combos = 1
        for _ in 0 ..< keysPerByte {
            combos *= numBuckets
        }
        var flattened: [Int32] = []
        flattened.reserveCapacity(combos * keysPerByte)

        func build(level: Int, current: [Int32]) {
            if level == keysPerByte {
                flattened.append(contentsOf: current)
                return
            }
            for bucket in 0 ..< numBuckets {
                var next = current
                next.append(Int32(bucket))
                build(level: level + 1, current: next)
            }
        }

        if combos > 0 {
            build(level: 0, current: [])
        }

        return MLXArray(flattened, [combos, keysPerByte]).asType(.int32, stream: stream)
    }
}

struct ResidualCodecTrainer {
    let nbits: Int
    let stream: StreamOrDevice

    init(nbits: Int, stream: StreamOrDevice = .default) {
        self.nbits = nbits
        self.stream = stream
    }

    func train(centroids: MLXArray, samples: MLXArray) throws -> ResidualCodec {
        precondition(samples.shape.count == 2, "Samples must be rank-2")
        let codes = Quantization.compressIntoCodes(
            embeddings: samples,
            centroids: centroids,
            stream: stream
        )
        let gathered = centroids.take(codes.asType(.int32, stream: stream), axis: 0, stream: stream)
        let residual = samples - gathered
        eval(residual)  // Checkpoint: Materialize residual before statistics
        let avgResidual = residual.abs(stream: stream).mean(axes: [0], stream: stream)

        let (cutoffs, weights) = computeQuantiles(residual: residual)

        return try ResidualCodec(
            nbits: nbits,
            centroids: centroids,
            avgResidual: avgResidual,
            bucketCutoffs: cutoffs,
            bucketWeights: weights,
            stream: stream
        )
    }

    private func computeQuantiles(residual: MLXArray) -> (MLXArray?, MLXArray?) {
        let numBuckets = 1 << nbits
        guard numBuckets > 0 else { return (nil, nil) }

        let cutQuantiles = (1 ..< numBuckets).map { Float($0) / Float(numBuckets) }
        let weightQuantiles = (0 ..< numBuckets).map { (Float($0) + 0.5) / Float(numBuckets) }

        // Flatten all residual values (across all dimensions) to match Rust implementation
        let residualValues = residual.asArray(Float32.self)
        var allValues = residualValues.map { Float($0) }
        allValues.sort()

        // Compute quantiles on the GLOBAL pool of residual values
        let cutoffValues = quantiles(sorted: allValues, probs: cutQuantiles)
        let weightValues = quantiles(sorted: allValues, probs: weightQuantiles)

        let dim = residual.shape[1]

        // Create broadcasted arrays (same value for all dimensions)
        var cutoffs: [Float] = []
        cutoffs.reserveCapacity((numBuckets - 1) * dim)
        for cutoff in cutoffValues {
            for _ in 0 ..< dim {
                cutoffs.append(cutoff)
            }
        }

        var weights: [Float] = []
        weights.reserveCapacity(numBuckets * dim)
        for weight in weightValues {
            for _ in 0 ..< dim {
                weights.append(weight)
            }
        }

        let cutoffArray =
            cutQuantiles.isEmpty
            ? nil
            : MLXArray(cutoffs, [numBuckets - 1, dim]).asType(.float32, stream: stream)
        let weightArray = MLXArray(weights, [numBuckets, dim]).asType(.float32, stream: stream)

        return (cutoffArray, weightArray)
    }

    private func quantiles(sorted: [Float], probs: [Float]) -> [Float] {
        guard !sorted.isEmpty else { return Array(repeating: 0, count: probs.count) }
        let n = sorted.count
        return probs.map { p in
            let clamped = min(max(p, 0), 1)
            if n == 1 { return sorted[0] }
            let position = clamped * Float(n - 1)
            let lowerIndex = Int(floor(position))
            let upperIndex = Int(ceil(position))
            if lowerIndex == upperIndex {
                return sorted[lowerIndex]
            }
            let lowerValue = sorted[lowerIndex]
            let upperValue = sorted[upperIndex]
            let weight = position - Float(lowerIndex)
            return lowerValue * (1 - weight) + upperValue * weight
        }
    }
}
