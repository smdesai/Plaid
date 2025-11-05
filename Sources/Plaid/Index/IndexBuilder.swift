import Foundation
import MLX

struct IndexBuildResult {
    let docVectors: [[Float]]
    let docLengths: [Int]
}

struct IndexBuilder {
    let outputURL: URL
    let embeddingDim: Int
    let nbits: Int
    let batchSize: Int
    let seed: UInt64?
    let documents: [[[Float]]]
    let centroids: [[Float]]
    let stream: StreamOrDevice = .default

    func build() throws -> IndexBuildResult {
        let (preparedDocs, docVectors, docLengths) = try prepareDocuments()
        let totalEmbeddings = docLengths.reduce(0, +)
        guard totalEmbeddings > 0 else {
            throw PlaidError.emptyEmbeddingSet
        }

        let centroidTensor = try makeCentroidTensor()
        let sampleIndices = makeSampleIndices(count: preparedDocs.count)
        let heldoutTensor = try makeHeldoutTensor(
            documents: preparedDocs, sampleIndices: sampleIndices)

        let trainer = ResidualCodecTrainer(nbits: nbits, stream: stream)
        let codec = try trainer.train(centroids: centroidTensor, samples: heldoutTensor)

        try resetOutputDirectory()
        let paths = IndexPaths(root: outputURL)

        let chunkSize = computeChunkSize(documentCount: preparedDocs.count)
        let numChunks = Int(ceil(Double(preparedDocs.count) / Double(chunkSize)))
        try writePlan(paths: paths, numChunks: numChunks, centroidCount: codec.numCentroids)
        try writeCodec(codec: codec, paths: paths)

        let cutoffsPerDim = try makeCutoffsPerDimension(from: codec)
        let bytesPerResidual = try residualBytesPerVector()

        var allCodes: [Int32] = []
        allCodes.reserveCapacity(totalEmbeddings)
        var doclensAccumulator: [Int] = []
        doclensAccumulator.reserveCapacity(docLengths.count)

        var embeddingOffset = 0
        var chunkIndex = 0
        var chunkStart = 0
        while chunkStart < preparedDocs.count {
            let end = min(chunkStart + chunkSize, preparedDocs.count)
            let chunkDocs = Array(preparedDocs[chunkStart ..< end])
            let chunkDocLengths = chunkDocs.map { $0.length }
            doclensAccumulator.append(contentsOf: chunkDocLengths)

            let (codes, packedResiduals, chunkEmbeddings) = try encodeChunk(
                documents: chunkDocs,
                codec: codec,
                cutoffsPerDim: cutoffsPerDim,
                bytesPerResidual: bytesPerResidual
            )

            allCodes.append(contentsOf: codes)

            try BinaryIO.writeInt32(codes, to: paths.codes(for: chunkIndex))
            try BinaryIO.writeUInt8(packedResiduals, to: paths.residuals(for: chunkIndex))
            try writeDocLengths(chunkDocLengths, to: paths.docLens(for: chunkIndex))

            let metadata = ChunkMetadata(
                num_passages: chunkDocLengths.count,
                num_embeddings: chunkEmbeddings,
                embedding_offset: embeddingOffset
            )
            try writeJSON(metadata, to: paths.chunkMetadata(for: chunkIndex))

            embeddingOffset += chunkEmbeddings
            chunkStart = end
            chunkIndex += 1
        }

        let estPartitions = estimatePartitions(
            totalEmbeddings: totalEmbeddings, documentCount: preparedDocs.count)
        let (ivf, ivfLengths) = try buildIVF(codes: allCodes, docLengths: docLengths, codec: codec)
        try BinaryIO.writeInt32(ivf, to: paths.ivf())
        try BinaryIO.writeInt32(ivfLengths, to: paths.ivfLengths())

        let avgDocLen =
            preparedDocs.isEmpty ? 0.0 : Double(totalEmbeddings) / Double(preparedDocs.count)
        let finalMetadata = FinalMetadata(
            num_chunks: numChunks,
            nbits: nbits,
            num_partitions: max(estPartitions, 1),
            num_embeddings: totalEmbeddings,
            avg_doclen: avgDocLen,
            embedding_dim: embeddingDim,
            total_documents: docLengths.count,
            bytes_per_residual: bytesPerResidual
        )
        try writeJSON(finalMetadata, to: paths.metadata())

        return IndexBuildResult(docVectors: docVectors, docLengths: docLengths)
    }
}

extension IndexBuilder {
    struct DocumentTensor {
        let flat: [Float32]
        let length: Int
    }

    struct IndexPaths {
        let root: URL

        func plan() -> URL { root.appendingPathComponent("plan.json") }
        func centroids() -> URL { root.appendingPathComponent("centroids.bin") }
        func bucketCutoffs() -> URL { root.appendingPathComponent("bucket_cutoffs.bin") }
        func bucketWeights() -> URL { root.appendingPathComponent("bucket_weights.bin") }
        func avgResidual() -> URL { root.appendingPathComponent("avg_residual.bin") }
        func bitHelper() -> URL { root.appendingPathComponent("bit_helper.bin") }
        func byteReversedBitsMap() -> URL {
            root.appendingPathComponent("byte_reversed_bits_map.bin")
        }
        func bucketWeightLookup() -> URL { root.appendingPathComponent("bucket_weight_lookup.bin") }
        func codes(for chunk: Int) -> URL {
            root.appendingPathComponent("chunk_\(chunk).codes.bin")
        }
        func residuals(for chunk: Int) -> URL {
            root.appendingPathComponent("chunk_\(chunk).residuals.bin")
        }
        func docLens(for chunk: Int) -> URL { root.appendingPathComponent("doclens.\(chunk).json") }
        func chunkMetadata(for chunk: Int) -> URL {
            root.appendingPathComponent("chunk_\(chunk).metadata.json")
        }
        func ivf() -> URL { root.appendingPathComponent("ivf.bin") }
        func ivfLengths() -> URL { root.appendingPathComponent("ivf_lengths.bin") }
        func metadata() -> URL { root.appendingPathComponent("metadata.json") }
    }

    struct ChunkMetadata: Codable {
        let num_passages: Int
        let num_embeddings: Int
        let embedding_offset: Int
    }

    struct FinalMetadata: Codable {
        let num_chunks: Int
        let nbits: Int
        let num_partitions: Int
        let num_embeddings: Int
        let avg_doclen: Double
        let embedding_dim: Int
        let total_documents: Int
        let bytes_per_residual: Int
    }

    func prepareDocuments() throws -> ([DocumentTensor], [[Float]], [Int]) {
        var tensors: [DocumentTensor] = []
        tensors.reserveCapacity(documents.count)
        var docVectors: [[Float]] = []
        docVectors.reserveCapacity(documents.count)
        var docLengths: [Int] = []
        docLengths.reserveCapacity(documents.count)

        for document in documents {
            guard !document.isEmpty else {
                throw PlaidError.emptyEmbeddingSet
            }
            var flat: [Float32] = []
            flat.reserveCapacity(document.count * embeddingDim)
            for row in document {
                guard row.count == embeddingDim else {
                    throw PlaidError.invalidEmbeddingDimensions(
                        expected: embeddingDim, actual: row.count)
                }
                for value in row {
                    flat.append(Float32(value))
                }
            }
            tensors.append(DocumentTensor(flat: flat, length: document.count))
            docLengths.append(document.count)
            docVectors.append(
                try VectorMath.averageAndNormalize(document, embeddingDim: embeddingDim))
        }

        return (tensors, docVectors, docLengths)
    }

    func makeCentroidTensor() throws -> MLXArray {
        guard !centroids.isEmpty else {
            throw PlaidError.emptyEmbeddingSet
        }
        let rows = centroids.count
        var flat: [Float32] = []
        flat.reserveCapacity(rows * embeddingDim)
        for row in centroids {
            guard row.count == embeddingDim else {
                throw PlaidError.invalidEmbeddingDimensions(
                    expected: embeddingDim, actual: row.count)
            }
            flat.append(contentsOf: row.map { Float32($0) })
        }
        return MLXArray(flat, [rows, embeddingDim])
    }

    func makeSampleIndices(count: Int) -> [Int] {
        guard count > 0 else { return [] }
        let sampleK = 16.0 * sqrt(120.0 * Double(count))
        let sampleCount = max(1, min(count, Int(sampleK.rounded(.down) + 1.0)))
        var indices = Array(0 ..< count)
        if let seed {
            var rng = SeededGenerator(seed: seed)
            indices.shuffle(using: &rng)
        } else {
            indices.shuffle()
        }
        return Array(indices.prefix(sampleCount))
    }

    func makeHeldoutTensor(documents: [DocumentTensor], sampleIndices: [Int]) throws -> MLXArray {
        guard !sampleIndices.isEmpty else {
            return try fallbackHeldoutTensor(documents: documents)
        }
        let totalSamples = sampleIndices.reduce(0) { $0 + documents[$1].length }
        let target = min(Int((0.05 * Double(totalSamples)).rounded()), 50_000)
        var heldoutFlat: [Float32] = []
        var heldoutCount = 0
        if target > 0 {
            for index in sampleIndices.reversed() {
                let doc = documents[index]
                let needed = target - heldoutCount
                if needed <= 0 { break }
                if doc.length == 0 { continue }
                let tokensToTake = min(needed, doc.length)
                let startToken = doc.length - tokensToTake
                let start = startToken * embeddingDim
                let end = start + tokensToTake * embeddingDim
                heldoutFlat.append(contentsOf: doc.flat[start ..< end])
                heldoutCount += tokensToTake
            }
        }
        if heldoutCount == 0 {
            return try fallbackHeldoutTensor(documents: documents)
        }
        return MLXArray(heldoutFlat, [heldoutCount, embeddingDim])
    }

    func fallbackHeldoutTensor(documents: [DocumentTensor]) throws -> MLXArray {
        var flat: [Float32] = []
        var count = 0
        for doc in documents {
            flat.append(contentsOf: doc.flat)
            count += doc.length
        }
        guard count > 0 else {
            throw PlaidError.emptyEmbeddingSet
        }
        return MLXArray(flat, [count, embeddingDim])
    }

    func resetOutputDirectory() throws {
        if FileManager.default.fileExists(atPath: outputURL.path) {
            try FileManager.default.removeItem(at: outputURL)
        }
        try FileManager.default.createDirectory(at: outputURL, withIntermediateDirectories: true)
    }

    func computeChunkSize(documentCount: Int) -> Int {
        guard documentCount > 0 else { return 1 }
        let minChunk = max(1, batchSize)
        return min(minChunk, documentCount + 1)
    }

    func writePlan(paths: IndexPaths, numChunks: Int, centroidCount: Int) throws {
        struct Plan: Codable {
            let nbits: Int
            let num_chunks: Int
            let centroid_count: Int
        }
        let plan = Plan(nbits: nbits, num_chunks: numChunks, centroid_count: centroidCount)
        try writeJSON(plan, to: paths.plan())
    }

    func writeCodec(codec: ResidualCodec, paths: IndexPaths) throws {
        let centroids = codec.centroids.asArray(Float32.self).map { Float($0) }
        try BinaryIO.writeFloat32(centroids, to: paths.centroids())

        let avgResidual = codec.avgResidual.asArray(Float32.self).map { Float($0) }
        try BinaryIO.writeFloat32(avgResidual, to: paths.avgResidual())

        if let cutoffs = codec.bucketCutoffs?.asArray(Float32.self) {
            try BinaryIO.writeFloat32(cutoffs.map { Float($0) }, to: paths.bucketCutoffs())
        }
        if let weights = codec.bucketWeights?.asArray(Float32.self) {
            try BinaryIO.writeFloat32(weights.map { Float($0) }, to: paths.bucketWeights())
        }

        try CodecSerialization.save(codec: codec, paths: paths)
    }

    func makeCutoffsPerDimension(from codec: ResidualCodec) throws -> [[Float]] {
        guard let cutoffsTensor = codec.bucketCutoffs else {
            throw PlaidError.emptyEmbeddingSet
        }
        let raw = cutoffsTensor.asArray(Float32.self)
        let rows = codec.numBuckets - 1
        var perDim = Array(repeating: [Float](), count: embeddingDim)
        for dim in 0 ..< embeddingDim {
            perDim[dim].reserveCapacity(rows)
        }
        for row in 0 ..< rows {
            for dim in 0 ..< embeddingDim {
                let index = row * embeddingDim + dim
                perDim[dim].append(Float(raw[index]))
            }
        }
        return perDim
    }

    func residualBytesPerVector() throws -> Int {
        let bitsPerVector = embeddingDim * nbits
        guard bitsPerVector % 8 == 0 else {
            throw PlaidError.invalidEmbeddingDimensions(expected: 8, actual: bitsPerVector % 8)
        }
        return bitsPerVector / 8
    }

    func encodeChunk(
        documents: [DocumentTensor],
        codec: ResidualCodec,
        cutoffsPerDim: [[Float]],
        bytesPerResidual: Int
    ) throws -> ([Int32], [UInt8], Int) {
        let totalEmbeddings = documents.reduce(0) { $0 + $1.length }
        guard totalEmbeddings > 0 else { return ([], [], 0) }

        var buffer: [Float32] = []
        buffer.reserveCapacity(totalEmbeddings * embeddingDim)
        for doc in documents {
            buffer.append(contentsOf: doc.flat)
        }

        let embeddingsTensor = MLXArray(buffer, [totalEmbeddings, embeddingDim])
        let codesTensor64 = Quantization.compressIntoCodes(
            embeddings: embeddingsTensor,
            centroids: codec.centroids,
            stream: stream
        )
        let codesTensor = codesTensor64.asType(.int32, stream: stream)
        let codes = codesTensor.asArray(Int32.self)

        let gathered = codec.centroids.take(codesTensor, axis: 0, stream: stream)
        let residualTensor = embeddingsTensor - gathered
        let residualValues = residualTensor.asArray(Float32.self)

        let packedResiduals = packResiduals(
            residuals: residualValues,
            vectorCount: totalEmbeddings,
            cutoffsPerDim: cutoffsPerDim,
            bytesPerVector: bytesPerResidual
        )

        return (codes, packedResiduals, totalEmbeddings)
    }

    func packResiduals(
        residuals: [Float32],
        vectorCount: Int,
        cutoffsPerDim: [[Float]],
        bytesPerVector: Int
    ) -> [UInt8] {
        let bucketBits = nbits
        var packed = [UInt8](repeating: 0, count: vectorCount * bytesPerVector)
        for vectorIndex in 0 ..< vectorCount {
            for dim in 0 ..< embeddingDim {
                let value = Float(residuals[vectorIndex * embeddingDim + dim])
                let bucket = bucketIndex(for: value, cutoffs: cutoffsPerDim[dim])
                for bitOffset in 0 ..< bucketBits {
                    let bit = (bucket >> (bucketBits - 1 - bitOffset)) & 1
                    let bitIndex = dim * bucketBits + bitOffset
                    let byteIndex = bitIndex / 8
                    let shift = 7 - (bitIndex % 8)
                    let idx = vectorIndex * bytesPerVector + byteIndex
                    packed[idx] |= UInt8(bit) << shift
                }
            }
        }
        return packed
    }

    func bucketIndex(for value: Float, cutoffs: [Float]) -> Int {
        var low = 0
        var high = cutoffs.count
        while low < high {
            let mid = (low + high) / 2
            if value <= cutoffs[mid] {
                high = mid
            } else {
                low = mid + 1
            }
        }
        return low
    }

    func writeDocLengths(_ docLengths: [Int], to url: URL) throws {
        try writeJSON(docLengths, to: url)
    }

    func writeJSON<T: Encodable>(_ value: T, to url: URL) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(value)
        try data.write(to: url, options: .atomic)
    }

    func estimatePartitions(totalEmbeddings: Int, documentCount: Int) -> Int {
        guard documentCount > 0, totalEmbeddings > 0 else { return 1 }
        let target = Double(totalEmbeddings)
        guard target > 0 else { return 1 }
        let scaled = 16.0 * sqrt(target)
        guard scaled > 0 else { return 1 }
        let exponent = floor(log2(scaled))
        let value = pow(2.0, max(0.0, exponent))
        return max(Int(value), 1)
    }

    func buildIVF(
        codes: [Int32],
        docLengths: [Int],
        codec: ResidualCodec
    ) throws -> ([Int32], [Int32]) {
        let totalEmbeddings = codes.count
        var codeCounts = [Int](repeating: 0, count: codec.numCentroids)
        for code in codes {
            let idx = Int(code)
            guard idx >= 0 && idx < codec.numCentroids else {
                throw PlaidError.invalidEmbeddingDimensions(
                    expected: codec.numCentroids, actual: idx)
            }
            codeCounts[idx] += 1
        }

        let order = codes.enumerated().sorted { $0.element < $1.element }
        let sortedIndices = order.map { Int($0.offset) }

        var embToPid: [Int32] = []
        embToPid.reserveCapacity(totalEmbeddings)
        for (pid, length) in docLengths.enumerated() {
            embToPid.append(contentsOf: Array(repeating: Int32(pid), count: length))
        }

        var ivf: [Int32] = []
        var ivfLengths: [Int32] = []
        ivf.reserveCapacity(totalEmbeddings)
        var offset = 0
        for centroid in 0 ..< codec.numCentroids {
            let count = codeCounts[centroid]
            if count == 0 {
                ivfLengths.append(0)
                continue
            }
            let end = offset + count
            let slice = sortedIndices[offset ..< end]
            offset = end
            var seen = Set<Int32>()
            seen.reserveCapacity(count)
            var unique: [Int32] = []
            unique.reserveCapacity(count)
            for index in slice {
                let pid = embToPid[index]
                if seen.insert(pid).inserted {
                    unique.append(pid)
                }
            }
            ivf.append(contentsOf: unique)
            ivfLengths.append(Int32(unique.count))
        }

        return (ivf, ivfLengths)
    }
}
