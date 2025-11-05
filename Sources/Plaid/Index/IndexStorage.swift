import Foundation
import MLX

struct IndexArtifacts {
    let embeddingDim: Int
    let nbits: Int
    let numChunks: Int
    let numPartitions: Int
    let avgDocLen: Double
    let docLengths: [Int]
    let docOffsets: [Int]
    let embeddingOffsets: [Int]
    let codes: [Int32]
    let residuals: Data
    let bytesPerResidual: Int
    let centroids: MLXArray
    let bucketCutoffs: MLXArray?
    let bucketWeights: MLXArray?
    let avgResidual: MLXArray
    let ivfLists: [Int32]
    let ivfLengths: [Int32]
    let codecArtifacts: CodecArtifacts
}

enum IndexStorage {
    static func loadArtifacts(from url: URL, stream: StreamOrDevice = .default) throws
        -> IndexArtifacts
    {
        let paths = IndexPaths(root: url)

        var finalMetadata: FinalMetadata = try Self.readJSON(from: paths.metadata())
        let plan: PlanMetadata = try Self.readJSON(from: paths.plan())

        if finalMetadata.nbits == 0 {
            finalMetadata.nbits = plan.nbits
        }
        if finalMetadata.num_chunks == 0 {
            finalMetadata.num_chunks = plan.numChunks
        }

        let centroidData = try Data(contentsOf: paths.centroids())
        let centroidFloats = centroidData.count / MemoryLayout<Float32>.size
        let avgResidualData = try Data(contentsOf: paths.avgResidual())
        let avgResidualFloats = avgResidualData.count / MemoryLayout<Float32>.size
        var embeddingDim = finalMetadata.embedding_dim
        if embeddingDim == 0 {
            embeddingDim = avgResidualFloats
        }
        guard embeddingDim > 0 else {
            throw PlaidError.invalidEmbeddingDimensions(expected: 1, actual: embeddingDim)
        }
        guard avgResidualFloats == embeddingDim else {
            throw PlaidError.invalidEmbeddingDimensions(
                expected: embeddingDim, actual: avgResidualFloats)
        }

        let centroidCount: Int = {
            if let recorded = plan.centroidCount, recorded > 0 {
                return recorded
            }
            guard embeddingDim > 0, centroidFloats % embeddingDim == 0 else {
                return 0
            }
            return centroidFloats / embeddingDim
        }()
        guard centroidCount > 0 else {
            throw PlaidError.invalidEmbeddingDimensions(expected: 1, actual: centroidCount)
        }
        guard centroidCount * embeddingDim == centroidFloats else {
            throw PlaidError.invalidEmbeddingDimensions(
                expected: centroidCount * embeddingDim,
                actual: centroidFloats
            )
        }

        let centroids: [Float32] = centroidData.withUnsafeBytes { raw in
            let ptr = raw.baseAddress!.assumingMemoryBound(to: Float32.self)
            return Array(UnsafeBufferPointer(start: ptr, count: centroidFloats))
        }
        let avgResidual: [Float32] = avgResidualData.withUnsafeBytes { raw in
            let ptr = raw.baseAddress!.assumingMemoryBound(to: Float32.self)
            return Array(UnsafeBufferPointer(start: ptr, count: avgResidualFloats))
        }

        let numBuckets = max(1, 1 << finalMetadata.nbits)

        let cutoffScalars: [Float32]? = {
            if FileManager.default.fileExists(atPath: paths.bucketCutoffs().path) {
                return try? Self.loadFloatArray(from: paths.bucketCutoffs())
            }
            let npyURL = paths.root.appendingPathComponent("bucket_cutoffs.npy")
            if FileManager.default.fileExists(atPath: npyURL.path) {
                return try? Self.readNpyFloat32(from: npyURL)
            }
            return nil
        }()

        let weightScalars: [Float32]? = {
            if FileManager.default.fileExists(atPath: paths.bucketWeights().path) {
                return try? Self.loadFloatArray(from: paths.bucketWeights())
            }
            let npyURL = paths.root.appendingPathComponent("bucket_weights.npy")
            if FileManager.default.fileExists(atPath: npyURL.path) {
                return try? Self.readNpyFloat32(from: npyURL)
            }
            return nil
        }()

        var docLengths: [Int] = []
        var docOffsets: [Int] = []
        var allCodes: [Int32] = []
        var residualData = Data()
        docLengths.reserveCapacity(finalMetadata.total_documents)
        docOffsets.reserveCapacity(plan.numChunks)

        for chunk in 0 ..< plan.numChunks {
            let metadataURL = try paths.chunkMetadata(for: chunk)
            let metadata: ChunkMetadata = try Self.readJSON(from: metadataURL)
            let docLensURL = try paths.docLens(for: chunk)
            let lengths: [Int] = try Self.readJSON(from: docLensURL)
            docLengths.append(contentsOf: lengths)
            docOffsets.append(metadata.embedding_offset)

            let codes: [Int32]
            do {
                let codesURL = try paths.codes(for: chunk)
                codes = try Self.readInt32Array(from: codesURL)
            } catch PlaidError.indexNotFound {
                let npyURL = paths.root.appendingPathComponent("\(chunk).codes.npy")
                codes = try Self.readNpyInt32(from: npyURL)
            }
            allCodes.append(contentsOf: codes)

            let residualDataChunk: Data
            do {
                let residualURL = try paths.residuals(for: chunk)
                residualDataChunk = try Data(contentsOf: residualURL)
            } catch PlaidError.indexNotFound {
                let npyURL = paths.root.appendingPathComponent("\(chunk).residuals.npy")
                residualDataChunk = try Self.readNpyUInt8Data(from: npyURL)
            }
            residualData.append(residualDataChunk)
        }

        let ivf = try Self.readInt32Array(from: paths.ivf())
        let ivfLengths = try Self.readInt32Array(from: paths.ivfLengths())

        var embeddingOffsets: [Int] = Array(repeating: 0, count: docLengths.count + 1)
        for i in 0 ..< docLengths.count {
            embeddingOffsets[i + 1] = embeddingOffsets[i] + docLengths[i]
        }

        let totalDocuments = docLengths.count
        let totalEmbeddings = embeddingOffsets.last ?? 0

        var bytesPerResidual = finalMetadata.bytes_per_residual
        if bytesPerResidual == 0 && totalEmbeddings > 0 {
            bytesPerResidual = residualData.count / max(1, totalEmbeddings)
        }
        if bytesPerResidual == 0 {
            bytesPerResidual = max(1, embeddingDim * finalMetadata.nbits / 8)
        }

        finalMetadata.embedding_dim = embeddingDim
        finalMetadata.total_documents = totalDocuments
        finalMetadata.bytes_per_residual = bytesPerResidual
        if finalMetadata.avg_doclen == 0, totalDocuments > 0 {
            finalMetadata.avg_doclen = Double(totalEmbeddings) / Double(totalDocuments)
        }

        let codecArtifacts = try CodecSerialization.load(from: paths, stream: stream)

        return IndexArtifacts(
            embeddingDim: embeddingDim,
            nbits: finalMetadata.nbits,
            numChunks: plan.numChunks,
            numPartitions: finalMetadata.num_partitions,
            avgDocLen: finalMetadata.avg_doclen,
            docLengths: docLengths,
            docOffsets: docOffsets,
            embeddingOffsets: embeddingOffsets,
            codes: allCodes,
            residuals: residualData,
            bytesPerResidual: bytesPerResidual,
            centroids: MLXArray(centroids, [centroidCount, embeddingDim]).asType(
                .float32, stream: stream),
            bucketCutoffs: {
                guard let values = cutoffScalars else { return nil }
                let rowCount = max(0, numBuckets - 1)
                if values.count == rowCount * embeddingDim {
                    return MLXArray(values, [rowCount, embeddingDim]).asType(
                        .float32, stream: stream)
                } else if values.count == rowCount {
                    var expanded: [Float32] = []
                    expanded.reserveCapacity(rowCount * embeddingDim)
                    for value in values {
                        expanded.append(contentsOf: Array(repeating: value, count: embeddingDim))
                    }
                    return MLXArray(expanded, [rowCount, embeddingDim]).asType(
                        .float32, stream: stream)
                } else {
                    return nil
                }
            }(),
            bucketWeights: {
                guard let values = weightScalars else { return nil }
                if values.count == numBuckets * embeddingDim {
                    return MLXArray(values, [numBuckets, embeddingDim]).asType(
                        .float32, stream: stream)
                } else if values.count == numBuckets {
                    var expanded: [Float32] = []
                    expanded.reserveCapacity(numBuckets * embeddingDim)
                    for value in values {
                        expanded.append(contentsOf: Array(repeating: value, count: embeddingDim))
                    }
                    return MLXArray(expanded, [numBuckets, embeddingDim]).asType(
                        .float32, stream: stream)
                } else {
                    return nil
                }
            }(),
            avgResidual: MLXArray(avgResidual, [embeddingDim]).asType(.float32, stream: stream),
            ivfLists: ivf,
            ivfLengths: ivfLengths,
            codecArtifacts: codecArtifacts
        )
    }
}

extension IndexStorage {
    struct IndexPaths {
        let root: URL
        func plan() -> URL { root.appendingPathComponent("plan.json") }
        func metadata() -> URL { root.appendingPathComponent("metadata.json") }
        func centroids() -> URL { root.appendingPathComponent("centroids.bin") }
        func bucketCutoffs() -> URL { root.appendingPathComponent("bucket_cutoffs.bin") }
        func bucketWeights() -> URL { root.appendingPathComponent("bucket_weights.bin") }
        func avgResidual() -> URL { root.appendingPathComponent("avg_residual.bin") }
        func bitHelper() -> URL { root.appendingPathComponent("bit_helper.bin") }
        func byteReversedBitsMap() -> URL {
            root.appendingPathComponent("byte_reversed_bits_map.bin")
        }
        func bucketWeightLookup() -> URL { root.appendingPathComponent("bucket_weight_lookup.bin") }
        func codes(for chunk: Int) throws -> URL {
            try existingURL(
                primary: root.appendingPathComponent("chunk_\(chunk).codes.bin"),
                fallbacks: ["\(chunk).codes.bin"]
            )
        }
        func residuals(for chunk: Int) throws -> URL {
            try existingURL(
                primary: root.appendingPathComponent("chunk_\(chunk).residuals.bin"),
                fallbacks: ["\(chunk).residuals.bin"]
            )
        }
        func docLens(for chunk: Int) throws -> URL {
            try existingURL(
                primary: root.appendingPathComponent("doclens.\(chunk).json"),
                fallbacks: ["chunk_\(chunk).doclens.json", "\(chunk).doclens.json"]
            )
        }
        func chunkMetadata(for chunk: Int) throws -> URL {
            try existingURL(
                primary: root.appendingPathComponent("chunk_\(chunk).metadata.json"),
                fallbacks: ["\(chunk).metadata.json"]
            )
        }
        func ivf() -> URL { root.appendingPathComponent("ivf.bin") }
        func ivfLengths() -> URL { root.appendingPathComponent("ivf_lengths.bin") }

        private func existingURL(primary: URL, fallbacks: [String]) throws -> URL {
            if FileManager.default.fileExists(atPath: primary.path) {
                return primary
            }
            for candidate in fallbacks {
                let url = root.appendingPathComponent(candidate)
                if FileManager.default.fileExists(atPath: url.path) {
                    return url
                }
            }
            throw PlaidError.indexNotFound(primary)
        }
    }

    struct PlanMetadata: Codable {
        let nbits: Int
        let num_chunks: Int
        let centroid_count: Int?

        var centroidCount: Int? { centroid_count }
        var numChunks: Int { num_chunks }
    }

    struct FinalMetadata: Codable {
        var num_chunks: Int
        var nbits: Int
        var num_partitions: Int
        var num_embeddings: Int
        var avg_doclen: Double
        var embedding_dim: Int
        var total_documents: Int
        var bytes_per_residual: Int

        var numBuckets: Int { 1 << nbits }

        init(
            num_chunks: Int = 0,
            nbits: Int = 0,
            num_partitions: Int = 0,
            num_embeddings: Int = 0,
            avg_doclen: Double = 0,
            embedding_dim: Int = 0,
            total_documents: Int = 0,
            bytes_per_residual: Int = 0
        ) {
            self.num_chunks = num_chunks
            self.nbits = nbits
            self.num_partitions = num_partitions
            self.num_embeddings = num_embeddings
            self.avg_doclen = avg_doclen
            self.embedding_dim = embedding_dim
            self.total_documents = total_documents
            self.bytes_per_residual = bytes_per_residual
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            num_chunks = try container.decodeIfPresent(Int.self, forKey: .num_chunks) ?? 0
            nbits = try container.decodeIfPresent(Int.self, forKey: .nbits) ?? 0
            num_partitions = try container.decodeIfPresent(Int.self, forKey: .num_partitions) ?? 0
            num_embeddings = try container.decodeIfPresent(Int.self, forKey: .num_embeddings) ?? 0
            avg_doclen = try container.decodeIfPresent(Double.self, forKey: .avg_doclen) ?? 0
            embedding_dim = try container.decodeIfPresent(Int.self, forKey: .embedding_dim) ?? 0
            total_documents = try container.decodeIfPresent(Int.self, forKey: .total_documents) ?? 0
            bytes_per_residual =
                try container.decodeIfPresent(Int.self, forKey: .bytes_per_residual) ?? 0
        }
    }

    struct ChunkMetadata: Codable {
        let num_passages: Int
        let num_embeddings: Int
        let embedding_offset: Int
    }

    static func readJSON<T: Decodable>(from url: URL) throws -> T {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(T.self, from: data)
    }

    static func readFloat32Array(from url: URL, shape: [Int]) throws -> [Float32] {
        let data = try Data(contentsOf: url)
        let count = shape.reduce(1, *)
        guard data.count == count * MemoryLayout<Float32>.size else {
            throw PlaidError.invalidEmbeddingDimensions(
                expected: count * MemoryLayout<Float>.size, actual: data.count)
        }
        return data.withUnsafeBytes { raw in
            let ptr = raw.baseAddress!.assumingMemoryBound(to: Float32.self)
            return Array(UnsafeBufferPointer(start: ptr, count: count))
        }
    }

    static func readInt32Array(from url: URL) throws -> [Int32] {
        let data = try Data(contentsOf: url)
        let count = data.count / MemoryLayout<Int32>.size
        return data.withUnsafeBytes { raw in
            let ptr = raw.baseAddress!.assumingMemoryBound(to: Int32.self)
            return Array(UnsafeBufferPointer(start: ptr, count: count))
        }
    }

    static func loadFloatArray(from url: URL) throws -> [Float32] {
        let data = try Data(contentsOf: url)
        let count = data.count / MemoryLayout<Float32>.size
        return data.withUnsafeBytes { raw in
            let ptr = raw.baseAddress!.assumingMemoryBound(to: Float32.self)
            return Array(UnsafeBufferPointer(start: ptr, count: count))
        }
    }

    private struct NpyHeader: Decodable {
        let descr: String
        let fortran_order: Bool
        let shape: [Int]
    }

    static func readNpyInt32(from url: URL) throws -> [Int32] {
        let npy = try readNpyArray(from: url)
        let count = npy.header.shape.reduce(1, *)
        switch npy.header.descr {
        case "<i4", "|i4":
            let expectedBytes = count * MemoryLayout<Int32>.size
            guard npy.body.count == expectedBytes else {
                throw PlaidError.invalidEmbeddingDimensions(
                    expected: expectedBytes, actual: npy.body.count)
            }
            return npy.body.withUnsafeBytes { raw in
                let ptr = raw.baseAddress!.assumingMemoryBound(to: Int32.self)
                return Array(UnsafeBufferPointer(start: ptr, count: count))
            }
        case "<i8", "|i8":
            let expectedBytes = count * MemoryLayout<Int64>.size
            guard npy.body.count == expectedBytes else {
                throw PlaidError.invalidEmbeddingDimensions(
                    expected: expectedBytes, actual: npy.body.count)
            }
            return npy.body.withUnsafeBytes { raw in
                let ptr = raw.baseAddress!.assumingMemoryBound(to: Int64.self)
                let buffer = UnsafeBufferPointer(start: ptr, count: count)
                return buffer.map { Int32(truncatingIfNeeded: $0) }
            }
        case "<i2", "|i2":
            let expectedBytes = count * MemoryLayout<Int16>.size
            guard npy.body.count == expectedBytes else {
                throw PlaidError.invalidEmbeddingDimensions(
                    expected: expectedBytes, actual: npy.body.count)
            }
            return npy.body.withUnsafeBytes { raw in
                let ptr = raw.baseAddress!.assumingMemoryBound(to: Int16.self)
                let buffer = UnsafeBufferPointer(start: ptr, count: count)
                return buffer.map { Int32($0) }
            }
        default:
            throw PlaidError.invalidEmbeddingDimensions(expected: 0, actual: 0)
        }
    }

    static func readNpyUInt8Data(from url: URL) throws -> Data {
        let npy = try readNpyArray(from: url)
        guard npy.header.descr == "|u1" || npy.header.descr == "<u1" else {
            throw PlaidError.invalidEmbeddingDimensions(expected: 0, actual: 0)
        }
        return npy.body
    }

    static func readNpyFloat32(from url: URL) throws -> [Float32] {
        let npy = try readNpyArray(from: url)
        guard npy.header.descr == "<f4" || npy.header.descr == "|f4" else {
            throw PlaidError.invalidEmbeddingDimensions(expected: 0, actual: 0)
        }
        let count = npy.header.shape.reduce(1, *)
        let expectedBytes = count * MemoryLayout<Float32>.size
        guard npy.body.count == expectedBytes else {
            throw PlaidError.invalidEmbeddingDimensions(
                expected: expectedBytes, actual: npy.body.count)
        }
        return npy.body.withUnsafeBytes { raw in
            let ptr = raw.baseAddress!.assumingMemoryBound(to: Float32.self)
            return Array(UnsafeBufferPointer(start: ptr, count: count))
        }
    }

    private static func readNpyArray(from url: URL) throws -> (header: NpyHeader, body: Data) {
        let data = try Data(contentsOf: url)
        guard data.count >= 10 else {
            throw PlaidError.invalidEmbeddingDimensions(expected: 10, actual: data.count)
        }
        guard data.prefix(6) == Data([0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59]) else {
            throw PlaidError.invalidEmbeddingDimensions(expected: 0, actual: 0)
        }
        let major = data[6]
        _ = data[7]
        let headerLen: Int
        let headerStart: Int
        switch major {
        case 1:
            let le: UInt16 = data.subdata(in: 8 ..< 10).withUnsafeBytes { $0.load(as: UInt16.self) }
            headerLen = Int(UInt16(littleEndian: le))
            headerStart = 10
        case 2:
            let le: UInt32 = data.subdata(in: 8 ..< 12).withUnsafeBytes { $0.load(as: UInt32.self) }
            headerLen = Int(UInt32(littleEndian: le))
            headerStart = 12
        default:
            throw PlaidError.invalidEmbeddingDimensions(expected: 0, actual: Int(major))
        }
        let headerEnd = headerStart + headerLen
        guard headerEnd <= data.count else {
            throw PlaidError.invalidEmbeddingDimensions(expected: headerEnd, actual: data.count)
        }
        let headerData = data.subdata(in: headerStart ..< headerEnd)
        guard var headerStr = String(data: headerData, encoding: .ascii) else {
            throw PlaidError.invalidEmbeddingDimensions(expected: 0, actual: 0)
        }
        headerStr = headerStr.trimmingCharacters(in: .whitespacesAndNewlines)
        var jsonText = headerStr
        jsonText = jsonText.replacingOccurrences(of: "'", with: "\"")
        jsonText = jsonText.replacingOccurrences(of: "False", with: "false")
        jsonText = jsonText.replacingOccurrences(of: "True", with: "true")
        jsonText = jsonText.replacingOccurrences(of: "(", with: "[")
        jsonText = jsonText.replacingOccurrences(of: ")", with: "]")
        jsonText = jsonText.replacingOccurrences(of: ", }", with: "}")
        jsonText = jsonText.replacingOccurrences(of: ",}", with: "}")
        jsonText = jsonText.replacingOccurrences(of: ", ]", with: "]")
        jsonText = jsonText.replacingOccurrences(of: ",]", with: "]")
        if !jsonText.hasPrefix("{") {
            jsonText = "{\(jsonText)"
        }
        if !jsonText.hasSuffix("}") {
            jsonText += "}"
        }
        guard let jsonData = jsonText.data(using: .utf8) else {
            throw PlaidError.invalidEmbeddingDimensions(expected: 0, actual: 0)
        }
        let header = try JSONDecoder().decode(NpyHeader.self, from: jsonData)
        guard header.fortran_order == false else {
            throw PlaidError.invalidEmbeddingDimensions(expected: 0, actual: 1)
        }
        let body = data.subdata(in: headerEnd ..< data.count)
        return (header, body)
    }
}
