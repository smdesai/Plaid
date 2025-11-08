import Foundation
import MLX

public enum PlaidError: Error, LocalizedError {
    case invalidEmbeddingDimensions(expected: Int, actual: Int)
    case emptyEmbeddingSet
    case mismatchedQueryDimension(expected: Int, actual: Int)
    case indexNotFound(URL)
    case invalidSubset(String)
    case invalidDocumentId(Int, totalDocuments: Int)

    public var errorDescription: String? {
        switch self {
        case .invalidEmbeddingDimensions(let expected, let actual):
            return "Embedding dimension mismatch. Expected \(expected), got \(actual)."
        case .emptyEmbeddingSet:
            return "At least one embedding row is required."
        case .mismatchedQueryDimension(let expected, let actual):
            return "Query dimension mismatch. Expected \(expected), got \(actual)."
        case .indexNotFound(let url):
            return "No index materialized at \(url.path)."
        case .invalidSubset(let reason):
            return "Subset validation failed: \(reason)."
        case .invalidDocumentId(let docId, let totalDocuments):
            return "Document ID \(docId) is out of range. Valid range is 0..<\(totalDocuments)."
        }
    }
}

public struct SearchParameters: Codable, Sendable {
    public var batchSize: Int
    public var nFullScores: Int
    public var topK: Int
    public var nIvfProbe: Int
    public var logTiming: Bool

    enum CodingKeys: String, CodingKey {
        case batchSize
        case nFullScores
        case topK
        case nIvfProbe
        case logTiming
    }

    public init(
        batchSize: Int, nFullScores: Int, topK: Int, nIvfProbe: Int, logTiming: Bool = false
    ) {
        self.batchSize = batchSize
        self.nFullScores = nFullScores
        self.topK = topK
        self.nIvfProbe = nIvfProbe
        self.logTiming = logTiming
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        batchSize = try container.decode(Int.self, forKey: .batchSize)
        nFullScores = try container.decode(Int.self, forKey: .nFullScores)
        topK = try container.decode(Int.self, forKey: .topK)
        nIvfProbe = try container.decode(Int.self, forKey: .nIvfProbe)
        logTiming = try container.decodeIfPresent(Bool.self, forKey: .logTiming) ?? false
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(batchSize, forKey: .batchSize)
        try container.encode(nFullScores, forKey: .nFullScores)
        try container.encode(topK, forKey: .topK)
        try container.encode(nIvfProbe, forKey: .nIvfProbe)
        if logTiming {
            try container.encode(logTiming, forKey: .logTiming)
        }
    }
}

public struct QueryResult: Codable, Sendable {
    public let queryId: Int
    public let passageIds: [Int]
    public let scores: [Float]

    public init(queryId: Int, passageIds: [Int], scores: [Float]) {
        self.queryId = queryId
        self.passageIds = passageIds
        self.scores = scores
    }
}

/// Caches extracted arrays from MLX tensors to avoid repeated conversions
/// during search operations. This provides significant performance improvements
/// by eliminating redundant asArray() calls in hot paths.
private struct CachedSearchArtifacts {
    let centroidValues: [Float]
    let bucketWeightsArray: [Float]?
    let byteMapArray: [Int]
    let bucketLookupArray: [Int]?
    let centroidCount: Int
    let embeddingDim: Int
    let codecNBits: Int
    let numBuckets: Int
    let keysPerByte: Int
    let bucketWeightsShape: [Int]

    /// Creates cached artifacts from index artifacts
    static func from(artifacts: IndexArtifacts, codecArtifacts: CodecArtifacts)
        -> CachedSearchArtifacts?
    {
        let centroidTensor = artifacts.centroids
        let centroidValues = centroidTensor.asArray(Float.self)
        let centroidCount = centroidTensor.shape[0]
        let embeddingDim = artifacts.embeddingDim

        guard embeddingDim > 0, centroidCount > 0 else {
            return nil
        }

        let bucketWeightsArray: [Float]? = artifacts.bucketWeights?.asArray(Float.self)
        let byteMapArray = codecArtifacts.byteReversedBitsMap.asArray(Int.self)
        let bucketLookupArray = codecArtifacts.bucketWeightLookup?.asArray(Int.self)

        let codecNBits = artifacts.nbits
        guard codecNBits > 0 && codecNBits <= 8 else {
            return nil
        }
        let numBuckets = 1 << codecNBits
        let keysPerByte = max(1, 8 / codecNBits)
        let bucketWeightsShape = artifacts.bucketWeights?.shape ?? []

        return CachedSearchArtifacts(
            centroidValues: centroidValues,
            bucketWeightsArray: bucketWeightsArray,
            byteMapArray: byteMapArray,
            bucketLookupArray: bucketLookupArray,
            centroidCount: centroidCount,
            embeddingDim: embeddingDim,
            codecNBits: codecNBits,
            numBuckets: numBuckets,
            keysPerByte: keysPerByte,
            bucketWeightsShape: bucketWeightsShape
        )
    }
}

private struct PersistedIndex: Codable {
    let embeddingDim: Int
    let nbits: Int
    let docVectors: [[Float]]
    let docLengths: [Int]
}

private struct PlaidIndex {
    let artifacts: IndexArtifacts?
    let docVectors: [[Float]]
    let docLengths: [Int]
    let embeddingDim: Int
    let nbits: Int

    init(artifacts: IndexArtifacts?, summary: PersistedIndex) {
        self.artifacts = artifacts
        self.docVectors = summary.docVectors
        self.docLengths = summary.docLengths
        self.embeddingDim = summary.embeddingDim
        self.nbits = summary.nbits
    }
}

private struct CacheKey: Hashable {
    let path: String
}

private final class PlaidIndexCache {
    static let shared = PlaidIndexCache()

    private var cache: [CacheKey: PlaidIndex] = [:]
    private let queue = DispatchQueue(label: "plaid.cache", attributes: .concurrent)

    func value(for key: CacheKey) -> PlaidIndex? {
        queue.sync {
            cache[key]
        }
    }

    func store(_ value: PlaidIndex, for key: CacheKey) {
        queue.async(flags: .barrier) {
            self.cache[key] = value
        }
    }

    func clear(for path: String) {
        queue.async(flags: .barrier) {
            self.cache = self.cache.filter { $0.key.path != path }
        }
    }
}

private final class SearchArtifactsCache {
    static let shared = SearchArtifactsCache()

    private var cache: [CacheKey: CachedSearchArtifacts] = [:]
    private let queue = DispatchQueue(
        label: "plaid.search_artifacts_cache", attributes: .concurrent)

    func value(for key: CacheKey) -> CachedSearchArtifacts? {
        queue.sync {
            cache[key]
        }
    }

    func store(_ value: CachedSearchArtifacts, for key: CacheKey) {
        queue.async(flags: .barrier) {
            self.cache[key] = value
        }
    }

    func clear(for path: String) {
        queue.async(flags: .barrier) {
            self.cache = self.cache.filter { $0.key.path != path }
        }
    }
}

public enum Plaid {
    private static let indexFileName = "plaid_index.json"

    public static func initializeTorch(torchPath: String) {
        // No-op placeholder retained for API parity.
    }

    public static func create(
        indexPath: String,
        torchPath: String? = nil,
        embeddingDim: Int,
        nbits: Int,
        embeddings: [[[Float]]],
        centroids: [[Float]] = [],
        batchSize: Int,
        seed: UInt64? = nil
    ) throws {
        try create(
            indexURL: URL(fileURLWithPath: indexPath),
            torchPath: torchPath,
            embeddingDim: embeddingDim,
            nbits: nbits,
            embeddings: embeddings,
            centroids: centroids,
            batchSize: batchSize,
            seed: seed
        )
    }

    public static func create(
        indexURL: URL,
        torchPath: String? = nil,
        embeddingDim: Int,
        nbits: Int,
        embeddings: [[[Float]]],
        centroids: [[Float]] = [],
        batchSize: Int,
        seed: UInt64? = nil
    ) throws {
        guard !embeddings.isEmpty else {
            throw PlaidError.emptyEmbeddingSet
        }

        guard !centroids.isEmpty else {
            throw PlaidError.emptyEmbeddingSet
        }

        let builder = IndexBuilder(
            outputURL: indexURL,
            embeddingDim: embeddingDim,
            nbits: nbits,
            batchSize: batchSize,
            seed: seed,
            documents: embeddings,
            centroids: centroids
        )

        let buildResult = try builder.build()

        let summary = PersistedIndex(
            embeddingDim: embeddingDim,
            nbits: nbits,
            docVectors: buildResult.docVectors,
            docLengths: buildResult.docLengths
        )

        try persist(summary: summary, at: indexURL)

        PlaidIndexCache.shared.clear(for: cachePath(indexURL))
        SearchArtifactsCache.shared.clear(for: cachePath(indexURL))
    }

    public static func update(
        indexURL: URL,
        torchPath: String? = nil,
        embeddings: [[[Float]]],
        batchSize: Int
    ) throws {
        guard !embeddings.isEmpty else {
            return
        }

        // Acquire lock to prevent concurrent modifications
        let lock = try IndexLock.acquire(for: indexURL)
        defer { lock.release() }

        let summary = try loadSummary(from: indexURL)
        let artifacts = try IndexStorage.loadArtifacts(from: indexURL)

        guard summary.embeddingDim == artifacts.embeddingDim else {
            throw PlaidError.invalidEmbeddingDimensions(
                expected: summary.embeddingDim,
                actual: artifacts.embeddingDim
            )
        }

        let appendResult = try appendDocuments(
            indexURL: indexURL,
            embeddings: embeddings,
            batchSize: batchSize,
            summary: summary,
            artifacts: artifacts
        )

        guard appendResult.newDocumentCount > 0 else {
            return
        }

        let updatedSummary = PersistedIndex(
            embeddingDim: summary.embeddingDim,
            nbits: summary.nbits,
            docVectors: appendResult.updatedDocVectors,
            docLengths: appendResult.updatedDocLengths
        )

        try persist(summary: updatedSummary, at: indexURL)
        PlaidIndexCache.shared.clear(for: cachePath(indexURL))
        SearchArtifactsCache.shared.clear(for: cachePath(indexURL))
    }

    public static func update(
        indexPath: String,
        torchPath: String? = nil,
        embeddings: [[[Float]]],
        batchSize: Int
    ) throws {
        try update(
            indexURL: URL(fileURLWithPath: indexPath),
            torchPath: torchPath,
            embeddings: embeddings,
            batchSize: batchSize
        )
    }

    public static func preloadIndex(
        indexURL: URL,
        torchPath: String? = nil
    ) throws {
        let key = CacheKey(path: cachePath(indexURL))
        if PlaidIndexCache.shared.value(for: key) != nil {
            return
        }
        let index = try loadIndexFromDisk(indexURL: indexURL)
        PlaidIndexCache.shared.store(index, for: key)
    }

    public static func preloadIndex(
        indexPath: String,
        torchPath: String? = nil
    ) throws {
        try preloadIndex(
            indexURL: URL(fileURLWithPath: indexPath),
            torchPath: torchPath
        )
    }

    public static func loadAndSearch(
        indexURL: URL,
        torchPath: String? = nil,
        queries: [[[Float]]],
        searchParameters: SearchParameters,
        showProgress: Bool,
        preloadIndex: Bool,
        subset: [[Int]]? = nil
    ) throws -> [QueryResult] {
        let key = CacheKey(path: cachePath(indexURL))
        let index: PlaidIndex

        if preloadIndex, let cached = PlaidIndexCache.shared.value(for: key) {
            index = cached
        } else {
            index = try loadIndexFromDisk(indexURL: indexURL)
            if preloadIndex {
                PlaidIndexCache.shared.store(index, for: key)
            }
        }

        if let artifacts = index.artifacts {
            // Try to get cached search artifacts, or create and cache them
            let cachedArtifacts: CachedSearchArtifacts
            if let cached = SearchArtifactsCache.shared.value(for: key) {
                cachedArtifacts = cached
            } else {
                guard
                    let created = CachedSearchArtifacts.from(
                        artifacts: artifacts,
                        codecArtifacts: artifacts.codecArtifacts
                    )
                else {
                    return try runLegacySearch(
                        index: index, queries: queries, params: searchParameters, subset: subset)
                }
                cachedArtifacts = created
                SearchArtifactsCache.shared.store(cachedArtifacts, for: key)
            }

            return try runSearch(
                artifacts: artifacts,
                summary: index,
                queries: queries,
                params: searchParameters,
                subset: subset,
                cachedArtifacts: cachedArtifacts
            )
        }

        return try runLegacySearch(
            index: index, queries: queries, params: searchParameters, subset: subset)
    }

    public static func loadAndSearch(
        indexPath: String,
        torchPath: String? = nil,
        queries: [[[Float]]],
        searchParameters: SearchParameters,
        showProgress: Bool,
        preloadIndex: Bool,
        subset: [[Int]]? = nil
    ) throws -> [QueryResult] {
        try loadAndSearch(
            indexURL: URL(fileURLWithPath: indexPath),
            torchPath: torchPath,
            queries: queries,
            searchParameters: searchParameters,
            showProgress: showProgress,
            preloadIndex: preloadIndex,
            subset: subset
        )
    }

    public static func delete(
        indexURL: URL,
        torchPath: String? = nil,
        subset: [Int]
    ) throws {
        guard !subset.isEmpty else {
            return
        }

        // Acquire lock to prevent concurrent modifications
        let lock = try IndexLock.acquire(for: indexURL)
        defer { lock.release() }

        let summary = try loadSummary(from: indexURL)
        let validIds = Set(subset.filter { $0 >= 0 && $0 < summary.docLengths.count })
        guard !validIds.isEmpty else { return }

        let artifacts = try IndexStorage.loadArtifacts(from: indexURL)

        let removal = try removeDocuments(
            indexURL: indexURL,
            deleteSet: validIds,
            summary: summary,
            artifacts: artifacts
        )

        guard removal.didModify else { return }

        let updatedSummary = PersistedIndex(
            embeddingDim: summary.embeddingDim,
            nbits: summary.nbits,
            docVectors: removal.updatedDocVectors,
            docLengths: removal.updatedDocLengths
        )
        try persist(summary: updatedSummary, at: indexURL)
        PlaidIndexCache.shared.clear(for: cachePath(indexURL))
        SearchArtifactsCache.shared.clear(for: cachePath(indexURL))
    }

    public static func delete(
        indexPath: String,
        torchPath: String? = nil,
        subset: [Int]
    ) throws {
        try delete(
            indexURL: URL(fileURLWithPath: indexPath),
            torchPath: torchPath,
            subset: subset
        )
    }

    /// Retrieves the full decompressed token-level embeddings for a specific document.
    ///
    /// - Parameters:
    ///   - indexURL: URL to the index directory
    ///   - documentId: The ID of the document to retrieve embeddings for
    ///   - torchPath: Optional torch path (unused, retained for API parity)
    /// - Returns: A 2D array of embeddings where each inner array represents a token embedding.
    ///           Returns `[[Float]]` with shape `[num_tokens, embedding_dim]`
    /// - Throws: `PlaidError.invalidDocumentId` if the document ID is out of range,
    ///          `PlaidError.indexNotFound` if the index doesn't exist
    public static func getDocumentEmbeddings(
        indexURL: URL,
        documentId: Int,
        torchPath: String? = nil
    ) throws -> [[Float]] {
        // Load index artifacts
        let artifacts = try IndexStorage.loadArtifacts(from: indexURL)

        // Validate document ID
        guard documentId >= 0 && documentId < artifacts.docLengths.count else {
            throw PlaidError.invalidDocumentId(
                documentId, totalDocuments: artifacts.docLengths.count)
        }

        // Get document boundaries
        let docStart = artifacts.embeddingOffsets[documentId]
        let docEnd = artifacts.embeddingOffsets[documentId + 1]
        let docLen = docEnd - docStart

        guard docLen > 0 else {
            // Empty document, return empty array
            return []
        }

        // Decompress embeddings
        let embeddingTensor = artifacts.residuals.withUnsafeBytes { rawBuffer -> MLXArray? in
            let residualBytes = rawBuffer.bindMemory(to: UInt8.self)
            guard let baseAddress = residualBytes.baseAddress else { return nil }

            // Determine which decompression method to use
            let useVectorized: Bool
            if let bucketWeights = artifacts.bucketWeights,
                artifacts.codecArtifacts.bucketWeightLookup != nil
            {
                let shape = bucketWeights.shape
                let nbits = artifacts.nbits
                let numBuckets = 1 << nbits
                useVectorized =
                    shape.count == 2
                    && shape[0] == numBuckets
                    && shape[1] == artifacts.embeddingDim
            } else {
                useVectorized = false
            }

            if useVectorized {
                return decompressDocumentVectorized(
                    docStart: docStart,
                    docLen: docLen,
                    artifacts: artifacts,
                    residualBytes: baseAddress,
                    stream: .default
                )
            } else {
                // Create cached artifacts for scalar decompression
                let centroidValues = artifacts.centroids.asArray(Float.self)
                let bucketWeightsArray = artifacts.bucketWeights?.asArray(Float.self)
                let byteMapArray = artifacts.codecArtifacts.byteReversedBitsMap.asArray(Int.self)
                let bucketLookupArray = artifacts.codecArtifacts.bucketWeightLookup?.asArray(
                    Int.self)

                let nbits = artifacts.nbits
                let numBuckets = 1 << nbits
                let keysPerByte = max(1, 8 / nbits)

                return decompressDocumentScalar(
                    docStart: docStart,
                    docLen: docLen,
                    embeddingDim: artifacts.embeddingDim,
                    nbits: nbits,
                    codes: artifacts.codes,
                    centroidValues: centroidValues,
                    bucketWeights: bucketWeightsArray,
                    residualBytes: baseAddress,
                    bytesPerResidual: artifacts.bytesPerResidual,
                    numBuckets: numBuckets,
                    residualCount: artifacts.residuals.count,
                    keysPerByte: keysPerByte,
                    byteMap: byteMapArray,
                    bucketLookup: bucketLookupArray
                )
            }
        }

        guard let tensor = embeddingTensor else {
            throw PlaidError.emptyEmbeddingSet
        }

        // Convert MLXArray to [[Float]]
        let values = tensor.asArray(Float32.self)
        let numTokens = docLen
        let embeddingDim = artifacts.embeddingDim

        var result: [[Float]] = []
        result.reserveCapacity(numTokens)

        for tokenIdx in 0 ..< numTokens {
            let start = tokenIdx * embeddingDim
            let end = start + embeddingDim
            let tokenEmbedding = values[start ..< end].map { Float($0) }
            result.append(tokenEmbedding)
        }

        return result
    }

    /// Retrieves the full decompressed token-level embeddings for a specific document.
    ///
    /// - Parameters:
    ///   - indexPath: Path to the index directory
    ///   - documentId: The ID of the document to retrieve embeddings for
    ///   - torchPath: Optional torch path (unused, retained for API parity)
    /// - Returns: A 2D array of embeddings where each inner array represents a token embedding.
    ///           Returns `[[Float]]` with shape `[num_tokens, embedding_dim]`
    /// - Throws: `PlaidError.invalidDocumentId` if the document ID is out of range,
    ///          `PlaidError.indexNotFound` if the index doesn't exist
    public static func getDocumentEmbeddings(
        indexPath: String,
        documentId: Int,
        torchPath: String? = nil
    ) throws -> [[Float]] {
        try getDocumentEmbeddings(
            indexURL: URL(fileURLWithPath: indexPath),
            documentId: documentId,
            torchPath: torchPath
        )
    }
}

extension Plaid {
    fileprivate struct AppendResult {
        let updatedDocVectors: [[Float]]
        let updatedDocLengths: [Int]
        let newDocumentCount: Int
    }

    fileprivate static func cachePath(_ indexURL: URL) -> String {
        indexURL.standardizedFileURL.path
    }

    fileprivate static func persist(summary: PersistedIndex, at url: URL) throws {
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(summary)
        let fileURL = url.appendingPathComponent(indexFileName)
        try data.write(to: fileURL, options: .atomic)
    }

    fileprivate static func loadSummary(from indexURL: URL) throws -> PersistedIndex {
        let fileURL = indexURL.appendingPathComponent(indexFileName)
        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            throw PlaidError.indexNotFound(fileURL)
        }
        let data = try Data(contentsOf: fileURL)
        let decoder = JSONDecoder()
        return try decoder.decode(PersistedIndex.self, from: data)
    }

    fileprivate static func loadIndexFromDisk(indexURL: URL) throws -> PlaidIndex {
        let fileURL = indexURL.appendingPathComponent(indexFileName)
        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            throw PlaidError.indexNotFound(fileURL)
        }

        let data = try Data(contentsOf: fileURL)
        let decoder = JSONDecoder()
        let summary = try decoder.decode(PersistedIndex.self, from: data)

        let artifacts = try? IndexStorage.loadArtifacts(from: indexURL)
        return PlaidIndex(artifacts: artifacts, summary: summary)
    }

    fileprivate static func appendDocuments(
        indexURL: URL,
        embeddings: [[[Float]]],
        batchSize: Int,
        summary: PersistedIndex,
        artifacts: IndexArtifacts
    ) throws -> AppendResult {
        guard !embeddings.isEmpty else {
            return AppendResult(
                updatedDocVectors: summary.docVectors,
                updatedDocLengths: summary.docLengths,
                newDocumentCount: 0
            )
        }

        let centroidRows = extractRows(from: artifacts.centroids)
        guard !centroidRows.isEmpty else {
            throw PlaidError.emptyEmbeddingSet
        }

        let builder = IndexBuilder(
            outputURL: indexURL,
            embeddingDim: artifacts.embeddingDim,
            nbits: artifacts.nbits,
            batchSize: batchSize,
            seed: nil,
            documents: embeddings,
            centroids: centroidRows
        )

        let (documentTensors, docVectors, docLengths) = try builder.prepareDocuments()
        guard !documentTensors.isEmpty else {
            return AppendResult(
                updatedDocVectors: summary.docVectors,
                updatedDocLengths: summary.docLengths,
                newDocumentCount: 0
            )
        }

        guard let bucketCutoffs = artifacts.bucketCutoffs,
            let bucketWeights = artifacts.bucketWeights
        else {
            throw PlaidError.emptyEmbeddingSet
        }

        let codec = try ResidualCodec(
            nbits: artifacts.nbits,
            centroids: artifacts.centroids,
            avgResidual: artifacts.avgResidual,
            bucketCutoffs: bucketCutoffs,
            bucketWeights: bucketWeights
        )

        let cutoffsPerDim = try builder.makeCutoffsPerDimension(from: codec)
        let bytesPerResidual = artifacts.bytesPerResidual
        guard bytesPerResidual > 0 else {
            throw PlaidError.invalidEmbeddingDimensions(expected: 1, actual: bytesPerResidual)
        }

        let existingChunkCount = artifacts.numChunks
        let chunkSize = builder.computeChunkSize(documentCount: documentTensors.count)
        let paths = IndexBuilder.IndexPaths(root: indexURL)

        var nextChunkIndex = existingChunkCount
        var docIndex = 0
        var embeddingOffset = artifacts.embeddingOffsets.last ?? artifacts.codes.count
        var accumulatedCodes: [Int32] = []
        accumulatedCodes.reserveCapacity(documentTensors.reduce(0) { $0 + $1.length })

        while docIndex < documentTensors.count {
            let end = min(docIndex + chunkSize, documentTensors.count)
            let chunkDocs = Array(documentTensors[docIndex ..< end])
            let chunkDocLengths = chunkDocs.map { $0.length }

            let (codes, packedResiduals, chunkEmbeddings) = try builder.encodeChunk(
                documents: chunkDocs,
                codec: codec,
                cutoffsPerDim: cutoffsPerDim,
                bytesPerResidual: bytesPerResidual
            )

            accumulatedCodes.append(contentsOf: codes)
            try BinaryIO.writeInt32(codes, to: paths.codes(for: nextChunkIndex))
            try BinaryIO.writeUInt8(packedResiduals, to: paths.residuals(for: nextChunkIndex))
            try builder.writeDocLengths(chunkDocLengths, to: paths.docLens(for: nextChunkIndex))

            let metadata = IndexBuilder.ChunkMetadata(
                num_passages: chunkDocLengths.count,
                num_embeddings: chunkEmbeddings,
                embedding_offset: embeddingOffset
            )
            try builder.writeJSON(metadata, to: paths.chunkMetadata(for: nextChunkIndex))

            embeddingOffset += chunkEmbeddings
            nextChunkIndex += 1
            docIndex = end
        }

        let newChunkCount = nextChunkIndex - existingChunkCount

        var updatedDocVectors = summary.docVectors
        updatedDocVectors.append(contentsOf: docVectors)
        var updatedDocLengths = summary.docLengths
        updatedDocLengths.append(contentsOf: docLengths)

        var combinedCodes = artifacts.codes
        combinedCodes.append(contentsOf: accumulatedCodes)

        let (ivf, ivfLengths) = try builder.buildIVF(
            codes: combinedCodes,
            docLengths: updatedDocLengths,
            codec: codec
        )
        try BinaryIO.writeInt32(ivf, to: paths.ivf())
        try BinaryIO.writeInt32(ivfLengths, to: paths.ivfLengths())

        try builder.writePlan(
            paths: paths,
            numChunks: existingChunkCount + newChunkCount,
            centroidCount: codec.numCentroids
        )

        let totalEmbeddings = combinedCodes.count
        let totalDocuments = updatedDocLengths.count
        let avgDocLen =
            totalDocuments > 0
            ? Double(totalEmbeddings) / Double(totalDocuments)
            : 0.0

        let finalMetadata = IndexBuilder.FinalMetadata(
            num_chunks: existingChunkCount + newChunkCount,
            nbits: artifacts.nbits,
            num_partitions: artifacts.numPartitions,
            num_embeddings: totalEmbeddings,
            avg_doclen: avgDocLen,
            embedding_dim: artifacts.embeddingDim,
            total_documents: totalDocuments,
            bytes_per_residual: bytesPerResidual
        )
        try builder.writeJSON(finalMetadata, to: paths.metadata())

        return AppendResult(
            updatedDocVectors: updatedDocVectors,
            updatedDocLengths: updatedDocLengths,
            newDocumentCount: docVectors.count
        )
    }

    fileprivate static func extractRows(from tensor: MLXArray) -> [[Float]] {
        guard tensor.shape.count == 2 else { return [] }
        let rows = tensor.shape[0]
        let cols = tensor.shape[1]
        let values = tensor.asArray(Float32.self)
        var result: [[Float]] = []
        result.reserveCapacity(rows)
        for row in 0 ..< rows {
            let start = row * cols
            let end = start + cols
            var rowValues: [Float] = []
            rowValues.reserveCapacity(cols)
            for idx in start ..< end {
                rowValues.append(Float(values[idx]))
            }
            result.append(rowValues)
        }
        return result
    }

    fileprivate struct RemovalResult {
        let updatedDocVectors: [[Float]]
        let updatedDocLengths: [Int]
        let didModify: Bool
    }

    fileprivate static func removeDocuments(
        indexURL: URL,
        deleteSet: Set<Int>,
        summary: PersistedIndex,
        artifacts: IndexArtifacts
    ) throws -> RemovalResult {
        guard !deleteSet.isEmpty else {
            return RemovalResult(
                updatedDocVectors: summary.docVectors,
                updatedDocLengths: summary.docLengths,
                didModify: false
            )
        }

        let centroidRows = extractRows(from: artifacts.centroids)
        guard !centroidRows.isEmpty else {
            throw PlaidError.emptyEmbeddingSet
        }

        guard let bucketCutoffs = artifacts.bucketCutoffs,
            let bucketWeights = artifacts.bucketWeights
        else {
            throw PlaidError.emptyEmbeddingSet
        }

        let builder = IndexBuilder(
            outputURL: indexURL,
            embeddingDim: artifacts.embeddingDim,
            nbits: artifacts.nbits,
            batchSize: 1,
            seed: nil,
            documents: [],
            centroids: centroidRows
        )

        let codec = try ResidualCodec(
            nbits: artifacts.nbits,
            centroids: artifacts.centroids,
            avgResidual: artifacts.avgResidual,
            bucketCutoffs: bucketCutoffs,
            bucketWeights: bucketWeights
        )

        let bytesPerResidual = artifacts.bytesPerResidual
        guard bytesPerResidual > 0 else {
            throw PlaidError.invalidEmbeddingDimensions(expected: 1, actual: bytesPerResidual)
        }

        let paths = IndexBuilder.IndexPaths(root: indexURL)
        let numChunks = artifacts.numChunks

        var updatedDocLengths: [Int] = []
        updatedDocLengths.reserveCapacity(summary.docLengths.count - deleteSet.count)
        var updatedCodes: [Int32] = []
        updatedCodes.reserveCapacity(max(0, artifacts.codes.count - deleteSet.count))

        var embeddingOffset = 0
        var globalDocId = 0
        var anyRemoved = false

        for chunkIndex in 0 ..< numChunks {
            let docLensURL = paths.docLens(for: chunkIndex)
            let originalDocLens: [Int] = try IndexStorage.readJSON(from: docLensURL)
            var keepDocLens: [Int] = []
            keepDocLens.reserveCapacity(originalDocLens.count)

            var embeddingMask: [Bool] = []
            embeddingMask.reserveCapacity(originalDocLens.reduce(0, +))

            for length in originalDocLens {
                let keep = !deleteSet.contains(globalDocId)
                globalDocId += 1
                if keep {
                    keepDocLens.append(length)
                }
                if length > 0 {
                    embeddingMask.append(contentsOf: Array(repeating: keep, count: length))
                }
            }

            if keepDocLens.count != originalDocLens.count {
                anyRemoved = true
            }

            let codesURL = paths.codes(for: chunkIndex)
            let residualURL = paths.residuals(for: chunkIndex)

            let codes = try IndexStorage.readInt32Array(from: codesURL)
            guard codes.count == embeddingMask.count else {
                throw PlaidError.invalidEmbeddingDimensions(
                    expected: embeddingMask.count,
                    actual: codes.count
                )
            }

            let residualData = try Data(contentsOf: residualURL)
            guard residualData.count == codes.count * bytesPerResidual else {
                throw PlaidError.invalidEmbeddingDimensions(
                    expected: codes.count * bytesPerResidual,
                    actual: residualData.count
                )
            }
            let residualBytes = [UInt8](residualData)

            var filteredCodes: [Int32] = []
            filteredCodes.reserveCapacity(embeddingMask.filter { $0 }.count)
            var filteredResiduals: [UInt8] = []
            filteredResiduals.reserveCapacity(filteredCodes.capacity * bytesPerResidual)

            for (idx, keep) in embeddingMask.enumerated() {
                if keep {
                    filteredCodes.append(codes[idx])
                    let byteStart = idx * bytesPerResidual
                    let byteEnd = byteStart + bytesPerResidual

                    // Bounds validation for residual byte slicing
                    guard byteEnd <= residualBytes.count else {
                        throw PlaidError.invalidEmbeddingDimensions(
                            expected: byteEnd,
                            actual: residualBytes.count
                        )
                    }

                    filteredResiduals.append(contentsOf: residualBytes[byteStart ..< byteEnd])
                }
            }

            updatedDocLengths.append(contentsOf: keepDocLens)
            updatedCodes.append(contentsOf: filteredCodes)

            try BinaryIO.writeInt32(filteredCodes, to: codesURL)
            try BinaryIO.writeUInt8(filteredResiduals, to: residualURL)
            try builder.writeDocLengths(keepDocLens, to: docLensURL)

            let metadata = IndexBuilder.ChunkMetadata(
                num_passages: keepDocLens.count,
                num_embeddings: filteredCodes.count,
                embedding_offset: embeddingOffset
            )
            try builder.writeJSON(metadata, to: paths.chunkMetadata(for: chunkIndex))
            embeddingOffset += filteredCodes.count
        }

        guard globalDocId == summary.docLengths.count else {
            throw PlaidError.invalidSubset("Document count mismatch during delete")
        }

        guard anyRemoved else {
            return RemovalResult(
                updatedDocVectors: summary.docVectors,
                updatedDocLengths: summary.docLengths,
                didModify: false
            )
        }

        let (ivf, ivfLengths) = try builder.buildIVF(
            codes: updatedCodes,
            docLengths: updatedDocLengths,
            codec: codec
        )
        try BinaryIO.writeInt32(ivf, to: paths.ivf())
        try BinaryIO.writeInt32(ivfLengths, to: paths.ivfLengths())

        try builder.writePlan(
            paths: paths,
            numChunks: numChunks,
            centroidCount: codec.numCentroids
        )

        let totalEmbeddings = updatedCodes.count
        let totalDocuments = updatedDocLengths.count
        let avgDocLen =
            totalDocuments > 0
            ? Double(totalEmbeddings) / Double(totalDocuments)
            : 0.0

        let finalMetadata = IndexBuilder.FinalMetadata(
            num_chunks: numChunks,
            nbits: artifacts.nbits,
            num_partitions: artifacts.numPartitions,
            num_embeddings: totalEmbeddings,
            avg_doclen: avgDocLen,
            embedding_dim: artifacts.embeddingDim,
            total_documents: totalDocuments,
            bytes_per_residual: artifacts.bytesPerResidual
        )
        try builder.writeJSON(finalMetadata, to: paths.metadata())

        var updatedDocVectors: [[Float]] = []
        updatedDocVectors.reserveCapacity(totalDocuments)
        for (idx, vector) in summary.docVectors.enumerated() {
            if !deleteSet.contains(idx) {
                updatedDocVectors.append(vector)
            }
        }

        return RemovalResult(
            updatedDocVectors: updatedDocVectors,
            updatedDocLengths: updatedDocLengths,
            didModify: true
        )
    }

    fileprivate static func runLegacySearch(
        index: PlaidIndex,
        queries: [[[Float]]],
        params: SearchParameters,
        subset: [[Int]]?
    ) throws -> [QueryResult] {
        var results: [QueryResult] = []
        results.reserveCapacity(queries.count)

        for (queryIdx, rawQuery) in queries.enumerated() {
            let queryVector = try VectorMath.averageAndNormalize(
                rawQuery, embeddingDim: index.embeddingDim)
            let candidateIds: [Int]
            if let subsetList = subset {
                guard queryIdx < subsetList.count else {
                    throw PlaidError.invalidSubset("Missing subset entry for query \(queryIdx)")
                }
                candidateIds = subsetList[queryIdx]
                    .filter { $0 >= 0 && $0 < index.docVectors.count }
            } else {
                candidateIds = Array(0 ..< index.docVectors.count)
            }

            guard !candidateIds.isEmpty else {
                results.append(QueryResult(queryId: queryIdx, passageIds: [], scores: []))
                continue
            }

            var scored: [(Int, Float)] = []
            scored.reserveCapacity(candidateIds.count)
            for docId in candidateIds {
                let docVector = index.docVectors[docId]
                let score = VectorMath.dot(queryVector, docVector)
                scored.append((docId, score))
            }

            scored.sort { $0.1 > $1.1 }
            let topK = min(params.topK, scored.count)
            let topResults = scored.prefix(topK)
            let passageIds = topResults.map { $0.0 }
            let scores = topResults.map { $0.1 }
            results.append(QueryResult(queryId: queryIdx, passageIds: passageIds, scores: scores))
        }

        return results
    }

    fileprivate static func runSearch(
        artifacts: IndexArtifacts,
        summary: PlaidIndex,
        queries: [[[Float]]],
        params: SearchParameters,
        subset: [[Int]]?,
        cachedArtifacts: CachedSearchArtifacts
    ) throws -> [QueryResult] {
        var scalarDecompressTime: Double = 0
        var scalarDocs = 0
        var vectorDecompressTime: Double = 0
        var vectorDocs = 0

        // Use cached values instead of extracting from tensors
        let centroidValues = cachedArtifacts.centroidValues
        let centroidCount = cachedArtifacts.centroidCount
        let embeddingDim = cachedArtifacts.embeddingDim
        let bucketWeightsArray = cachedArtifacts.bucketWeightsArray
        let byteMapArray = cachedArtifacts.byteMapArray
        let bucketLookupArray = cachedArtifacts.bucketLookupArray
        let codecNBits = cachedArtifacts.codecNBits
        let numBuckets = cachedArtifacts.numBuckets
        let keysPerByte = cachedArtifacts.keysPerByte
        let bucketWeightsShape = cachedArtifacts.bucketWeightsShape

        let embeddingOffsets = artifacts.embeddingOffsets
        let docLengths = artifacts.docLengths
        let bytesPerResidual = artifacts.bytesPerResidual
        let residualCount = artifacts.residuals.count
        let codes = artifacts.codes

        var results: [QueryResult] = []
        results.reserveCapacity(queries.count)

        // Precompute centroid offsets within the flattened IVF list.
        var offsets: [Int] = Array(repeating: 0, count: centroidCount + 1)
        for i in 0 ..< centroidCount {
            offsets[i + 1] = offsets[i] + Int(artifacts.ivfLengths[i])
        }

        for (queryIdx, rawQuery) in queries.enumerated() {
            let normalizedQueryTokens = rawQuery
            let queryTokensMLX: MLXArray? = {
                guard !normalizedQueryTokens.isEmpty else { return nil }
                let flat = normalizedQueryTokens.flatMap { token -> [Float32] in
                    token.map { Float32($0) }
                }
                return MLXArray(flat, [normalizedQueryTokens.count, embeddingDim]).asType(
                    .float32, stream: .default)
            }()

            let subsetForQuery: [Int]? = subset.flatMap { set in
                guard queryIdx < set.count else { return nil }
                return set[queryIdx]
            }

            let selectedDocIds = selectCandidates(
                queryTokens: normalizedQueryTokens,
                centroidValues: centroidValues,
                centroidCount: centroidCount,
                embeddingDim: embeddingDim,
                artifacts: artifacts,
                offsets: offsets,
                nProbe: max(1, min(params.nIvfProbe, centroidCount)),
                subset: subsetForQuery
            )

            if selectedDocIds.isEmpty {
                results.append(QueryResult(queryId: queryIdx, passageIds: [], scores: []))
                continue
            }

            var exactScores: [(Int, Float)] = []
            exactScores.reserveCapacity(selectedDocIds.count)

            let useVectorized =
                bucketWeightsShape.count == 2
                && bucketWeightsShape[0] == numBuckets
                && bucketWeightsShape[1] == embeddingDim
                && bucketLookupArray != nil

            artifacts.residuals.withUnsafeBytes { rawBuffer in
                let residualBytes = rawBuffer.bindMemory(to: UInt8.self)
                guard let baseAddress = residualBytes.baseAddress else { return }
                for docId in selectedDocIds {
                    guard docId >= 0, docId < docLengths.count else { continue }
                    let docStart = embeddingOffsets[docId]
                    let docEnd = embeddingOffsets[docId + 1]
                    let docLen = docEnd - docStart
                    guard docLen > 0 else { continue }

                    let docTokens: MLXArray?
                    var usedVectorBranch = false
                    let start = DispatchTime.now()
                    if useVectorized {
                        usedVectorBranch = true
                        docTokens = decompressDocumentVectorized(
                            docStart: docStart,
                            docLen: docLen,
                            artifacts: artifacts,
                            residualBytes: baseAddress,
                            stream: .default
                        )
                    } else {
                        docTokens = decompressDocumentScalar(
                            docStart: docStart,
                            docLen: docLen,
                            embeddingDim: embeddingDim,
                            nbits: codecNBits,
                            codes: codes,
                            centroidValues: centroidValues,
                            bucketWeights: bucketWeightsArray,
                            residualBytes: baseAddress,
                            bytesPerResidual: bytesPerResidual,
                            numBuckets: numBuckets,
                            residualCount: residualCount,
                            keysPerByte: keysPerByte,
                            byteMap: byteMapArray,
                            bucketLookup: bucketLookupArray
                        )
                    }
                    let elapsed =
                        Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds)
                        / 1_000_000_000

                    guard let docTokens else { continue }
                    if usedVectorBranch {
                        vectorDecompressTime += elapsed
                        vectorDocs += 1
                    } else {
                        scalarDecompressTime += elapsed
                        scalarDocs += 1
                    }
                    if let queryTokensMLX {
                        let score = colbertScore(queryTokens: queryTokensMLX, docTokens: docTokens)
                        exactScores.append((docId, score))
                    }
                }
            }

            exactScores.sort { $0.1 > $1.1 }

            let finalCount = min(params.topK, exactScores.count)
            let top = exactScores.prefix(finalCount)
            let ids = top.map { $0.0 }
            let scores = top.map { $0.1 }
            results.append(QueryResult(queryId: queryIdx, passageIds: ids, scores: scores))
        }

        if params.logTiming {
            let vectorAverage = vectorDocs > 0 ? vectorDecompressTime / Double(vectorDocs) : 0
            let scalarAverage = scalarDocs > 0 ? scalarDecompressTime / Double(scalarDocs) : 0
            print(
                String(
                    format:
                        "[Timing] decompression avg latency → vector: %.6fs (%d docs), scalar: %.6fs (%d docs)",
                    vectorAverage,
                    vectorDocs,
                    scalarAverage,
                    scalarDocs
                ))
        }

        return results
    }

}

extension Plaid {
    fileprivate static func selectCandidates(
        queryTokens: [[Float]],
        centroidValues: [Float32],
        centroidCount: Int,
        embeddingDim: Int,
        artifacts: IndexArtifacts,
        offsets: [Int],
        nProbe: Int,
        subset: [Int]?
    ) -> [Int] {
        if centroidCount == 0 {
            return subset ?? []
        }

        let probeCount = max(1, min(nProbe, centroidCount))
        var selectedCentroidIndices = Set<Int>()
        if queryTokens.isEmpty {
            selectedCentroidIndices.formUnion(0 ..< centroidCount)
        } else {
            for token in queryTokens {
                guard token.count == embeddingDim else { continue }
                var scored: [(Int, Float)] = []
                scored.reserveCapacity(centroidCount)
                for centroidIdx in 0 ..< centroidCount {
                    let start = centroidIdx * embeddingDim
                    var score: Float = 0
                    for dim in 0 ..< embeddingDim {
                        score += Float(centroidValues[start + dim]) * token[dim]
                    }
                    scored.append((centroidIdx, score))
                }
                scored.sort { $0.1 > $1.1 }
                for idx in 0 ..< min(probeCount, scored.count) {
                    selectedCentroidIndices.insert(scored[idx].0)
                }
            }
        }

        if selectedCentroidIndices.isEmpty {
            selectedCentroidIndices.formUnion(0 ..< centroidCount)
        }

        var candidates = Set<Int>()
        candidates.reserveCapacity(artifacts.ivfLists.count / max(1, centroidCount))

        for centroidIdx in selectedCentroidIndices {
            let start = offsets[centroidIdx]
            let end = offsets[centroidIdx + 1]
            guard start < end else { continue }
            for idx in start ..< end {
                let docId = Int(artifacts.ivfLists[idx])
                candidates.insert(docId)
            }
        }

        var docIds = Array(candidates)
        if let subsetList = subset {
            let allowed = Set(subsetList)
            docIds = docIds.filter { allowed.contains($0) }
        }

        return docIds
    }

    fileprivate static func decompressDocumentVectorized(
        docStart: Int,
        docLen: Int,
        artifacts: IndexArtifacts,
        residualBytes: UnsafePointer<UInt8>,
        stream: StreamOrDevice = .default
    ) -> MLXArray? {
        guard docLen > 0 else { return nil }
        let embeddingDim = artifacts.embeddingDim
        guard embeddingDim > 0 else { return nil }
        let nbits = artifacts.nbits
        guard nbits > 0, nbits <= 8 else { return nil }
        let keysPerByte = 8 / nbits
        guard keysPerByte > 0 else { return nil }
        guard let bucketWeights = artifacts.bucketWeights,
            let bucketLookupRaw = artifacts.codecArtifacts.bucketWeightLookup
        else {
            return nil
        }

        let bytesPerResidual = artifacts.bytesPerResidual
        guard bytesPerResidual > 0 else { return nil }

        let residualBase = docStart * bytesPerResidual
        let residualEnd = residualBase + docLen * bytesPerResidual
        guard residualEnd <= artifacts.residuals.count else { return nil }

        let residualPointer = residualBytes + residualBase
        let residualBuffer = UnsafeBufferPointer(
            start: residualPointer, count: docLen * bytesPerResidual)
        let residualArray = Array(residualBuffer)
        guard !residualArray.isEmpty else { return nil }

        let residualTensor = MLXArray(
            residualArray.map { Float32($0) },
            [docLen, bytesPerResidual]
        ).asType(.int32, stream: stream)

        guard docStart >= 0, docStart + docLen <= artifacts.codes.count else {
            return nil
        }
        let codesSlice = Array(artifacts.codes[docStart ..< docStart + docLen])
        let codesTensor = MLXArray(
            codesSlice.map { Float32($0) },
            [docLen]
        ).asType(.int32, stream: stream)
        let centroidsGathered = artifacts.centroids.take(codesTensor, axis: 0, stream: stream)
        eval(centroidsGathered)  // Checkpoint: Materialize centroid lookup

        let byteMap = artifacts.codecArtifacts.byteReversedBitsMap.asType(.int32, stream: stream)
        let flatResidualIndices = residualTensor.flattened(stream: stream)
        let reversedFlat = byteMap.take(flatResidualIndices, stream: stream).asType(
            .int32, stream: stream)
        let packedDim = bytesPerResidual
        let reversedCodes = reversedFlat.reshaped([docLen, packedDim], stream: stream)
        eval(reversedCodes)  // Checkpoint: Materialize byte reversal operations
        let flatCombinationIndices = reversedCodes.flattened(stream: stream)

        let lookupSize = bucketLookupRaw.shape.reduce(1, *)
        guard lookupSize % keysPerByte == 0 else { return nil }
        let combos = lookupSize / keysPerByte
        guard lookupSize > 0 else { return nil }
        let bucketLookup =
            bucketLookupRaw
            .asType(.int32, stream: stream)
            .reshaped([combos, keysPerByte], stream: stream)
        let selectedBucketsFlat = bucketLookup.take(flatCombinationIndices, axis: 0, stream: stream)
        let selectedBuckets = selectedBucketsFlat.reshaped(
            [docLen, packedDim, keysPerByte],
            stream: stream
        )
        eval(selectedBuckets)  // Checkpoint: Materialize bucket lookup

        let dimensionSequence = (0 ..< embeddingDim).map { Float32($0) }
        let dimensionBase = MLXArray(dimensionSequence, [embeddingDim])
            .asType(.int32, stream: stream)
            .reshaped([packedDim, keysPerByte], stream: stream)
        let dimensionIndices = broadcast(
            dimensionBase,
            to: [docLen, packedDim, keysPerByte],
            stream: stream
        )

        let weightIndices =
            (selectedBuckets.asType(.float32, stream: stream) * Float32(embeddingDim)
            + dimensionIndices.asType(.float32, stream: stream)).asType(.int32, stream: stream)
        let weightIndicesFlat = weightIndices.flattened(stream: stream)

        let bucketWeightsFlat = bucketWeights.flattened(stream: stream)
        let residualWeightsFlat = bucketWeightsFlat.take(weightIndicesFlat, stream: stream)
        let residualWeights = residualWeightsFlat.reshaped(
            [docLen, packedDim, keysPerByte],
            stream: stream
        )
        let residualMatrix = residualWeights.reshaped([docLen, embeddingDim], stream: stream)
        eval(residualMatrix)  // Checkpoint: Materialize residual weight computation

        let docMatrix = (centroidsGathered + residualMatrix).asType(.float32, stream: stream)
        eval(docMatrix)  // Checkpoint: Materialize final matrix before extraction
        let docValues = docMatrix.asArray(Float32.self)
        var normalized: [Float32] = []
        normalized.reserveCapacity(docLen * embeddingDim)
        for row in 0 ..< docLen {
            let start = row * embeddingDim
            let end = start + embeddingDim
            var rowValues: [Float] = []
            rowValues.reserveCapacity(embeddingDim)
            for value in docValues[start ..< end] {
                rowValues.append(Float(value))
            }
            let normed = VectorMath.normalize(rowValues)
            normalized.append(contentsOf: normed.map { Float32($0) })
        }

        guard !normalized.isEmpty else { return nil }
        return MLXArray(normalized, [docLen, embeddingDim]).asType(.float32, stream: stream)
    }

    fileprivate static func decompressDocumentScalar(
        docStart: Int,
        docLen: Int,
        embeddingDim: Int,
        nbits: Int,
        codes: [Int32],
        centroidValues: [Float],
        bucketWeights: [Float]?,
        residualBytes: UnsafePointer<UInt8>,
        bytesPerResidual: Int,
        numBuckets: Int,
        residualCount: Int,
        keysPerByte: Int,
        byteMap: [Int],
        bucketLookup: [Int]?
    ) -> MLXArray? {
        if docLen <= 0 || embeddingDim <= 0 {
            return nil
        }

        let hasResiduals = bucketWeights != nil && numBuckets > 0 && bytesPerResidual > 0
        var tokens: [Float] = []
        tokens.reserveCapacity(docLen * embeddingDim)
        var tokenCount = 0

        let packedDim = max(1, embeddingDim * nbits / 8)
        let keys = max(1, keysPerByte)
        let mask = (1 << nbits) - 1
        let bucketCount = numBuckets

        for localIndex in 0 ..< docLen {
            let globalIndex = docStart + localIndex
            guard globalIndex < codes.count else { break }

            let code = Int(codes[globalIndex])
            let centroidBase = code * embeddingDim
            guard centroidBase + embeddingDim <= centroidValues.count else { continue }

            var vector = [Float](repeating: 0, count: embeddingDim)
            for dim in 0 ..< embeddingDim {
                vector[dim] = centroidValues[centroidBase + dim]
            }

            if hasResiduals, let weights = bucketWeights {
                let residualBase = globalIndex * bytesPerResidual
                let residualEnd = residualBase + bytesPerResidual
                guard residualEnd <= residualCount else { break }

                for packedIndex in 0 ..< packedDim {
                    let dimBase = packedIndex * keys
                    if dimBase >= embeddingDim { break }
                    let byteValue = Int(residualBytes[residualBase + packedIndex]) & 0xFF

                    var buckets = [Int](repeating: 0, count: keys)
                    if let lookup = bucketLookup,
                        !lookup.isEmpty,
                        byteValue < byteMap.count
                    {
                        let combinationIndex = byteMap[byteValue]
                        let rowStart = combinationIndex * keys
                        if rowStart + keys <= lookup.count {
                            for offset in 0 ..< keys {
                                buckets[offset] = lookup[rowStart + offset]
                            }
                        } else {
                            for offset in 0 ..< keys {
                                buckets[offset] = 0
                            }
                        }
                    } else {
                        for offset in 0 ..< keys {
                            let shift = (keys - 1 - offset) * nbits
                            buckets[offset] = (byteValue >> shift) & mask
                        }
                    }

                    for offset in 0 ..< keys {
                        let dimIndex = dimBase + offset
                        if dimIndex >= embeddingDim { break }
                        var bucketIndex = buckets[offset]
                        if bucketCount > 0 {
                            bucketIndex = bucketIndex % bucketCount
                        }
                        guard bucketIndex >= 0 else { continue }
                        if weights.count == bucketCount * embeddingDim {
                            let weightIndex = bucketIndex * embeddingDim + dimIndex
                            if weightIndex < weights.count {
                                vector[dimIndex] += weights[weightIndex]
                            }
                        } else if weights.count == bucketCount {
                            let weightIndex = bucketIndex
                            if weightIndex < weights.count {
                                vector[dimIndex] += weights[weightIndex]
                            }
                        }
                    }
                }
            }

            vector = VectorMath.normalize(vector)
            tokens.append(contentsOf: vector)
            tokenCount += 1
        }

        guard tokenCount > 0 else { return nil }
        let flat = tokens.map { Float32($0) }
        return MLXArray(flat, [tokenCount, embeddingDim]).asType(.float32, stream: .default)
    }

    fileprivate static func colbertScore(queryTokens: MLXArray, docTokens: MLXArray) -> Float {
        guard queryTokens.shape.count == 2,
            docTokens.shape.count == 2,
            queryTokens.shape[1] == docTokens.shape[1],
            queryTokens.shape[0] > 0,
            docTokens.shape[0] > 0
        else {
            return 0
        }

        let sims = docTokens.matmul(queryTokens.transposed(1, 0))
        let maxSim = sims.max(axis: 0, stream: .default)
        let values = maxSim.asArray(Float32.self)
        return values.reduce(0, +)
    }
}
