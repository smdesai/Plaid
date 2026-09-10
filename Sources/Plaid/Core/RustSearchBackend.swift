import Foundation
import NextPlaidBindings

/// `SearchBackend` over the Rust `next-plaid` engine (via UniFFI).
///
/// Two seam responsibilities live here:
///  1. **Normalization** — the CoreML encoder emits raw vectors; the engine
///     expects unit-L2 rows and normalizes nothing, so every token row is
///     normalized here before crossing the FFI.
///  2. **Packing** — `[[Float]]` token matrices become `EmbeddingMatrix`
///     (little-endian, row-major `f32`), matching the Rust decode.
///
/// Live `PlaidIndex` handles are cached per index path. `add`/`remove` mutate
/// the cached handle in place (the Rust side reloads under its own lock), so the
/// cache stays valid across those calls; `create` replaces the handle.
public final class RustSearchBackend: SearchBackend {
    /// Engine defaults for parameters the Swift `SearchParameters` struct omits.
    private static let defaultCentroidBatchSize: UInt64 = 100_000
    private static let defaultCentroidScoreThreshold: Float = 0.4

    private var handles: [String: PlaidIndex] = [:]
    private let lock = NSLock()

    public init() {}

    // MARK: - Handle cache

    private func key(_ url: URL) -> String { url.standardizedFileURL.path }

    private func cachedHandle(_ url: URL) -> PlaidIndex? {
        lock.lock()
        defer { lock.unlock() }
        return handles[key(url)]
    }

    private func storeHandle(_ index: PlaidIndex, for url: URL) {
        lock.lock()
        defer { lock.unlock() }
        handles[key(url)] = index
    }

    /// Cached handle, or open the on-disk index and cache it.
    private func handle(for url: URL) throws -> PlaidIndex {
        if let cached = cachedHandle(url) { return cached }
        let opened = try PlaidIndex.open(path: key(url))
        storeHandle(opened, for: url)
        return opened
    }

    // MARK: - Packing / normalization

    /// Normalize each token row (unit L2) and pack as little-endian `f32`.
    private func matrix(from tokens: [[Float]]) -> EmbeddingMatrix {
        let rows = tokens.count
        let cols = rows > 0 ? tokens[0].count : 0
        var floats = [Float32]()
        floats.reserveCapacity(rows * cols)
        for token in tokens {
            for value in VectorMath.normalize(token) {
                floats.append(Float32(value))
            }
        }
        let data = floats.withUnsafeBufferPointer { Data(buffer: $0) }
        return EmbeddingMatrix(data: data, rows: UInt32(rows), cols: UInt32(cols))
    }

    /// Unpack an `EmbeddingMatrix` (little-endian `f32`, row-major) to `[[Float]]`.
    private func tokens(from matrix: EmbeddingMatrix) -> [[Float]] {
        let rows = Int(matrix.rows)
        let cols = Int(matrix.cols)
        guard rows > 0, cols > 0 else { return [] }
        let floats: [Float32] = matrix.data.withUnsafeBytes { raw in
            Array(raw.bindMemory(to: Float32.self))
        }
        guard floats.count >= rows * cols else { return [] }
        var out = [[Float]]()
        out.reserveCapacity(rows)
        for row in 0 ..< rows {
            let start = row * cols
            out.append(floats[start ..< start + cols].map { Float($0) })
        }
        return out
    }

    // MARK: - SearchBackend

    public func create(
        indexURL: URL,
        embeddingDim: Int,
        nbits: Int,
        embeddings: [[[Float]]],
        centroids: [[Float]],
        batchSize: Int,
        seed: UInt64?
    ) throws {
        // `centroids` is ignored: the Rust engine computes its own k-means.
        guard !embeddings.isEmpty else { throw PlaidError.emptyEmbeddingSet }
        let matrices = embeddings.map { matrix(from: $0) }
        let config = FfiIndexConfig(
            nbits: UInt64(nbits),
            batchSize: UInt64(batchSize),
            seed: seed,
            kmeansNiters: 4,
            maxPointsPerCentroid: 256,
            nSamplesKmeans: nil,
            startFromScratch: 999,
            forceCpu: false,
            binary: false
        )
        let index = try PlaidIndex.create(path: key(indexURL), embeddings: matrices, config: config)
        storeHandle(index, for: indexURL)
    }

    @discardableResult
    public func update(
        indexURL: URL,
        embeddings: [[[Float]]],
        batchSize: Int
    ) throws -> [Int] {
        guard !embeddings.isEmpty else { return [] }
        let index = try handle(for: indexURL)
        let matrices = embeddings.map { matrix(from: $0) }
        let config = FfiUpdateConfig(
            batchSize: UInt64(batchSize),
            kmeansNiters: 4,
            maxPointsPerCentroid: 256,
            nSamplesKmeans: nil,
            seed: 42,
            startFromScratch: 999,
            bufferSize: 100,
            forceCpu: false
        )
        let ids = try index.add(embeddings: matrices, config: config)
        return ids.map { Int($0) }
    }

    public func loadAndSearch(
        indexURL: URL,
        queries: [[[Float]]],
        searchParameters: SearchParameters,
        showProgress: Bool,
        preloadIndex: Bool,
        subset: [[Int]]?
    ) throws -> [QueryResult] {
        let index = try handle(for: indexURL)
        let params = FfiSearchParameters(
            batchSize: UInt64(searchParameters.batchSize),
            nFullScores: UInt64(searchParameters.nFullScores),
            topK: UInt64(searchParameters.topK),
            nIvfProbe: UInt64(searchParameters.nIvfProbe),
            centroidBatchSize: Self.defaultCentroidBatchSize,
            centroidScoreThreshold: Self.defaultCentroidScoreThreshold
        )

        var results = [QueryResult]()
        results.reserveCapacity(queries.count)
        // The engine takes one subset per call, so run queries individually to
        // honor the per-query subset the Swift seam allows.
        for (queryIdx, rawQuery) in queries.enumerated() {
            let query = matrix(from: rawQuery)
            let subsetForQuery: [Int64]? = subset.flatMap { list in
                guard queryIdx < list.count else { return nil }
                return list[queryIdx].map { Int64($0) }
            }
            let hits = try index.search(query: query, params: params, subset: subsetForQuery)
            results.append(
                QueryResult(
                    queryId: queryIdx,
                    passageIds: hits.map { Int($0.docId) },
                    scores: hits.map { $0.score }
                )
            )
        }
        return results
    }

    @discardableResult
    public func delete(
        indexURL: URL,
        subset: [Int]
    ) throws -> DeleteOutcome {
        guard !subset.isEmpty else { return DeleteOutcome(deletedIdsSorted: []) }
        let index = try handle(for: indexURL)
        let outcome = try index.remove(ids: subset.map { Int64($0) })
        return DeleteOutcome(deletedIdsSorted: outcome.deletedIdsSorted.map { Int($0) })
    }

    public func getDocumentEmbeddings(
        indexURL: URL,
        documentId: Int
    ) throws -> [[Float]] {
        let index = try handle(for: indexURL)
        let matrices = try index.reconstruct(ids: [Int64(documentId)])
        guard let first = matrices.first else { return [] }
        return tokens(from: first)
    }
}
