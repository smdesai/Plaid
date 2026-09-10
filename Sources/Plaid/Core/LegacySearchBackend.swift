import Foundation

/// `SearchBackend` over the pure-Swift `Plaid` engine. Retained as a fallback
/// and as the parity oracle for `RustSearchBackend` (M4).
public struct LegacySearchBackend: SearchBackend {
    public init() {}

    public func create(
        indexURL: URL,
        embeddingDim: Int,
        nbits: Int,
        embeddings: [[[Float]]],
        centroids: [[Float]],
        batchSize: Int,
        seed: UInt64?
    ) throws {
        try Plaid.create(
            indexURL: indexURL,
            embeddingDim: embeddingDim,
            nbits: nbits,
            embeddings: embeddings,
            centroids: centroids,
            batchSize: batchSize,
            seed: seed
        )
    }

    @discardableResult
    public func update(
        indexURL: URL,
        embeddings: [[[Float]]],
        batchSize: Int
    ) throws -> [Int] {
        // Legacy append is positional: new docs take the ids [before, after).
        let before = (try? Plaid.documentCount(indexURL: indexURL)) ?? 0
        try Plaid.update(indexURL: indexURL, embeddings: embeddings, batchSize: batchSize)
        let after = (try? Plaid.documentCount(indexURL: indexURL)) ?? before
        guard after > before else { return [] }
        return Array(before ..< after)
    }

    public func loadAndSearch(
        indexURL: URL,
        queries: [[[Float]]],
        searchParameters: SearchParameters,
        showProgress: Bool,
        preloadIndex: Bool,
        subset: [[Int]]?
    ) throws -> [QueryResult] {
        try Plaid.loadAndSearch(
            indexURL: indexURL,
            queries: queries,
            searchParameters: searchParameters,
            showProgress: showProgress,
            preloadIndex: preloadIndex,
            subset: subset
        )
    }

    @discardableResult
    public func delete(
        indexURL: URL,
        subset: [Int]
    ) throws -> DeleteOutcome {
        // Sanitize the same way the Rust engine does (in-range, deduped, sorted)
        // so both backends report an identical removed set for the remap.
        let count = (try? Plaid.documentCount(indexURL: indexURL)) ?? 0
        let sanitized = Array(Set(subset.filter { $0 >= 0 && $0 < count })).sorted()
        try Plaid.delete(indexURL: indexURL, subset: sanitized)
        return DeleteOutcome(deletedIdsSorted: sanitized)
    }

    public func getDocumentEmbeddings(
        indexURL: URL,
        documentId: Int
    ) throws -> [[Float]] {
        try Plaid.getDocumentEmbeddings(indexURL: indexURL, documentId: documentId)
    }
}
