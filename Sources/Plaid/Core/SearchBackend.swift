import Foundation

/// Outcome of a delete: the internal ids the engine removed, ascending.
///
/// Both engines compact survivors after a delete, renumbering
/// `new = old − count(deletedIdsSorted < old)`. Callers that keep an external
/// id map replay that transform using `deletedIdsSorted`.
public struct DeleteOutcome: Sendable {
    public let deletedIdsSorted: [Int]
    public init(deletedIdsSorted: [Int]) {
        self.deletedIdsSorted = deletedIdsSorted
    }
}

/// The vector-engine seam, capturing today's `[[[Float]]]` → `[QueryResult]`
/// contract so the pure-Swift `Plaid` engine and the Rust `next-plaid` engine
/// are interchangeable.
///
/// Embeddings are `[[[Float]]]` = documents × tokens × dim, **raw/unnormalized**
/// as produced by the CoreML encoder. Each implementation normalizes as its
/// engine requires (the Rust engine expects unit-L2 rows and normalizes nothing).
public protocol SearchBackend {
    /// Build a fresh index at `indexURL` from `embeddings`.
    /// `centroids` are only used by the legacy engine; the Rust engine computes
    /// its own and ignores the argument.
    func create(
        indexURL: URL,
        embeddingDim: Int,
        nbits: Int,
        embeddings: [[[Float]]],
        centroids: [[Float]],
        batchSize: Int,
        seed: UInt64?
    ) throws

    /// Append documents; returns the ids assigned to them.
    @discardableResult
    func update(
        indexURL: URL,
        embeddings: [[[Float]]],
        batchSize: Int
    ) throws -> [Int]

    func loadAndSearch(
        indexURL: URL,
        queries: [[[Float]]],
        searchParameters: SearchParameters,
        showProgress: Bool,
        preloadIndex: Bool,
        subset: [[Int]]?
    ) throws -> [QueryResult]

    /// Delete documents by internal id; returns the sanitized removed set for
    /// the caller's id remap.
    @discardableResult
    func delete(
        indexURL: URL,
        subset: [Int]
    ) throws -> DeleteOutcome

    func getDocumentEmbeddings(
        indexURL: URL,
        documentId: Int
    ) throws -> [[Float]]
}
