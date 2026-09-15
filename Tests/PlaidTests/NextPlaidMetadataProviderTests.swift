import Foundation
import XCTest

@testable import Plaid

/// Exercises `NextPlaidMetadataProvider` against the real SQLite `metadata.db`
/// the Rust engine writes inside an index directory. Proves per-chunk text is
/// tied to its vector `doc_id` and follows the engine's delete-renumbering, so
/// a search hit always resolves to the correct text.
///
/// The metadata store lives beside the vectors, so these tests drive both the
/// `NextPlaidBackend` (vectors) and the provider (text) through the same index
/// directory — exactly as the app does.
final class NextPlaidMetadataProviderTests: XCTestCase {
    private let dim = 64
    private let indexName = "default"

    private func oneHot(axis: Int, tokens: Int = 4) -> [[Float]] {
        var row = [Float](repeating: 0, count: dim)
        row[axis] = 1
        return Array(repeating: row, count: tokens)
    }

    private func tempDir() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("rust_meta_\(UUID().uuidString)", isDirectory: true)
    }

    func testStoreGetRoundTripAndDeleteRenumberKeepsTextTied() async throws {
        let backend = NextPlaidBackend()
        let dir = tempDir()
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let provider = NextPlaidMetadataProvider(indexURL: dir)

        // Build vectors along axes 0..2 so a query on an axis maps to a known id.
        try backend.create(
            indexURL: dir,
            embeddingDim: dim,
            nbits: 2,
            embeddings: [oneHot(axis: 0), oneHot(axis: 1), oneHot(axis: 2)],
            batchSize: 50_000,
            seed: 42
        )

        // Store text tied to each doc id.
        try await provider.registerDocuments(
            [
                (
                    plaidDocId: 0, documentName: "a.txt", chunkText: "alpha text", chunkIndex: 0,
                    embeddingCount: 10, filePath: nil
                ),
                (
                    plaidDocId: 1, documentName: "b.txt", chunkText: "bravo text", chunkIndex: 1,
                    embeddingCount: 20, filePath: "/tmp/b.txt"
                ),
                (
                    plaidDocId: 2, documentName: "c.txt", chunkText: "charlie text", chunkIndex: 2,
                    embeddingCount: 30, filePath: nil
                ),
            ],
            indexName: indexName
        )

        let countAfterStore = try await provider.documentCount(indexName: indexName)
        XCTAssertEqual(countAfterStore, 3)

        // Round-trip: order preserved, fields intact, nil filePath stays nil.
        let docs = try await provider.getDocuments(plaidDocIds: [2, 0, 1], indexName: indexName)
        XCTAssertEqual(docs.map { $0.plaidDocId }, [2, 0, 1])
        XCTAssertEqual(docs.map { $0.chunkText }, ["charlie text", "alpha text", "bravo text"])
        XCTAssertEqual(docs.map { $0.documentName }, ["c.txt", "a.txt", "b.txt"])
        XCTAssertEqual(docs.map { $0.chunkIndex }, [2, 0, 1])
        XCTAssertEqual(docs.map { $0.embeddingCount }, [30, 10, 20])
        XCTAssertNil(docs.first { $0.plaidDocId == 0 }?.filePath)
        XCTAssertEqual(docs.first { $0.plaidDocId == 1 }?.filePath, "/tmp/b.txt")

        // Delete the middle doc (id 1). The engine compacts survivors: old id 2
        // becomes new id 1 for BOTH the vectors and the co-located metadata.
        let outcome = try backend.delete(indexURL: dir, subset: [1])
        XCTAssertEqual(outcome.deletedIdsSorted, [1])
        let countAfterDelete = try await provider.documentCount(indexName: indexName)
        XCTAssertEqual(countAfterDelete, 2)

        // Text must have followed the renumber: id 0 -> "alpha", id 1 -> "charlie".
        let survivors = try await provider.getDocuments(plaidDocIds: [0, 1], indexName: indexName)
        XCTAssertEqual(survivors.map { $0.chunkText }, ["alpha text", "charlie text"])
        XCTAssertEqual(survivors.map { $0.documentName }, ["a.txt", "c.txt"])

        // And the vector search agrees: axis-2 content now tops out at new id 1,
        // whose stored text is "charlie text" — search result → text stays correct.
        let hits = try backend.loadAndSearch(
            indexURL: dir,
            queries: [oneHot(axis: 2, tokens: 1)],
            searchParameters: SearchParameters(
                batchSize: 2000, nFullScores: 4096, topK: 2, nIvfProbe: 1024),
            showProgress: false,
            preloadIndex: false,
            subset: nil
        )
        let topId = try XCTUnwrap(hits.first?.passageIds.first)
        XCTAssertEqual(topId, 1)
        let topDoc = try await provider.getDocument(plaidDocId: topId, indexName: indexName)
        XCTAssertEqual(topDoc?.chunkText, "charlie text")
    }

    /// Mirrors `SearchEngine.deleteDocument(named:)`: a document is many chunks
    /// sharing a `documentName`. Collect all its ids by scanning the dense id
    /// space, delete them in one pass, and confirm the *other* document's chunks
    /// survive with their text intact after renumbering.
    func testDeleteEntireDocumentByNameRemovesAllItsChunks() async throws {
        let backend = NextPlaidBackend()
        let dir = tempDir()
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let provider = NextPlaidMetadataProvider(indexURL: dir)

        // doc "A" -> chunks at ids 0,1 ; doc "B" -> chunks at ids 2,3.
        try backend.create(
            indexURL: dir,
            embeddingDim: dim,
            nbits: 2,
            embeddings: [oneHot(axis: 0), oneHot(axis: 1), oneHot(axis: 2), oneHot(axis: 3)],
            batchSize: 50_000,
            seed: 42
        )
        try await provider.registerDocuments(
            [
                (
                    plaidDocId: 0, documentName: "A", chunkText: "A0", chunkIndex: 0,
                    embeddingCount: 4, filePath: nil
                ),
                (
                    plaidDocId: 1, documentName: "A", chunkText: "A1", chunkIndex: 1,
                    embeddingCount: 4, filePath: nil
                ),
                (
                    plaidDocId: 2, documentName: "B", chunkText: "B0", chunkIndex: 0,
                    embeddingCount: 4, filePath: nil
                ),
                (
                    plaidDocId: 3, documentName: "B", chunkText: "B1", chunkIndex: 1,
                    embeddingCount: 4, filePath: nil
                ),
            ],
            indexName: indexName
        )

        // Collect every id belonging to "A" via the SQLite-pushed name filter.
        let chunksForA = try await provider.documentChunks(named: "A", indexName: indexName)
        XCTAssertEqual(chunksForA.map { $0.chunkText }, ["A0", "A1"])
        let idsForA = chunksForA.map { $0.plaidDocId }
        XCTAssertEqual(idsForA.sorted(), [0, 1])

        // A name with no chunks returns empty.
        let none = try await provider.documentChunks(named: "missing", indexName: indexName)
        XCTAssertTrue(none.isEmpty)

        // Delete the whole document in one renumbering pass.
        let outcome = try backend.delete(indexURL: dir, subset: idsForA)
        XCTAssertEqual(outcome.deletedIdsSorted, [0, 1])

        // Only "B" remains, renumbered to a dense 0,1 with text preserved.
        let remainingCount = try await provider.documentCount(indexName: indexName)
        XCTAssertEqual(remainingCount, 2)
        let survivors = try await provider.getDocuments(
            plaidDocIds: [0, 1], indexName: indexName)
        XCTAssertEqual(survivors.map { $0.documentName }, ["B", "B"])
        XCTAssertEqual(survivors.map { $0.chunkText }, ["B0", "B1"])
    }

    /// `indexedDocuments` folds the store's chunks into one entry per
    /// `documentName`, summing chunk + embedding counts, ordered by name — and
    /// stays consistent after a whole-document delete renumbers the survivors.
    func testIndexedDocumentsGroupsChunksByDocumentName() async throws {
        let backend = NextPlaidBackend()
        let dir = tempDir()
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let provider = NextPlaidMetadataProvider(indexURL: dir)

        // "Zeta" -> 1 chunk (7 embeddings); "Alpha" -> 2 chunks (3 + 5 = 8).
        try backend.create(
            indexURL: dir,
            embeddingDim: dim,
            nbits: 2,
            embeddings: [oneHot(axis: 0), oneHot(axis: 1), oneHot(axis: 2)],
            batchSize: 50_000,
            seed: 42
        )
        try await provider.registerDocuments(
            [
                (
                    plaidDocId: 0, documentName: "Zeta", chunkText: "z0", chunkIndex: 0,
                    embeddingCount: 7, filePath: "/tmp/zeta"
                ),
                (
                    plaidDocId: 1, documentName: "Alpha", chunkText: "a0", chunkIndex: 0,
                    embeddingCount: 3, filePath: nil
                ),
                (
                    plaidDocId: 2, documentName: "Alpha", chunkText: "a1", chunkIndex: 1,
                    embeddingCount: 5, filePath: nil
                ),
            ],
            indexName: indexName
        )

        // One row per document, sorted by name; counts summed per document.
        let docs = try await provider.indexedDocuments(indexName: indexName)
        XCTAssertEqual(docs.map { $0.documentName }, ["Alpha", "Zeta"])
        XCTAssertEqual(docs.map { $0.chunkCount }, [2, 1])
        XCTAssertEqual(docs.map { $0.embeddingCount }, [8, 7])
        XCTAssertEqual(docs.first { $0.documentName == "Zeta" }?.filePath, "/tmp/zeta")
        XCTAssertNil(docs.first { $0.documentName == "Alpha" }?.filePath)

        // After deleting "Alpha", only "Zeta" remains (now dense id 0).
        let alphaIds = try await provider.documentChunks(named: "Alpha", indexName: indexName)
            .map { $0.plaidDocId }
        _ = try backend.delete(indexURL: dir, subset: alphaIds)
        let remaining = try await provider.indexedDocuments(indexName: indexName)
        XCTAssertEqual(remaining.map { $0.documentName }, ["Zeta"])
        XCTAssertEqual(remaining.first?.chunkCount, 1)
        XCTAssertEqual(remaining.first?.embeddingCount, 7)
    }
}
