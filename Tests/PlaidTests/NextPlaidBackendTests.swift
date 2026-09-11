import Foundation
import XCTest

@testable import Plaid

/// Round-trips the Rust `next-plaid` engine through `NextPlaidBackend`:
/// build a tiny index, search, append, delete (with compaction), reconstruct.
/// Uses one-hot unit vectors so a query along an axis maps to a known doc id.
final class NextPlaidBackendTests: XCTestCase {
    private let dim = 64

    /// A document whose every token points along `axis` (already unit-L2).
    private func oneHot(axis: Int, tokens: Int = 4) -> [[Float]] {
        var row = [Float](repeating: 0, count: dim)
        row[axis] = 1
        return Array(repeating: row, count: tokens)
    }

    private func params(topK: Int) -> SearchParameters {
        // Probe every cell; tiny indexes have very few centroids.
        SearchParameters(batchSize: 2000, nFullScores: 4096, topK: topK, nIvfProbe: 1024)
    }

    private func tempDir() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("rust_backend_\(UUID().uuidString)", isDirectory: true)
    }

    func testCreateSearchAddDeleteReconstruct() throws {
        let backend = NextPlaidBackend()
        let dir = tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        // Create with docs along axes 0, 1, 2.
        try backend.create(
            indexURL: dir,
            embeddingDim: dim,
            nbits: 2,
            embeddings: [oneHot(axis: 0), oneHot(axis: 1), oneHot(axis: 2)],
            batchSize: 50_000,
            seed: 42
        )

        // Query along axis 1 -> top hit is doc 1.
        let results = try backend.loadAndSearch(
            indexURL: dir,
            queries: [oneHot(axis: 1, tokens: 1)],
            searchParameters: params(topK: 3),
            showProgress: false,
            preloadIndex: false,
            subset: nil
        )
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].passageIds.first, 1)

        // Append a doc along axis 3 -> id 3, existing ids unchanged.
        let newIds = try backend.update(
            indexURL: dir, embeddings: [oneHot(axis: 3)], batchSize: 50_000)
        XCTAssertEqual(newIds, [3])

        // Delete the middle doc (id 1); engine compacts survivors.
        let outcome = try backend.delete(indexURL: dir, subset: [1])
        XCTAssertEqual(outcome.deletedIdsSorted, [1])

        // After compaction old id 2 -> new id 1: axis-2 content now tops at id 1.
        let afterDelete = try backend.loadAndSearch(
            indexURL: dir,
            queries: [oneHot(axis: 2, tokens: 1)],
            searchParameters: params(topK: 3),
            showProgress: false,
            preloadIndex: false,
            subset: nil
        )
        XCTAssertEqual(afterDelete[0].passageIds.first, 1)

        // Reconstruct doc 0: shape [tokens, dim].
        let recon = try backend.getDocumentEmbeddings(indexURL: dir, documentId: 0)
        XCTAssertEqual(recon.count, 4)
        XCTAssertEqual(recon.first?.count, dim)
    }
}
