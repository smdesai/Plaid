import Foundation
import XCTest

@testable import Plaid

/// One entry of the Python PLAID reference output (`python_results.json`):
/// the ground-truth top-k passage ids + scores for a query.
private struct PythonResult: Decodable {
    let query_id: Int
    let passage_ids: [Int]
    let scores: [Double]
}

/// Cross-implementation parity against the Python PLAID reference.
///
/// The fixtures are generated out-of-tree by a Python script and are **not**
/// checked in, so this test `XCTSkip`s when they are absent. When present it
/// builds a `NextPlaidBackend` index from `documents.json`, searches it with
/// `queries.json`, and checks that the engine's ranking matches Python's
/// `python_results.json`.
///
/// Quantization differs between the two implementations, so the assertion is on
/// *ranking* (top-1 identical, strong top-k overlap), never on raw scores.
final class PlaidTests: XCTestCase {
    private static let packageRoot: URL = {
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0 ..< 3 { url.deleteLastPathComponent() }  // Tests/PlaidTests/<file>
        return url
    }()

    private static let fixturesRoot = packageRoot.appendingPathComponent(
        "fixtures", isDirectory: true)

    private func overlapAtK(_ a: [Int], _ b: [Int], k: Int) -> Double {
        let sa = Set(a.prefix(k))
        let sb = Set(b.prefix(k))
        guard !sa.isEmpty else { return sb.isEmpty ? 1.0 : 0.0 }
        return Double(sa.intersection(sb).count) / Double(sa.count)
    }

    func testFixtureParity() throws {
        let docsURL = Self.fixturesRoot.appendingPathComponent("documents.json")
        let queriesURL = Self.fixturesRoot.appendingPathComponent("queries.json")
        let resultsURL = Self.fixturesRoot.appendingPathComponent("python_results.json")

        for url in [docsURL, queriesURL, resultsURL] {
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw XCTSkip("Fixture missing: \(url.path)")
            }
        }

        let documents = try JSONDecoder().decode(
            [[[Float]]].self, from: Data(contentsOf: docsURL))
        let queries = try JSONDecoder().decode(
            [[[Float]]].self, from: Data(contentsOf: queriesURL))
        let expected = try JSONDecoder().decode(
            [PythonResult].self, from: Data(contentsOf: resultsURL))

        XCTAssertFalse(documents.isEmpty, "no documents in fixture")
        XCTAssertEqual(queries.count, expected.count, "query/result count mismatch")

        let dim = documents[0].first?.count ?? 0
        XCTAssertGreaterThan(dim, 0, "documents have zero-width embeddings")
        let topK = expected.first?.passage_ids.count ?? 10

        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("fixture_parity_\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let backend = NextPlaidBackend()
        try backend.create(
            indexURL: dir, embeddingDim: dim, nbits: 4,
            embeddings: documents, batchSize: 50_000, seed: 42)

        let params = SearchParameters(
            batchSize: max(1, queries.count), nFullScores: 4096, topK: topK, nIvfProbe: 1024)
        let results = try backend.loadAndSearch(
            indexURL: dir, queries: queries, searchParameters: params,
            showProgress: false, preloadIndex: false, subset: nil)

        XCTAssertEqual(results.count, expected.count)

        var overlapSum = 0.0
        for (rust, python) in zip(results, expected) {
            XCTAssertEqual(rust.queryId, python.query_id)
            if let rTop = rust.passageIds.first, let pTop = python.passage_ids.first {
                XCTAssertEqual(rTop, pTop, "top-1 mismatch for query \(python.query_id)")
            }
            overlapSum += overlapAtK(rust.passageIds, python.passage_ids, k: topK)
        }

        let avgOverlap = overlapSum / Double(expected.count)
        XCTAssertGreaterThanOrEqual(
            avgOverlap, 0.8, "avg top-\(topK) overlap vs Python reference too low: \(avgOverlap)")
    }
}
