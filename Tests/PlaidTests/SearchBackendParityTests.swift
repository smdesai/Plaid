import Foundation
import XCTest

@testable import Plaid

/// Correctness oracle for `NextPlaidBackend`: the engine's approximate
/// late-interaction ranking is checked against an **exact brute-force MaxSim CPU
/// reference** (the ground-truth ColBERT score the engine approximates). Fully
/// in-process and environment-independent.
///
/// Quantization is lossy, so assertions compare *ranking* (top-1 and top-k
/// overlap), never raw scores.
final class SearchBackendParityTests: XCTestCase {
    private let dim = 64
    private let nbits = 4  // 16 centroids: fine enough that ranking is clean.

    // Deterministic RNG (splitmix64) so the corpus/queries are reproducible.
    private struct SeededRNG: RandomNumberGenerator {
        var state: UInt64
        init(seed: UInt64) { state = seed &+ 0x9E37_79B9_7F4A_7C15 }
        mutating func next() -> UInt64 {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
    }

    private func normalize(_ v: [Float]) -> [Float] {
        let n = sqrt(v.reduce(0) { $0 + $1 * $1 })
        return n > 1e-12 ? v.map { $0 / n } : v
    }

    private func randUnit(_ rng: inout SeededRNG) -> [Float] {
        normalize((0 ..< dim).map { _ in Float.random(in: -1 ... 1, using: &rng) })
    }

    private func dot(_ a: [Float], _ b: [Float]) -> Float {
        var s: Float = 0
        for i in 0 ..< min(a.count, b.count) { s += a[i] * b[i] }
        return s
    }

    /// Deterministic corpus with **graded relevance** so the top-k is a genuine
    /// ordered ladder (not just a single meaningful hit).
    ///
    /// Six near-orthogonal topic bases; per topic, five docs whose base is
    /// `normalize(α·T + √(1-α²)·U)` for α = 1.0, 0.82, 0.64, 0.46, 0.28 (U a
    /// per-doc distractor orthogonalized against T). A single-token query near
    /// topic T then scores that topic's five docs at ≈α (descending, clearly
    /// separated) and every other doc at ≈0 — an unambiguous top-5 the engine
    /// should recover. Returns raw vectors; the backend normalizes as needed.
    private func makeCorpus() -> (docs: [[[Float]]], queries: [[[Float]]], targets: [Int]) {
        var rng = SeededRNG(seed: 42)
        let nTopics = 6
        let ranksPerTopic = 5
        let tokensPerDoc = 8
        let alphas: [Float] = [1.0, 0.82, 0.64, 0.46, 0.28]

        let topics = (0 ..< nTopics).map { _ in randUnit(&rng) }

        var docs: [[[Float]]] = []
        for topic in 0 ..< nTopics {
            for rank in 0 ..< ranksPerTopic {
                // Distractor orthogonal to this topic, mixed in at weight √(1-α²).
                var u = randUnit(&rng)
                let proj = dot(u, topics[topic])
                u = normalize(zip(u, topics[topic]).map { $0 - proj * $1 })
                let a = alphas[rank]
                let w = (1 - a * a).squareRoot()
                let base = normalize(zip(topics[topic], u).map { a * $0 + w * $1 })
                let doc = (0 ..< tokensPerDoc).map { _ in
                    base.map { $0 + Float.random(in: -0.02 ... 0.02, using: &rng) }
                }
                docs.append(doc)
            }
        }

        // One query per topic, aimed at the α=1.0 doc (rank 0) of each group.
        let targets = (0 ..< nTopics).map { $0 * ranksPerTopic }
        let queries: [[[Float]]] = (0 ..< nTopics).map { topic in
            [topics[topic].map { $0 + Float.random(in: -0.02 ... 0.02, using: &rng) }]
        }
        return (docs, queries, targets)
    }

    /// Exact ColBERT MaxSim over unit-normalized tokens: Σ_q max_d (q · d).
    private func maxSim(query: [[Float]], doc: [[Float]]) -> Float {
        let q = query.map { normalize($0) }
        let d = doc.map { normalize($0) }
        var total: Float = 0
        for qt in q {
            var best = -Float.greatestFiniteMagnitude
            for dt in d {
                var s: Float = 0
                for i in 0 ..< min(qt.count, dt.count) { s += qt[i] * dt[i] }
                if s > best { best = s }
            }
            total += best
        }
        return total
    }

    /// Exact top-k document ids for a query, ranked by MaxSim.
    private func exactRanking(query: [[Float]], docs: [[[Float]]], k: Int) -> [Int] {
        docs.enumerated()
            .map { (idx, doc) in (idx, maxSim(query: query, doc: doc)) }
            .sorted { $0.1 > $1.1 }
            .prefix(k)
            .map { $0.0 }
    }

    private func tempDir(_ tag: String) -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("parity_\(tag)_\(UUID().uuidString)", isDirectory: true)
    }

    private func overlapAtK(_ a: [Int], _ b: [Int], k: Int) -> Double {
        let sa = Set(a.prefix(k))
        let sb = Set(b.prefix(k))
        guard !sa.isEmpty else { return sb.isEmpty ? 1.0 : 0.0 }
        return Double(sa.intersection(sb).count) / Double(sa.count)
    }

    private func params(topK: Int) -> SearchParameters {
        SearchParameters(batchSize: 2000, nFullScores: 4096, topK: topK, nIvfProbe: 1024)
    }

    // MARK: - Rust vs exact MaxSim

    func testRustMatchesExactMaxSimRanking() throws {
        let (docs, queries, targets) = makeCorpus()

        let dir = tempDir("rust")
        defer { try? FileManager.default.removeItem(at: dir) }

        let rust = NextPlaidBackend()
        try rust.create(
            indexURL: dir, embeddingDim: dim, nbits: nbits,
            embeddings: docs, batchSize: 50_000, seed: 42)

        let rustRes = try rust.loadAndSearch(
            indexURL: dir, queries: queries, searchParameters: params(topK: 5),
            showProgress: false, preloadIndex: false, subset: nil)

        XCTAssertEqual(rustRes.count, targets.count)

        var overlapSum = 0.0
        for (q, target) in targets.enumerated() {
            let rIds = rustRes[q].passageIds
            let exact = exactRanking(query: queries[q], docs: docs, k: 5)
            XCTAssertEqual(exact.first, target, "exact top-1 wrong for query \(q)")
            XCTAssertEqual(rIds.first, target, "rust top-1 wrong for query \(q)")
            overlapSum += overlapAtK(rIds, exact, k: 5)
        }

        let avgOverlap = overlapSum / Double(targets.count)
        XCTAssertGreaterThanOrEqual(
            avgOverlap, 0.6, "avg top-5 overlap vs exact too low: \(avgOverlap)")
    }
}
