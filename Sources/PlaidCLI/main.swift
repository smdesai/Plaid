import Foundation
import Plaid

@main
enum PlaidCLI {
    private static let repoRoot: URL = {
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0 ..< 5 {
            url.deleteLastPathComponent()
        }
        return url
    }()

    private static let fixturesDirectory: URL = repoRoot.appendingPathComponent(
        "fixtures", isDirectory: true)
    private static let defaultTokenizerModelId = "LiquidAI/LFM2-ColBERT-350M"

    static func main() async throws {
        let args = Array(CommandLine.arguments.dropFirst())
        guard let command = args.first else {
            printUsage()
            return
        }
        let commandArgs = Array(args.dropFirst())

        switch command.lowercased() {
        case "quickstart":
            try runQuickStart()
        case "update":
            try runUpdate()
        case "tokenize":
            try await runTokenizer(arguments: commandArgs)
        case "similarity":
            try await runSimilarity(arguments: commandArgs)
        default:
            print("Unknown command: \(command)\n")
            printUsage()
        }
    }

    private static func printUsage() {
        let exe = (CommandLine.arguments.first as NSString?)?.lastPathComponent ?? "PlaidCLI"
        print(
            """
            Usage: \(exe) <command>

              quickstart   Create an index, execute a search, and print the top results.
              update       Create an index, append new documents, and search again.
              tokenize     Tokenize a string using a pretrained tokenizer and print tokens/ids.
                           Usage: tokenize [--query|--doc] [--pretrained MODEL_ID] TEXT
                           Default: \(defaultTokenizerModelId)
                           Example: tokenize "test" or tokenize --pretrained CUSTOM_MODEL "test"
              similarity   Encode a query/document pair with the Core ML ColBERT model and print their score.
                           Usage: similarity --query "..." --doc "..." [--pretrained MODEL_ID]
            """)
    }

    private static func runQuickStart() throws {
        let documents = try loadFloatTensor3(named: "documents.json")
        let queries = try loadFloatTensor3(named: "queries.json")
        let centroids = try loadFloatTensor2(named: "centroids.json")
        let config: FixtureConfig? = try? loadFixture(named: "config.json")

        guard
            let embeddingDim = centroids.first?.count,
            embeddingDim > 0,
            let centroidCount = centroids.first.map({ _ in centroids.count }),
            centroidCount > 0
        else {
            throw NSError(
                domain: "PlaidCLI", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Centroid fixture is empty."])
        }

        if let expectedDim = config?.embedding_dim, expectedDim != embeddingDim {
            throw NSError(
                domain: "PlaidCLI",
                code: 3,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "Fixture embedding_dim mismatch (expected \(expectedDim), got \(embeddingDim))."
                ]
            )
        }

        let nbits = config?.nbits ?? Int(round(log2(Double(centroidCount))))
        guard 1 << nbits == centroidCount else {
            throw NSError(
                domain: "PlaidCLI", code: 2,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "Centroid count \(centroidCount) does not match nbits=\(nbits)."
                ])
        }

        let topK = config?.top_k ?? (try? loadFixtureResults().first?.passage_ids.count) ?? 10

        print(
            "Loaded fixtures -> documents: \(documents.count) docs × \(documents.first?.count ?? 0) tokens × \(documents.first?.first?.count ?? 0) dim"
        )
        print(
            "Queries: \(queries.count) × \(queries.first?.count ?? 0) × \(queries.first?.first?.count ?? 0)"
        )
        print("Centroids: \(centroids.count) × \(centroids.first?.count ?? 0); nbits=\(nbits)")
        if let suspicious = documents.first(where: { !$0.allSatisfy { $0.count == embeddingDim } })
        {
            print(
                "⚠️  Found document with inconsistent dimension: tokens=\(suspicious.count), dims=\(suspicious.map { $0.count })"
            )
        }
        if let firstDoc = documents.first?.first {
            print("First token sample: \(firstDoc.prefix(8))")
        }

        let params = SearchParameters(
            batchSize: max(queries.count, 1),
            nFullScores: 1024,
            topK: topK,
            nIvfProbe: 8,
            logTiming: true
        )

        let indexURL = fixturesDirectory.appendingPathComponent("python_index", isDirectory: true)
        guard
            FileManager.default.fileExists(
                atPath: indexURL.appendingPathComponent("metadata.json").path),
            FileManager.default.fileExists(
                atPath: indexURL.appendingPathComponent("plaid_index.json").path)
        else {
            throw NSError(
                domain: "PlaidCLI",
                code: 4,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "Expected fixtures/python_index. Run python_parity_test.py first."
                ]
            )
        }

        let results = try Plaid.loadAndSearch(
            indexURL: indexURL,
            device: "cpu",
            queries: queries,
            searchParameters: params,
            showProgress: false,
            preloadIndex: false
        )

        print("Quick Start Results (top \(topK)):\n")
        printResults(results)

        if let expected = try? loadFixtureResults() {
            compareResults(swiftResults: results, pythonResults: expected, tolerance: 1e-2)
        } else {
            print("No python_results.json fixture found; skipping parity comparison.\n")
        }
    }

    private static func runUpdate() throws {
        let embeddingDim = 128
        let nbits = 4
        let documentCount = 100
        let tokensPerDocument = 300
        let updateCount = 50
        let queriesCount = 2
        let tokensPerQuery = 50
        let topK = 10

        var rng = SeededGenerator(seed: 1337)
        let indexURL = defaultIndexURL(named: "update")
        try resetIndexDirectory(at: indexURL)

        let initialDocs = generateDocuments(
            count: documentCount,
            tokens: tokensPerDocument,
            dim: embeddingDim,
            using: &rng
        )

        let centroids = generateCentroids(
            count: max(1 << nbits, 64),
            dim: embeddingDim,
            using: &rng
        )

        try Plaid.create(
            indexURL: indexURL,
            device: "cpu",
            embeddingDim: embeddingDim,
            nbits: nbits,
            embeddings: initialDocs,
            centroids: centroids,
            batchSize: 64
        )

        let queries = generateDocuments(
            count: queriesCount,
            tokens: tokensPerQuery,
            dim: embeddingDim,
            using: &rng
        )

        let params = SearchParameters(
            batchSize: 32,
            nFullScores: 32,
            topK: topK,
            nIvfProbe: 1
        )

        let beforeUpdate = try Plaid.loadAndSearch(
            indexURL: indexURL,
            device: "cpu",
            queries: queries,
            searchParameters: params,
            showProgress: false,
            preloadIndex: false
        )

        let newDocs = generateDocuments(
            count: updateCount,
            tokens: tokensPerDocument,
            dim: embeddingDim,
            using: &rng
        )

        do {
            try Plaid.update(
                indexURL: indexURL,
                device: "cpu",
                embeddings: newDocs,
                batchSize: 64
            )
        } catch {
            print("Warning: update not yet supported for Swift backend: \(error)\n")
        }

        let afterUpdate = try Plaid.loadAndSearch(
            indexURL: indexURL,
            device: "cpu",
            queries: queries,
            searchParameters: params,
            showProgress: false,
            preloadIndex: false
        )

        print("Update Example (top \(topK)):\n")
        print("Results before update:\n")
        printResults(beforeUpdate)

        print("\nResults after update (additional \(updateCount) docs):\n")
        printResults(afterUpdate)
    }

    private static func runTokenizer(arguments: [String]) async throws {
        var remaining = arguments
        var isQuery = true
        var modelId: String? = nil

        while let first = remaining.first?.lowercased() {
            switch first {
            case "--doc", "--document":
                isQuery = false
                remaining.removeFirst()
            case "--query":
                remaining.removeFirst()
            case "--pretrained", "--model":
                remaining.removeFirst()
                modelId = remaining.first
                if modelId != nil {
                    remaining.removeFirst()
                }
            default:
                break
            }

            // Only continue loop if we consumed a flag
            if !first.hasPrefix("--") {
                break
            }
        }

        guard !remaining.isEmpty else {
            print("tokenize command expects text input")
            print("Usage: tokenize [--query|--doc] [--pretrained MODEL_ID] TEXT")
            print("Default model: \(defaultTokenizerModelId)\n")
            return
        }

        let input = remaining.joined(separator: " ")
        let finalModelId = modelId ?? defaultTokenizerModelId

        print("Loading pretrained tokenizer from: \(finalModelId)...")
        let tokenizer = try await PreTrainedColbertTokenizer.from(pretrained: finalModelId)
        printTokenizerOutput(input: input, isQuery: isQuery, tokenizer: tokenizer)
    }

    private static func printTokenizerOutput(
        input: String, isQuery: Bool, tokenizer: PreTrainedColbertTokenizer
    ) {
        let tokens = tokenizer.tokenize(text: input)
        let tokenIds = tokenizer.tokenizeToIds(text: input)
        let encoded = tokenizer.buildModelTokens(sentence: input, isQuery: isQuery)

        print("Input: \(input)")
        print("Mode: \(isQuery ? "Query" : "Document")")
        print("\nWordPiece tokens:")
        print(tokens.joined(separator: ", "))
        print("\nToken IDs:")
        print(tokenIds.map(String.init).joined(separator: ", "))
        print("\nEncoded sequence (with \(isQuery ? "[Q]" : "[D]") prefix and padding):")
        print(encoded.map(String.init).joined(separator: ", "))
    }

    private static func runSimilarity(arguments: [String]) async throws {
        var queryText: String?
        var documentText: String?
        var modelId: String?

        var idx = 0
        while idx < arguments.count {
            let arg = arguments[idx]
            switch arg.lowercased() {
            case "--query", "-q":
                idx += 1
                guard idx < arguments.count else {
                    print("Missing value after --query\n")
                    return
                }
                queryText = arguments[idx]
            case "--doc", "--document", "-d":
                idx += 1
                guard idx < arguments.count else {
                    print("Missing value after --doc\n")
                    return
                }
                documentText = arguments[idx]
            case "--pretrained", "--model":
                idx += 1
                guard idx < arguments.count else {
                    print("Missing value after --pretrained\n")
                    return
                }
                modelId = arguments[idx]
            default:
                if queryText == nil {
                    queryText = arg
                } else if documentText == nil {
                    documentText = arg
                } else {
                    print("Unexpected argument: \(arg)\n")
                    return
                }
            }
            idx += 1
        }

        guard let queryText, let documentText else {
            print("similarity command expects a query and a document.")
            print("Usage: similarity --query \"...\" --doc \"...\" [--pretrained MODEL_ID]\n")
            return
        }

        let finalModelId = modelId ?? defaultTokenizerModelId
        print("Loading tokenizer/model: \(finalModelId)")
        let tokenizer = try await PreTrainedColbertTokenizer.from(pretrained: finalModelId)
        let generator = try LFM2ColbertEmbeddingGenerator(tokenizer: tokenizer)
        let colbert = ColbertModel(
            generator: generator,
            configuration: .init(
                batchSize: 1,
                embeddingDimension: 128,
                queryLength: tokenizer.maxSequenceLength,
                documentLength: tokenizer.maxSequenceLength
            )
        )

        print("=== query embedding ===")
        let queryEmbedding = try colbert.encode(sentence: queryText, isQuery: true)
        print("=== document embedding ===")
        let documentEmbedding = try colbert.encode(sentence: documentText, isQuery: false)
        print("=== similarity ===")
        let score = try colbert.similarity(query: queryEmbedding, document: documentEmbedding)

        print("\nQuery: \(queryText)")
        print("Document: \(documentText)")
        print(String(format: "ColBERT score: %.4f", score))
    }

    private static func printResults(_ results: [QueryResult]) {
        for result in results {
            print("Query \(result.queryId):")
            if result.passageIds.isEmpty {
                print("  (no matches)")
                continue
            }
            for (docId, score) in zip(result.passageIds, result.scores) {
                print(String(format: "  • doc %3d  score %.4f", docId, score))
            }
            print("")
        }
    }

    private static func generateDocuments(
        count: Int,
        tokens: Int,
        dim: Int,
        using rng: inout some RandomNumberGenerator
    ) -> [[[Float]]] {
        (0 ..< count).map { _ in
            (0 ..< tokens).map { _ in
                (0 ..< dim).map { _ in Float.random(in: -1.0 ... 1.0, using: &rng) }
            }
        }
    }

    private static func generateCentroids(
        count: Int,
        dim: Int,
        using rng: inout some RandomNumberGenerator
    ) -> [[Float]] {
        (0 ..< count).map { _ in
            (0 ..< dim).map { _ in Float.random(in: -1.0 ... 1.0, using: &rng) }
        }
    }

    private static func defaultIndexURL(named suffix: String) -> URL {
        let base =
            (ProcessInfo.processInfo.environment["PLAID_CLI_INDEX_DIR"]
                .map { URL(fileURLWithPath: $0, isDirectory: true) })
            ?? URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
            .appendingPathComponent(".plaid", isDirectory: true)
        return base.appendingPathComponent(suffix, isDirectory: true)
    }

    private static func resetIndexDirectory(at url: URL) throws {
        let parent = url.deletingLastPathComponent()
        if !FileManager.default.fileExists(atPath: parent.path) {
            try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
        }

        if FileManager.default.fileExists(atPath: url.path) {
            try? FileManager.default.removeItem(at: url)
        }
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    }

    private static func loadFixture<T: Decodable>(named name: String) throws -> T {
        let path = fixturesDirectory.appendingPathComponent(name)
        let data = try Data(contentsOf: path)
        return try JSONDecoder().decode(T.self, from: data)
    }

    private static func loadFloatTensor3(named name: String) throws -> [[[Float]]] {
        let path = fixturesDirectory.appendingPathComponent(name)
        let data = try Data(contentsOf: path)
        return try JSONDecoder().decode([[[Float]]].self, from: data)
    }

    private static func loadFloatTensor2(named name: String) throws -> [[Float]] {
        let path = fixturesDirectory.appendingPathComponent(name)
        let data = try Data(contentsOf: path)
        return try JSONDecoder().decode([[Float]].self, from: data)
    }

    private static func loadFixtureResults() throws -> [FixtureResult] {
        try loadFixture(named: "python_results.json")
    }

    private static func compareResults(
        swiftResults: [QueryResult],
        pythonResults: [FixtureResult],
        tolerance: Float
    ) {
        let expectedById = Dictionary(uniqueKeysWithValues: pythonResults.map { ($0.query_id, $0) })
        var mismatches = 0

        for result in swiftResults {
            guard let expected = expectedById[result.queryId] else {
                print("⚠️  No python baseline for query \(result.queryId)")
                mismatches += 1
                continue
            }

            let swiftPairs = zip(result.passageIds, result.scores)
            let expectedPairs = zip(expected.passage_ids, expected.scores)

            let sortedSwift =
                swiftPairs
                .map { ($0.0, $0.1) }
                .sorted { lhs, rhs in
                    let diff = lhs.1 - rhs.1
                    if abs(diff) > tolerance { return diff > 0 }
                    return lhs.0 < rhs.0
                }
            let sortedExpected =
                expectedPairs
                .map { (Int($0.0), $0.1) }
                .sorted { lhs, rhs in
                    let diff = lhs.1 - rhs.1
                    if abs(diff) > tolerance { return diff > 0 }
                    return lhs.0 < rhs.0
                }

            if sortedSwift.map(\.0) != sortedExpected.map(\.0) {
                print("⚠️  Passage ID mismatch for query \(result.queryId)")
                print("    Swift : \(sortedSwift.map(\.0))")
                print("    Python: \(sortedExpected.map(\.0))")
                mismatches += 1
            }

            let count = min(sortedSwift.count, sortedExpected.count)
            for idx in 0 ..< count {
                let delta = abs(sortedSwift[idx].1 - Float(sortedExpected[idx].1))
                if delta > tolerance {
                    print(
                        String(
                            format:
                                "⚠️  Score mismatch q%03d #%02d | swift=%.6f python=%.6f (Δ=%.6f)",
                            result.queryId,
                            idx,
                            sortedSwift[idx].1,
                            sortedExpected[idx].1,
                            delta
                        )
                    )
                    mismatches += 1
                }
            }
        }

        if mismatches == 0 {
            print("✅ Swift results match python_results.json within ±\(tolerance).")
        } else {
            print("⚠️  Detected \(mismatches) differences against python_results.json.")
        }
        print("")
    }

    private struct FixtureResult: Decodable {
        let query_id: Int
        let passage_ids: [Int]
        let scores: [Float]
    }

    private struct FixtureConfig: Decodable {
        let embedding_dim: Int?
        let nbits: Int
        let top_k: Int?
        let batch_size: Int?
    }
}
