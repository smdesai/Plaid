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
            try await runUserUpdate(arguments: commandArgs)
        case "update-test":
            try runUpdateTest()
        case "delete":
            try runUserDelete(arguments: commandArgs)
        case "tokenize":
            try await runTokenizer(arguments: commandArgs)
        case "similarity":
            try await runSimilarity(arguments: commandArgs)
        case "demo", "index-and-search":
            try await runDemo(arguments: commandArgs)
        default:
            print("Unknown command: \(command)\n")
            printUsage()
        }
    }

    private static func printUsage() {
        let exe = (CommandLine.arguments.first as NSString?)?.lastPathComponent ?? "PlaidCLI"
        print(
            """
            Usage: \(exe) <command> [options]

            Commands:
              demo         End-to-end demo: encode documents with ColBERT, create a Plaid index, and search.
                           Usage: demo --query "..." [--docs "..." | --files "..." | --index-path PATH] [OPTIONS]
                           Options: --pretrained MODEL_ID, --top-k N, --nbits N, --keep-index, --index-name NAME
                           Examples:
                             demo --query "..." --docs "text1" "text2" --files "doc1.txt" "doc2.md" --keep-index
                             demo --query "..." --index-path /path/to/index

              update       Add new documents to an existing Plaid index.
                           Usage: update --index-path PATH --files FILE... [OPTIONS]
                           Options: --pretrained MODEL_ID, --batch-size N
                           Example: update -i ~/.plaid/my_index -f doc1.txt doc2.txt

              delete       Remove documents from an existing Plaid index by document IDs.
                           Usage: delete --index-path PATH (--doc-ids ID... | --ids-file FILE)
                           Examples:
                             delete -i ~/.plaid/my_index -d 5 12 23 45
                             delete -i ~/.plaid/my_index -f to_delete.txt

              quickstart   Create an index, execute a search, and print the top results (test data).

              tokenize     Tokenize a string using a pretrained tokenizer and print tokens/ids.
                           Usage: tokenize [--query|--doc] [--pretrained MODEL_ID] TEXT
                           Default: \(defaultTokenizerModelId)
                           Example: tokenize "test" or tokenize --pretrained CUSTOM_MODEL "test"

              similarity   Encode a query/document pair with the Core ML ColBERT model and print their score.
                           Usage: similarity --query "..." --doc "..." [--pretrained MODEL_ID]

            For command-specific help, run: \(exe) <command> --help
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

    private static func runUpdateTest() throws {
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

        try Plaid.update(
            indexURL: indexURL,
            embeddings: newDocs,
            batchSize: 64
        )

        let afterUpdate = try Plaid.loadAndSearch(
            indexURL: indexURL,
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
        let chunker = TokenSplitter(withTokenizer: tokenizer)
        let colbert = ColbertModel(
            generator: generator,
            configuration: .init(
                batchSize: 1,
                embeddingDimension: 128,
                queryLength: tokenizer.maxSequenceLength,
                documentLength: tokenizer.maxSequenceLength
            ),
            chunker: chunker
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

    private static func runDemo(arguments: [String]) async throws {
        var queryText: String?
        var documentTexts: [String] = []
        var documentFiles: [String] = []
        var modelId: String?
        var topK: Int = 5
        var nbits: Int = 2
        var keepIndex: Bool = false
        var indexName: String? = nil
        var existingIndexPath: String? = nil

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
            case "--docs", "--documents", "-d":
                idx += 1
                // Collect all subsequent non-flag arguments as documents
                while idx < arguments.count && !arguments[idx].hasPrefix("--") {
                    documentTexts.append(arguments[idx])
                    idx += 1
                }
                idx -= 1  // Step back one since the outer loop will increment
            case "--files", "-f":
                idx += 1
                // Collect all subsequent non-flag arguments as file paths
                while idx < arguments.count && !arguments[idx].hasPrefix("--") {
                    documentFiles.append(arguments[idx])
                    idx += 1
                }
                idx -= 1  // Step back one since the outer loop will increment
            case "--index-path", "--index", "-i":
                idx += 1
                guard idx < arguments.count else {
                    print("Missing value after --index-path\n")
                    return
                }
                existingIndexPath = arguments[idx]
            case "--pretrained", "--model":
                idx += 1
                guard idx < arguments.count else {
                    print("Missing value after --pretrained\n")
                    return
                }
                modelId = arguments[idx]
            case "--top-k", "--topk", "-k":
                idx += 1
                guard idx < arguments.count, let k = Int(arguments[idx]) else {
                    print("Missing or invalid value after --top-k\n")
                    return
                }
                topK = k
            case "--nbits":
                idx += 1
                guard idx < arguments.count, let bits = Int(arguments[idx]) else {
                    print("Missing or invalid value after --nbits\n")
                    return
                }
                nbits = bits
            case "--keep-index", "--keep", "--persist":
                keepIndex = true
            case "--index-name", "--name":
                idx += 1
                guard idx < arguments.count else {
                    print("Missing value after --index-name\n")
                    return
                }
                indexName = arguments[idx]
            default:
                print("Unexpected argument: \(arg)\n")
                print(
                    "Usage: demo --query \"...\" [--docs \"...\" | --files \"...\" | --index-path PATH] [OPTIONS]\n"
                )
                print(
                    "Options: --pretrained MODEL_ID, --top-k N, --nbits N, --keep-index, --index-name NAME\n"
                )
                return
            }
            idx += 1
        }

        guard let queryText else {
            print("demo command requires a query.")
            print(
                "Usage: demo --query \"...\" [--docs \"doc1\" \"doc2\" ...] [--index-path PATH]\n")
            return
        }

        // Validate: either provide docs/files (to create new index) or index-path (to use existing), but not both
        if let existingIndexPath = existingIndexPath {
            if !documentTexts.isEmpty || !documentFiles.isEmpty {
                print("Error: Cannot specify both --docs/--files and --index-path.")
                print(
                    "Use --docs/--files to create a new index, or --index-path to use an existing one.\n"
                )
                return
            }
            // Use existing index path
            return try await runDemoWithExistingIndex(
                indexPath: existingIndexPath,
                queryText: queryText,
                modelId: modelId,
                topK: topK
            )
        }

        // Load document contents from files
        var allDocuments: [String] = documentTexts
        if !documentFiles.isEmpty {
            print("📂 Loading documents from files...\n")
            for filePath in documentFiles {
                let fileURL = URL(fileURLWithPath: filePath)

                // Check if file exists
                guard FileManager.default.fileExists(atPath: fileURL.path) else {
                    print("❌ Error: File not found: \(filePath)")
                    print("   Make sure the path is correct and the file exists.\n")
                    return
                }

                do {
                    print("  📖 Reading \(fileURL.lastPathComponent)...")
                    let content = try String(contentsOf: fileURL, encoding: .utf8)
                    let trimmed = content.trimmingCharacters(in: .whitespacesAndNewlines)

                    guard !trimmed.isEmpty else {
                        print("⚠️  Warning: Skipping empty file: \(filePath)")
                        continue
                    }

                    allDocuments.append(trimmed)
                    let preview = trimmed.prefix(60)
                    print("  ✅ \(fileURL.lastPathComponent): \(trimmed.count) chars")
                    print("     \"\(preview)\(trimmed.count > 60 ? "..." : "")\"\n")
                } catch {
                    print("❌ Error reading file \(filePath): \(error.localizedDescription)\n")
                    return
                }
            }
        }

        // Creating new index - require documents
        guard !allDocuments.isEmpty else {
            print(
                "demo command requires either --docs/--files (to create new index) or --index-path (to use existing)."
            )
            print("Usage: demo --query \"...\" --docs \"doc1\" \"doc2\" ... [OPTIONS]\n")
            print("   or: demo --query \"...\" --files \"file1.txt\" \"file2.md\" ... [OPTIONS]\n")
            print("   or: demo --query \"...\" --index-path /path/to/index [OPTIONS]\n")
            return
        }

        // Replace documentTexts with all combined documents
        let finalDocumentTexts = allDocuments

        let finalModelId = modelId ?? defaultTokenizerModelId

        print("╔══════════════════════════════════════════════════════════════════════╗")
        print("║  Plaid + ColBERT Demo: End-to-End Text Search                       ║")
        print("╚══════════════════════════════════════════════════════════════════════╝\n")

        print("📚 Documents to index: \(finalDocumentTexts.count)")
        for (i, doc) in finalDocumentTexts.enumerated() {
            let preview = doc.prefix(60)
            print("  [\(i)] \(preview)\(doc.count > 60 ? "..." : "")")
        }
        print("\n🔍 Query: \"\(queryText)\"\n")

        // Load ColBERT model
        print("⚙️  Loading ColBERT model: \(finalModelId)...")
        let tokenizer = try await PreTrainedColbertTokenizer.from(pretrained: finalModelId)
        let generator = try LFM2ColbertEmbeddingGenerator(tokenizer: tokenizer)
        let chunker = TokenSplitter(withTokenizer: tokenizer)
        let colbert = ColbertModel(
            generator: generator,
            configuration: .init(
                batchSize: 1,
                embeddingDimension: 128,
                queryLength: tokenizer.maxSequenceLength,
                documentLength: tokenizer.maxSequenceLength
            ),
            chunker: chunker
        )
        print("✅ Model loaded\n")

        // Encode documents
        print("🔢 Encoding documents with ColBERT...")
        var documentEmbeddings: [[[Float]]] = []
        for (i, docText) in finalDocumentTexts.enumerated() {
            let embedding = try colbert.encode(sentence: docText, isQuery: false)
            documentEmbeddings.append(embedding)
            print(
                "  [\(i)] Encoded: \(embedding.count) tokens × \(embedding.first?.count ?? 0) dims")
        }
        print("✅ All documents encoded\n")

        // Generate centroids from document embeddings
        print("🎯 Generating centroids for quantization...")
        let centroids = try generateCentroidsFromDocuments(
            documentEmbeddings: documentEmbeddings,
            nbits: nbits,
            embeddingDim: 128
        )
        print("✅ Generated \(centroids.count) centroids\n")

        // Create index
        let indexSuffix = indexName ?? "demo_\(UUID().uuidString)"
        let indexURL = defaultIndexURL(named: indexSuffix)
        try resetIndexDirectory(at: indexURL)

        print("📦 Creating Plaid index...")
        print("  Index location: \(indexURL.path)")
        print("  nbits: \(nbits)")
        print("  Centroids: \(centroids.count)")

        try Plaid.create(
            indexURL: indexURL,
            embeddingDim: 128,
            nbits: nbits,
            embeddings: documentEmbeddings,
            centroids: centroids,
            batchSize: 64
        )
        print("✅ Index created\n")

        // Encode query
        print("🔍 Encoding query with ColBERT...")
        let queryEmbedding = try colbert.encode(sentence: queryText, isQuery: true)
        print("  Query: \(queryEmbedding.count) tokens × \(queryEmbedding.first?.count ?? 0) dims")
        print("✅ Query encoded\n")

        // Search
        print("🔎 Searching index...")
        let params = SearchParameters(
            batchSize: 1,
            nFullScores: finalDocumentTexts.count,
            topK: min(topK, finalDocumentTexts.count),
            nIvfProbe: min(8, centroids.count),
            logTiming: true
        )

        let results = try Plaid.loadAndSearch(
            indexURL: indexURL,
            queries: [queryEmbedding],
            searchParameters: params,
            showProgress: false,
            preloadIndex: false
        )
        print("✅ Search complete\n")

        // Display results
        print("╔══════════════════════════════════════════════════════════════════════╗")
        print("║  Search Results                                                      ║")
        print("╚══════════════════════════════════════════════════════════════════════╝\n")

        guard let firstResult = results.first else {
            print("No results found.\n")
            return
        }

        print("Query: \"\(queryText)\"\n")
        print("Top \(firstResult.passageIds.count) matches:\n")

        for (rank, (docId, score)) in zip(firstResult.passageIds, firstResult.scores).enumerated() {
            let docText = finalDocumentTexts[docId]
            let preview = docText.prefix(80)
            print(String(format: "%2d. [Score: %.4f] Doc %d", rank + 1, score, docId))
            print("    \(preview)\(docText.count > 80 ? "..." : "")\n")
        }

        // Cleanup or persist
        if keepIndex {
            print("\n💾 Index saved at: \(indexURL.path)")
            print("   Use this path to search again without re-indexing.\n")
        } else {
            print("\n🧹 Cleaning up temporary index...")
            try? FileManager.default.removeItem(at: indexURL)
        }
        print("✅ Done!\n")
    }

    /// Generates centroids by clustering document embeddings
    private static func generateCentroidsFromDocuments(
        documentEmbeddings: [[[Float]]],
        nbits: Int,
        embeddingDim: Int
    ) throws -> [[Float]] {
        let numCentroids = 1 << nbits

        // Collect all embedding vectors from all documents
        var allVectors: [[Float]] = []
        for docEmbedding in documentEmbeddings {
            allVectors.append(contentsOf: docEmbedding)
        }

        guard !allVectors.isEmpty else {
            throw NSError(
                domain: "PlaidCLI",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "No vectors found in documents"]
            )
        }

        // Simple centroid initialization: sample vectors uniformly or use k-means++
        // For simplicity, we'll sample uniformly from the available vectors
        var centroids: [[Float]] = []

        if allVectors.count <= numCentroids {
            // If we have fewer vectors than centroids, use all vectors and pad with random vectors
            centroids = allVectors
            var rng = SeededGenerator(seed: 42)
            while centroids.count < numCentroids {
                let randomVector = (0 ..< embeddingDim).map { _ in
                    Float.random(in: -1.0 ... 1.0, using: &rng)
                }
                centroids.append(normalizeVector(randomVector))
            }
        } else {
            // Sample uniformly from available vectors
            let stride = allVectors.count / numCentroids
            for i in 0 ..< numCentroids {
                let index = min(i * stride, allVectors.count - 1)
                centroids.append(allVectors[index])
            }
        }

        return centroids
    }

    /// Runs demo command with an existing index (query-only mode)
    private static func runDemoWithExistingIndex(
        indexPath: String,
        queryText: String,
        modelId: String?,
        topK: Int
    ) async throws {
        let indexURL = URL(fileURLWithPath: indexPath)

        // Verify index exists
        guard FileManager.default.fileExists(atPath: indexURL.path) else {
            print("Error: Index not found at path: \(indexPath)")
            print("Make sure the path is correct and the index was created with --keep-index.\n")
            return
        }

        // Verify index has required files
        let metadataURL = indexURL.appendingPathComponent("metadata.json")
        guard FileManager.default.fileExists(atPath: metadataURL.path) else {
            print("Error: Invalid index at path: \(indexPath)")
            print("The directory exists but doesn't appear to be a valid Plaid index.\n")
            return
        }

        let finalModelId = modelId ?? defaultTokenizerModelId

        print("╔══════════════════════════════════════════════════════════════════════╗")
        print("║  Plaid + ColBERT Demo: Search Existing Index                        ║")
        print("╚══════════════════════════════════════════════════════════════════════╝\n")

        print("📂 Index location: \(indexPath)")
        print("🔍 Query: \"\(queryText)\"\n")

        // Load index metadata to show stats
        if let metadataData = try? Data(contentsOf: metadataURL),
            let metadata = try? JSONDecoder().decode([String: AnyCodable].self, from: metadataData)
        {
            print("📊 Index stats:")
            if let docs = metadata["total_documents"]?.value as? Int {
                print("  Documents: \(docs)")
            }
            if let embs = metadata["num_embeddings"]?.value as? Int {
                print("  Embeddings: \(embs)")
            }
            if let dim = metadata["embedding_dim"]?.value as? Int {
                print("  Dimension: \(dim)")
            }
            if let avgLen = metadata["avg_doclen"]?.value as? Double {
                print(String(format: "  Avg doc length: %.1f tokens", avgLen))
            }
            print("")
        }

        // Load ColBERT model
        print("⚙️  Loading ColBERT model: \(finalModelId)...")
        let tokenizer = try await PreTrainedColbertTokenizer.from(pretrained: finalModelId)
        let generator = try LFM2ColbertEmbeddingGenerator(tokenizer: tokenizer)
        let chunker = TokenSplitter(withTokenizer: tokenizer)
        let colbert = ColbertModel(
            generator: generator,
            configuration: .init(
                batchSize: 1,
                embeddingDimension: 128,
                queryLength: tokenizer.maxSequenceLength,
                documentLength: tokenizer.maxSequenceLength
            ),
            chunker: chunker
        )
        print("✅ Model loaded\n")

        // Encode query
        print("🔍 Encoding query with ColBERT...")
        let queryEmbedding = try colbert.encode(sentence: queryText, isQuery: true)
        print("  Query: \(queryEmbedding.count) tokens × \(queryEmbedding.first?.count ?? 0) dims")
        print("✅ Query encoded\n")

        // Determine appropriate search parameters
        let metadata = try? JSONDecoder().decode(
            [String: AnyCodable].self, from: Data(contentsOf: metadataURL))
        let numDocs = (metadata?["total_documents"]?.value as? Int) ?? 10
        let numPartitions = (metadata?["num_partitions"]?.value as? Int) ?? 32

        // Search
        print("🔎 Searching index...")
        let params = SearchParameters(
            batchSize: 1,
            nFullScores: numDocs,
            topK: min(topK, numDocs),
            nIvfProbe: min(8, numPartitions),
            logTiming: true
        )

        let results = try Plaid.loadAndSearch(
            indexURL: indexURL,
            queries: [queryEmbedding],
            searchParameters: params,
            showProgress: false,
            preloadIndex: false
        )
        print("✅ Search complete\n")

        // Display results
        print("╔══════════════════════════════════════════════════════════════════════╗")
        print("║  Search Results                                                      ║")
        print("╚══════════════════════════════════════════════════════════════════════╝\n")

        guard let firstResult = results.first else {
            print("No results found.\n")
            return
        }

        print("Query: \"\(queryText)\"\n")
        print("Top \(firstResult.passageIds.count) matches:\n")

        for (rank, (docId, score)) in zip(firstResult.passageIds, firstResult.scores).enumerated() {
            print(String(format: "%2d. [Score: %.4f] Document %d", rank + 1, score, docId))
        }

        print("\n✅ Done!\n")
    }

    /// Helper type for decoding arbitrary JSON
    private struct AnyCodable: Codable {
        let value: Any

        init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let intValue = try? container.decode(Int.self) {
                value = intValue
            } else if let doubleValue = try? container.decode(Double.self) {
                value = doubleValue
            } else if let stringValue = try? container.decode(String.self) {
                value = stringValue
            } else if let boolValue = try? container.decode(Bool.self) {
                value = boolValue
            } else {
                value = ""
            }
        }

        func encode(to encoder: Encoder) throws {
            var container = encoder.singleValueContainer()
            if let intValue = value as? Int {
                try container.encode(intValue)
            } else if let doubleValue = value as? Double {
                try container.encode(doubleValue)
            } else if let stringValue = value as? String {
                try container.encode(stringValue)
            } else if let boolValue = value as? Bool {
                try container.encode(boolValue)
            }
        }
    }

    /// Normalizes a vector to unit length
    private static func normalizeVector(_ vector: [Float]) -> [Float] {
        let norm = sqrt(vector.reduce(Float(0)) { $0 + $1 * $1 })
        if norm == 0 || norm.isNaN {
            return vector
        }
        return vector.map { $0 / norm }
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

    // MARK: - User-Facing CLI Commands

    private static func runUserUpdate(arguments: [String]) async throws {
        var indexPath: String?
        var documentFiles: [String] = []
        var pretrainedModel = defaultTokenizerModelId
        var batchSize = 64

        // Parse arguments
        var idx = 0
        while idx < arguments.count {
            let arg = arguments[idx]
            switch arg.lowercased() {
            case "--index-path", "-i":
                idx += 1
                guard idx < arguments.count else {
                    print("❌ Error: --index-path requires a value")
                    return
                }
                indexPath = arguments[idx]

            case "--files", "-f":
                idx += 1
                while idx < arguments.count && !arguments[idx].hasPrefix("--") {
                    documentFiles.append(arguments[idx])
                    idx += 1
                }
                idx -= 1

            case "--pretrained", "-p":
                idx += 1
                guard idx < arguments.count else {
                    print("❌ Error: --pretrained requires a value")
                    return
                }
                pretrainedModel = arguments[idx]

            case "--batch-size", "-b":
                idx += 1
                guard idx < arguments.count, let size = Int(arguments[idx]) else {
                    print("❌ Error: --batch-size requires a numeric value")
                    return
                }
                batchSize = size

            case "--help", "-h":
                print(
                    """
                    Usage: PlaidCLI update --index-path <PATH> --files <FILE1> <FILE2> ... [OPTIONS]

                    Add new documents to an existing Plaid index.

                    Required:
                      --index-path, -i PATH      Path to existing index directory
                      --files, -f FILE...        One or more text files to add

                    Optional:
                      --pretrained, -p MODEL     HuggingFace model ID (default: \(defaultTokenizerModelId))
                      --batch-size, -b SIZE      Batch size for encoding (default: 64)
                      --help, -h                 Show this help

                    Example:
                      PlaidCLI update -i ~/.plaid/my_index -f doc1.txt doc2.txt
                    """)
                return

            default:
                print("⚠️  Warning: Unknown argument '\(arg)'")
            }
            idx += 1
        }

        // Validate required parameters
        guard let indexPath = indexPath else {
            print("❌ Error: --index-path is required\n")
            print("Usage: PlaidCLI update --index-path <PATH> --files <FILE1> <FILE2> ...")
            print("Run 'PlaidCLI update --help' for more information")
            return
        }

        guard !documentFiles.isEmpty else {
            print("❌ Error: --files is required (provide at least one file)\n")
            print("Usage: PlaidCLI update --index-path <PATH> --files <FILE1> <FILE2> ...")
            print("Run 'PlaidCLI update --help' for more information")
            return
        }

        let indexURL = URL(fileURLWithPath: indexPath)

        // Verify index exists
        guard FileManager.default.fileExists(atPath: indexURL.path) else {
            print("❌ Error: Index not found at '\(indexPath)'")
            print("   Make sure the index directory exists.")
            return
        }

        print("╔══════════════════════════════════════════════════════════════════════╗")
        print("║  Plaid Index Update                                                  ║")
        print("╚══════════════════════════════════════════════════════════════════════╝\n")

        print("📂 Index: \(indexPath)")
        print("📄 Documents to add: \(documentFiles.count)\n")

        // Load ColBERT model
        print("⚙️  Loading ColBERT model: \(pretrainedModel)...")
        let tokenizer = try await PreTrainedColbertTokenizer.from(pretrained: pretrainedModel)
        let generator = try LFM2ColbertEmbeddingGenerator(tokenizer: tokenizer)
        let chunker = TokenSplitter(withTokenizer: tokenizer)
        let colbert = ColbertModel(
            generator: generator,
            configuration: .init(
                embeddingDimension: 128,
                queryLength: 32,
                documentLength: 180
            ),
            chunker: chunker
        )
        print("✅ Model loaded\n")

        // Load and encode documents
        print("📚 Loading and encoding documents...\n")
        var allEmbeddings: [[[Float]]] = []

        for (index, filePath) in documentFiles.enumerated() {
            let fileURL = URL(fileURLWithPath: filePath)

            guard FileManager.default.fileExists(atPath: fileURL.path) else {
                print("⚠️  Warning: File not found, skipping: \(filePath)")
                continue
            }

            do {
                print(
                    "  [\(index + 1)/\(documentFiles.count)] 📖 Reading \(fileURL.lastPathComponent)..."
                )
                let content = try String(contentsOf: fileURL, encoding: .utf8)
                let trimmed = content.trimmingCharacters(in: .whitespacesAndNewlines)

                guard !trimmed.isEmpty else {
                    print("⚠️  Warning: Empty file, skipping: \(filePath)")
                    continue
                }

                print("      ✅ Loaded \(trimmed.count) chars, encoding...")

                // Encode document
                let embeddings = try colbert.encode(sentence: trimmed, isQuery: false)
                allEmbeddings.append(embeddings)
                print("      ✅ Encoded: \(embeddings.count) embeddings\n")

            } catch {
                print("❌ Error reading file \(filePath): \(error.localizedDescription)")
                print("   Skipping this file.\n")
            }
        }

        guard !allEmbeddings.isEmpty else {
            print("❌ Error: No valid documents to add")
            return
        }

        print("✅ Encoded \(allEmbeddings.count) document(s)\n")

        // Update index
        print("📦 Updating index...")
        do {
            try Plaid.update(
                indexURL: indexURL,
                embeddings: allEmbeddings,
                batchSize: batchSize
            )
            print("✅ Index updated successfully!")
            print("\n💾 Updated index: \(indexPath)")
            print("   Added \(allEmbeddings.count) document(s)")

        } catch {
            print("❌ Error updating index: \(error.localizedDescription)")
            throw error
        }
    }

    private static func runUserDelete(arguments: [String]) throws {
        var indexPath: String?
        var docIds: [Int] = []
        var idsFromFile: String?

        // Parse arguments
        var idx = 0
        while idx < arguments.count {
            let arg = arguments[idx]
            switch arg.lowercased() {
            case "--index-path", "-i":
                idx += 1
                guard idx < arguments.count else {
                    print("❌ Error: --index-path requires a value")
                    return
                }
                indexPath = arguments[idx]

            case "--doc-ids", "-d":
                idx += 1
                while idx < arguments.count && !arguments[idx].hasPrefix("--") {
                    if let id = Int(arguments[idx]) {
                        docIds.append(id)
                    } else {
                        print("⚠️  Warning: Invalid document ID '\(arguments[idx])', skipping")
                    }
                    idx += 1
                }
                idx -= 1

            case "--ids-file", "-f":
                idx += 1
                guard idx < arguments.count else {
                    print("❌ Error: --ids-file requires a value")
                    return
                }
                idsFromFile = arguments[idx]

            case "--help", "-h":
                print(
                    """
                    Usage: PlaidCLI delete --index-path <PATH> (--doc-ids <ID...> | --ids-file <FILE>)

                    Remove documents from an existing Plaid index by their document IDs.

                    Required:
                      --index-path, -i PATH      Path to existing index directory

                    Required (one of):
                      --doc-ids, -d ID...        Space-separated document IDs to delete
                      --ids-file, -f FILE        File containing document IDs (one per line)

                    Optional:
                      --help, -h                 Show this help

                    Examples:
                      PlaidCLI delete -i ~/.plaid/my_index -d 5 12 23 45
                      PlaidCLI delete -i ~/.plaid/my_index -f to_delete.txt
                    """)
                return

            default:
                print("⚠️  Warning: Unknown argument '\(arg)'")
            }
            idx += 1
        }

        // Validate required parameters
        guard let indexPath = indexPath else {
            print("❌ Error: --index-path is required\n")
            print("Usage: PlaidCLI delete --index-path <PATH> --doc-ids <ID1> <ID2> ...")
            print("       PlaidCLI delete --index-path <PATH> --ids-file <FILE>")
            print("Run 'PlaidCLI delete --help' for more information")
            return
        }

        let indexURL = URL(fileURLWithPath: indexPath)

        // Verify index exists
        guard FileManager.default.fileExists(atPath: indexURL.path) else {
            print("❌ Error: Index not found at '\(indexPath)'")
            return
        }

        // Collect all IDs
        var allIds = docIds
        if let filePath = idsFromFile {
            guard FileManager.default.fileExists(atPath: filePath) else {
                print("❌ Error: IDs file not found at '\(filePath)'")
                return
            }

            do {
                let content = try String(contentsOfFile: filePath, encoding: .utf8)
                let fileIds = content.split(separator: "\n")
                    .map { $0.trimmingCharacters(in: .whitespaces) }
                    .compactMap { Int($0) }
                allIds.append(contentsOf: fileIds)
                print("📄 Loaded \(fileIds.count) document IDs from file")
            } catch {
                print("❌ Error reading IDs file: \(error.localizedDescription)")
                return
            }
        }

        guard !allIds.isEmpty else {
            print("❌ Error: No document IDs provided\n")
            print("Usage: PlaidCLI delete --index-path <PATH> --doc-ids <ID1> <ID2> ...")
            print("       PlaidCLI delete --index-path <PATH> --ids-file <FILE>")
            print("Run 'PlaidCLI delete --help' for more information")
            return
        }

        // Remove duplicates
        let uniqueIds = Array(Set(allIds)).sorted()

        print("╔══════════════════════════════════════════════════════════════════════╗")
        print("║  Plaid Index Deletion                                                ║")
        print("╚══════════════════════════════════════════════════════════════════════╝\n")

        print("📂 Index: \(indexPath)")
        print("🗑️  Documents to delete: \(uniqueIds.count)")
        if uniqueIds.count <= 10 {
            print("   IDs: \(uniqueIds.map { String($0) }.joined(separator: ", "))")
        } else {
            print("   IDs: \(uniqueIds.prefix(10).map { String($0) }.joined(separator: ", ")), ...")
        }
        print()

        // Perform deletion
        print("🗑️  Deleting documents...")
        do {
            try Plaid.delete(
                indexURL: indexURL,
                subset: uniqueIds
            )
            print("✅ Deletion complete!")
            print("\n💾 Updated index: \(indexPath)")
            print("   Removed \(uniqueIds.count) document(s)")

        } catch {
            print("❌ Error deleting documents: \(error.localizedDescription)")
            throw error
        }
    }
}
