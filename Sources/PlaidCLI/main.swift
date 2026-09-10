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

    // MARK: - Backend Selection

    /// The vector-engine backend behind every index operation: the Rust
    /// `next-plaid` engine. Conforms to `SearchBackend`, so the commands below
    /// are engine-agnostic.
    private static func makeBackend() -> SearchBackend {
        RustSearchBackend()
    }

    // MARK: - Model Selection

    enum CLIModel: String, CaseIterable {
        case lfm2 = "lfm2"
        case mxbai = "mxbai"

        var modelId: String {
            switch self {
            case .lfm2:
                return "LiquidAI/LFM2-ColBERT-350M"
            case .mxbai:
                return "mixedbread-ai/mxbai-edge-colbert-v0-32m"
            }
        }

        var embeddingDimension: Int {
            switch self {
            case .lfm2:
                return 128
            case .mxbai:
                return 64
            }
        }

        var displayName: String {
            switch self {
            case .lfm2:
                return "LFM2-ColBERT (128-dim)"
            case .mxbai:
                return "MXBAI-Edge (64-dim)"
            }
        }

        static func from(string: String) -> CLIModel? {
            let normalized = string.lowercased().replacingOccurrences(of: "-", with: "")
            switch normalized {
            case "lfm2", "lfm2colbert":
                return .lfm2
            case "mxbai", "mxbaiedge", "mxbaiedgecolbert":
                return .mxbai
            default:
                return nil
            }
        }
    }

    /// Creates the appropriate ColBERT generator for the given model
    private static func createGenerator(
        model: CLIModel,
        tokenizer: ColbertTokenizer
    ) async throws -> ColbertEmbeddingGenerator {
        // The Core ML encoder is fetched from the Hugging Face Hub on first use and cached.
        switch model {
        case .lfm2:
            return try await LFM2ColbertEmbeddingGenerator.download(tokenizer: tokenizer)
        case .mxbai:
            return try await MXBAIEdgeColbertEmbeddingGenerator.download(tokenizer: tokenizer)
        }
    }

    static func main() async throws {
        let args = Array(CommandLine.arguments.dropFirst())
        guard let command = args.first else {
            printUsage()
            return
        }
        let commandArgs = Array(args.dropFirst())

        switch command.lowercased() {
        case "update":
            try await runUserUpdate(arguments: commandArgs)
        case "remap-test":
            try runRemapTest()
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
                           Options: --model [lfm2|mxbai], --pretrained MODEL_ID, --top-k N, --nbits N, --keep-index, --index-name NAME
                           Examples:
                             demo --query "..." --docs "text1" "text2" --model mxbai
                             demo --query "..." --files "doc1.txt" "doc2.md" --keep-index
                             demo --query "..." --index-path /path/to/index

              update       Add new documents to an existing Plaid index.
                           Usage: update --index-path PATH --files FILE... [OPTIONS]
                           Options: --model [lfm2|mxbai], --pretrained MODEL_ID, --batch-size N
                           Example: update -i ~/.plaid/my_index -f doc1.txt doc2.txt --model lfm2

              delete       Remove documents from an existing Plaid index by document IDs.
                           Usage: delete --index-path PATH (--doc-ids ID... | --ids-file FILE)
                           Examples:
                             delete -i ~/.plaid/my_index -d 5 12 23 45
                             delete -i ~/.plaid/my_index -f to_delete.txt

              remap-test   Offline end-to-end check of the backend (no model download):
                           create → search → update → middle-delete → suffix-delete, asserting the
                           delete-renumber remap.

              tokenize     Tokenize a string using a pretrained tokenizer and print tokens/ids.
                           Usage: tokenize [--query|--doc] [--model MODEL | --pretrained MODEL_ID] TEXT
                           Default: LFM2-ColBERT
                           Example: tokenize "test" or tokenize --model mxbai "test"

              similarity   Encode a query/document pair with the Core ML ColBERT model and print their score.
                           Usage: similarity --query "..." --doc "..." [--model [lfm2|mxbai] | --pretrained MODEL_ID]

            Available Models:
              lfm2         LFM2-ColBERT (128-dimensional, default)
              mxbai        MXBAI-Edge (64-dimensional, faster)

            For command-specific help, run: \(exe) <command> --help
            """)
    }

    /// Offline end-to-end exercise of the active `SearchBackend`: create, search,
    /// append, then a middle delete and a suffix delete — asserting the engine's
    /// delete-renumber remap. Uses one-hot vectors so a query along an axis maps
    /// to a known document id; needs no model download or network.
    private static func runRemapTest() throws {
        let dim = 16
        // nbits must divide 8 (Rust). nbits=4 → the engine builds 16 quantization
        // buckets via its own k-means; one-hot docs quantize losslessly.
        let nbits = 4

        func oneHot(_ axis: Int, tokens: Int = 4) -> [[Float]] {
            var row = [Float](repeating: 0, count: dim)
            row[axis] = 1
            return Array(repeating: row, count: tokens)
        }
        func query(axis: Int) -> [[[Float]]] { [oneHot(axis, tokens: 1)] }
        let searchParams = SearchParameters(
            batchSize: 2000, nFullScores: 4096, topK: 1, nIvfProbe: 1024)

        var failures = 0
        func expect(_ got: Int?, _ want: Int, _ what: String) {
            if got == want {
                print("  ✅ \(what): id \(want)")
            } else {
                print("  ❌ \(what): expected \(want), got \(String(describing: got))")
                failures += 1
            }
        }

        let backend = makeBackend()
        let indexURL = defaultIndexURL(named: "remap_test")
        try resetIndexDirectory(at: indexURL)

        func topId(axis: Int) throws -> Int? {
            try backend.loadAndSearch(
                indexURL: indexURL, queries: query(axis: axis),
                searchParameters: searchParams, showProgress: false,
                preloadIndex: false, subset: nil
            ).first?.passageIds.first
        }

        print("╔══════════════════════════════════════════════════════════════════════╗")
        print("║  Backend Remap Test (create → update → middle/suffix delete)         ║")
        print("╚══════════════════════════════════════════════════════════════════════╝\n")
        print("📂 Index: \(indexURL.path)\n")

        // 1. Create five docs along axes 0…4 (id == axis).
        let docs = (0 ..< 5).map { oneHot($0) }
        try backend.create(
            indexURL: indexURL, embeddingDim: dim, nbits: nbits,
            embeddings: docs, batchSize: 64, seed: 42)
        print("① Created 5 docs along axes 0…4:")
        for axis in 0 ..< 5 { expect(try topId(axis: axis), axis, "axis \(axis)") }

        // 2. Append docs along axes 5 and 6 → ids 5, 6 (existing ids unchanged).
        let newIds = try backend.update(
            indexURL: indexURL, embeddings: [oneHot(5), oneHot(6)], batchSize: 64)
        print("\n② Appended axes 5,6 → new ids \(newIds):")
        if newIds != [5, 6] {
            print("  ❌ expected new ids [5, 6], got \(newIds)")
            failures += 1
        } else {
            print("  ✅ new ids [5, 6]")
        }
        expect(try topId(axis: 6), 6, "axis 6")

        // 3. Middle delete: remove id 2 (axis-2 doc). Survivors above shift down 1.
        let mid = try backend.delete(indexURL: indexURL, subset: [2])
        print("\n③ Middle-deleted id 2 → removed \(mid.deletedIdsSorted):")
        if mid.deletedIdsSorted != [2] {
            print("  ❌ expected removed [2]")
            failures += 1
        } else {
            print("  ✅ removed [2]")
        }
        print("   Expect renumber: axis3 3→2, axis4 4→3, axis5 5→4, axis6 6→5")
        expect(try topId(axis: 3), 2, "axis 3 after middle delete")
        expect(try topId(axis: 4), 3, "axis 4 after middle delete")
        expect(try topId(axis: 6), 5, "axis 6 after middle delete")

        // 4. Suffix delete: remove the current last id (5 = axis-6 doc). Ids stable.
        let suffix = try backend.delete(indexURL: indexURL, subset: [5])
        print("\n④ Suffix-deleted id 5 → removed \(suffix.deletedIdsSorted):")
        if suffix.deletedIdsSorted != [5] {
            print("  ❌ expected removed [5]")
            failures += 1
        } else {
            print("  ✅ removed [5]")
        }
        print("   Expect ids unchanged for survivors below 5:")
        expect(try topId(axis: 3), 2, "axis 3 after suffix delete (stable)")
        expect(try topId(axis: 5), 4, "axis 5 after suffix delete (stable)")

        try? FileManager.default.removeItem(at: indexURL)

        print("")
        if failures == 0 {
            print("✅ remap-test passed — create/update/delete + renumber all correct.")
        } else {
            print("❌ remap-test FAILED with \(failures) mismatch(es).")
            fflush(stdout)  // flush before the abort so the report above is visible
            throw NSError(
                domain: "PlaidCLI", code: 99,
                userInfo: [NSLocalizedDescriptionKey: "remap-test failed (\(failures) mismatches)"])
        }
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
        let tokenizer = try await ColbertTokenizer.from(pretrained: finalModelId)
        printTokenizerOutput(input: input, isQuery: isQuery, tokenizer: tokenizer)
    }

    private static func printTokenizerOutput(
        input: String, isQuery: Bool, tokenizer: ColbertTokenizer
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
        var selectedModel: CLIModel?

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
            case "--model", "-m":
                idx += 1
                guard idx < arguments.count else {
                    print("Missing value after --model\n")
                    return
                }
                if let model = CLIModel.from(string: arguments[idx]) {
                    selectedModel = model
                } else {
                    print("Unknown model: \(arguments[idx]). Use 'lfm2' or 'mxbai'\n")
                    return
                }
            case "--pretrained", "-p":
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
            print("Usage: similarity --query \"...\" --doc \"...\" [--model [lfm2|mxbai]]\n")
            return
        }

        // Determine model: --model flag takes precedence over --pretrained
        let model = selectedModel ?? .lfm2
        let finalModelId = modelId ?? model.modelId

        print("Loading tokenizer/model: \(finalModelId) (\(model.displayName))")
        let tokenizer = try await ColbertTokenizer.from(pretrained: finalModelId)
        let generator = try await createGenerator(model: model, tokenizer: tokenizer)
        let chunker = TokenSplitter(withTokenizer: tokenizer)
        let colbert = ColbertModel(
            generator: generator,
            configuration: .init(
                batchSize: 1,
                embeddingDimension: model.embeddingDimension,
                queryLength: tokenizer.maxSequenceLength,
                documentLength: tokenizer.maxSequenceLength
            ),
            chunker: chunker
        )

        print("=== query embedding ===")
        let queryEmbedding = try colbert.encode(sentence: queryText, isQuery: true)
        print(EmbeddingFormatting.formatEmbeddingsPreview(queryEmbedding))

        print("=== document embedding ===")
        let documentEmbedding = try colbert.encode(sentence: documentText, isQuery: false)
        print(EmbeddingFormatting.formatEmbeddingsPreview(documentEmbedding))

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
        var selectedModel: CLIModel?
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
            case "--model", "-m":
                idx += 1
                guard idx < arguments.count else {
                    print("Missing value after --model\n")
                    return
                }
                if let model = CLIModel.from(string: arguments[idx]) {
                    selectedModel = model
                } else {
                    print("Unknown model: \(arguments[idx]). Use 'lfm2' or 'mxbai'\n")
                    return
                }
            case "--pretrained", "-p":
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
                selectedModel: selectedModel,
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

        // Determine model: --model flag takes precedence over --pretrained
        let model = selectedModel ?? .lfm2
        let finalModelId = modelId ?? model.modelId

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
        print("⚙️  Loading ColBERT model: \(finalModelId) (\(model.displayName))...")
        let tokenizer = try await ColbertTokenizer.from(pretrained: finalModelId)
        let generator = try await createGenerator(model: model, tokenizer: tokenizer)
        let chunker = TokenSplitter(withTokenizer: tokenizer)
        let colbert = ColbertModel(
            generator: generator,
            configuration: .init(
                batchSize: 1,
                embeddingDimension: model.embeddingDimension,
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
            print(EmbeddingFormatting.formatEmbeddingsPreview(embedding))
        }
        print("✅ All documents encoded\n")

        // Create index
        let indexSuffix = indexName ?? "demo_\(UUID().uuidString)"
        let indexURL = defaultIndexURL(named: indexSuffix)
        try resetIndexDirectory(at: indexURL)

        print("📦 Creating Plaid index...")
        print("  Index location: \(indexURL.path)")
        print("  Model: \(model.displayName)")
        print("  Embedding dimension: \(model.embeddingDimension)")
        print("  nbits: \(nbits)")

        let backend = makeBackend()
        try backend.create(
            indexURL: indexURL,
            embeddingDim: model.embeddingDimension,
            nbits: nbits,
            embeddings: documentEmbeddings,
            batchSize: 64,
            seed: 42
        )
        print("✅ Index created\n")

        // Encode query
        print("🔍 Encoding query with ColBERT...")
        let queryEmbedding = try colbert.encode(sentence: queryText, isQuery: true)
        print("  Query: \(queryEmbedding.count) tokens × \(queryEmbedding.first?.count ?? 0) dims")
        print("✅ Query encoded\n")
        print(EmbeddingFormatting.formatEmbeddingsPreview(queryEmbedding))

        // Search
        print("🔎 Searching index...")
        let params = SearchParameters(
            batchSize: 1,
            nFullScores: finalDocumentTexts.count,
            topK: min(topK, finalDocumentTexts.count),
            nIvfProbe: min(8, 1 << nbits),
            logTiming: true
        )

        let results = try backend.loadAndSearch(
            indexURL: indexURL,
            queries: [queryEmbedding],
            searchParameters: params,
            showProgress: false,
            preloadIndex: false,
            subset: nil
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

    /// Runs demo command with an existing index (query-only mode)
    private static func runDemoWithExistingIndex(
        indexPath: String,
        queryText: String,
        selectedModel: CLIModel?,
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

        // Determine model: --model flag takes precedence over --pretrained
        let model = selectedModel ?? .lfm2
        let finalModelId = modelId ?? model.modelId

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
        print("⚙️  Loading ColBERT model: \(finalModelId) (\(model.displayName))...")
        let tokenizer = try await ColbertTokenizer.from(pretrained: finalModelId)
        let generator = try await createGenerator(model: model, tokenizer: tokenizer)
        let chunker = TokenSplitter(withTokenizer: tokenizer)
        let colbert = ColbertModel(
            generator: generator,
            configuration: .init(
                batchSize: 1,
                embeddingDimension: model.embeddingDimension,
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

        let backend = makeBackend()
        let results = try backend.loadAndSearch(
            indexURL: indexURL,
            queries: [queryEmbedding],
            searchParameters: params,
            showProgress: false,
            preloadIndex: false,
            subset: nil
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

    // MARK: - User-Facing CLI Commands

    private static func runUserUpdate(arguments: [String]) async throws {
        var indexPath: String?
        var documentFiles: [String] = []
        var pretrainedModel: String?
        var selectedModel: CLIModel?
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

            case "--model", "-m":
                idx += 1
                guard idx < arguments.count else {
                    print("❌ Error: --model requires a value")
                    return
                }
                if let model = CLIModel.from(string: arguments[idx]) {
                    selectedModel = model
                } else {
                    print("❌ Error: Unknown model '\(arguments[idx])'. Use 'lfm2' or 'mxbai'")
                    return
                }

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
                      --model, -m MODEL          Model to use: lfm2 (default) or mxbai
                      --pretrained, -p MODEL     HuggingFace model ID (overrides --model)
                      --batch-size, -b SIZE      Batch size for encoding (default: 64)
                      --help, -h                 Show this help

                    Examples:
                      PlaidCLI update -i ~/.plaid/my_index -f doc1.txt doc2.txt
                      PlaidCLI update -i ~/.plaid/my_index -f new_doc.txt --model mxbai
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

        // Determine model: --model flag takes precedence over --pretrained
        let model = selectedModel ?? .lfm2
        let finalModelId = pretrainedModel ?? model.modelId

        print("╔══════════════════════════════════════════════════════════════════════╗")
        print("║  Plaid Index Update                                                  ║")
        print("╚══════════════════════════════════════════════════════════════════════╝\n")

        print("📂 Index: \(indexPath)")
        print("📄 Documents to add: \(documentFiles.count)\n")

        // Load ColBERT model
        print("⚙️  Loading ColBERT model: \(finalModelId) (\(model.displayName))...")
        let tokenizer = try await ColbertTokenizer.from(pretrained: finalModelId)
        let generator = try await createGenerator(model: model, tokenizer: tokenizer)
        let chunker = TokenSplitter(withTokenizer: tokenizer)
        let colbert = ColbertModel(
            generator: generator,
            configuration: .init(
                embeddingDimension: model.embeddingDimension,
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
            let newIds = try makeBackend().update(
                indexURL: indexURL,
                embeddings: allEmbeddings,
                batchSize: batchSize
            )
            print("✅ Index updated successfully!")
            print("\n💾 Updated index: \(indexPath)")
            print("   Added \(allEmbeddings.count) document(s)")
            if !newIds.isEmpty {
                print("   Assigned document IDs: \(newIds.first!)…\(newIds.last!)")
            }

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
            let outcome = try makeBackend().delete(
                indexURL: indexURL,
                subset: uniqueIds
            )
            print("✅ Deletion complete!")
            print("\n💾 Updated index: \(indexPath)")
            print("   Removed \(outcome.deletedIdsSorted.count) document(s)")
            // The engine compacts survivors: any id above a deleted id shifts down
            // by the count of deleted ids below it. Callers with an external id map
            // must replay this using deletedIdsSorted.
            if !outcome.deletedIdsSorted.isEmpty {
                print("   Removed internal IDs (sorted): \(outcome.deletedIdsSorted)")
                print(
                    "   ⚠️  Survivors were renumbered: new = old − (count of removed IDs below old)."
                )
            }

        } catch {
            print("❌ Error deleting documents: \(error.localizedDescription)")
            throw error
        }
    }
}
