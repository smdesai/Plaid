import Foundation
import Plaid

/// Core search engine that orchestrates ColBERT indexing and searching
@MainActor
class SearchEngine: ObservableObject {
    @Published var isIndexing = false
    @Published var indexingProgress: Double = 0.0
    @Published var currentDocument = ""
    @Published var hasIndex = false
    @Published var indexState: IndexState?
    @Published var errorMessage: String?
    @Published var currentModel: ModelType?
    @Published var modelReady = false

    private let indexURL: URL
    private var tokenizer: ColbertTokenizer?
    private var colbert: ColbertModel?

    /// Per-chunk text store, tying each vector `doc_id` to its text. Backed by
    /// the SQLite `metadata.db` inside `indexURL` (via `NextPlaidFFI`). Injected
    /// like `backend` for testability; the default builds a
    /// `NextPlaidMetadataProvider` once `indexURL` is resolved.
    private let metadataProvider: any PlaidMetadataProvider

    /// Index name for the metadata store (the app is 1:1 on a single index).
    private let indexName = "default"

    private var embeddingDim: Int = 128  // Will be set based on model
    private let nbits = 2

    /// Vector-search engine seam. Defaults to the Rust `next-plaid` engine (via
    /// UniFFI); injectable so tests or a legacy fallback can swap it out.
    private let backend: SearchBackend

    private static let currentModelKey = "currentModel"

    /// UserDefaults key for the Core ML encode batch size (tunable in Settings
    /// for on-device throughput experiments).
    nonisolated static let encodeBatchSizeKey = "encodeBatchSize"
    /// Encode batch size used when the user hasn't chosen one.
    nonisolated static let defaultEncodeBatchSize = 32
    /// Batch sizes offered in Settings for sweeping encode throughput.
    nonisolated static let encodeBatchSizeOptions = [8, 16, 32, 48, 64]

    /// The encode batch size currently selected in Settings, or the default.
    /// Read fresh from `UserDefaults` so a change takes effect on the next index
    /// without reloading the model.
    nonisolated static var encodeBatchSize: Int {
        let stored = UserDefaults.standard.integer(forKey: encodeBatchSizeKey)
        return stored > 0 ? stored : defaultEncodeBatchSize
    }

    /// UserDefaults key for how many search results (top-k) to return.
    nonisolated static let resultCountKey = "searchResultCount"
    /// Result count used when the user hasn't chosen one.
    nonisolated static let defaultResultCount = 3
    /// Result counts offered in Settings.
    nonisolated static let resultCountOptions = [3, 5, 10, 20]

    /// The number of search results currently selected in Settings, or the
    /// default. Read fresh from `UserDefaults` so a change applies to the next
    /// search immediately.
    nonisolated static var resultCount: Int {
        let stored = UserDefaults.standard.integer(forKey: resultCountKey)
        return stored > 0 ? stored : defaultResultCount
    }

    /// - Parameters:
    ///   - backend: vector-search engine seam (defaults to the Rust engine).
    ///   - metadataProvider: factory for the per-chunk text store, given the
    ///     resolved `indexURL`. Defaults to the SQLite-backed
    ///     `NextPlaidMetadataProvider`; injectable so tests can substitute one.
    init(
        backend: SearchBackend = NextPlaidBackend(),
        metadataProvider: (URL) -> any PlaidMetadataProvider = {
            NextPlaidMetadataProvider(indexURL: $0)
        }
    ) {
        self.backend = backend

        // Set up index directory in Application Support
        let appSupport = FileManager.default.urls(
            for: .documentDirectory,
            in: .userDomainMask
        ).first!

        self.indexURL = appSupport.appendingPathComponent("PlaidIndex", isDirectory: true)

        // Create directory if needed
        try? FileManager.default.createDirectory(
            at: indexURL,
            withIntermediateDirectories: true
        )

        self.metadataProvider = metadataProvider(indexURL)
    }

    /// Whether the index currently holds searchable chunks. Views gate their
    /// search UI and decide whether to refresh or clear results after a delete
    /// on this, rather than reaching into `indexState`'s counters directly.
    var hasSearchableContent: Bool {
        (indexState?.totalDocuments ?? 0) > 0
    }

    /// Initialize the ColBERT model with specified model type
    func initialize(with model: ModelType) async throws {
        print("🔧 Initializing SearchEngine with \(model.displayName)...")

        // Load tokenizer
        print("📥 Loading tokenizer...")
        self.tokenizer = try await ColbertTokenizer.from(pretrained: model.modelId)

        guard let tokenizer = tokenizer else {
            throw SearchEngineError.modelLoadFailed
        }

        // Create appropriate embedding generator based on model type
        let generator: ColbertEmbeddingGenerator
        switch model {
        case .lfm2:
            print(
                "📦 Initializing LFM2 embedding generator (downloads the Core ML model on first use)..."
            )
            generator = try await LFM2ColbertEmbeddingGenerator.download(tokenizer: tokenizer)
        case .mxbaiEdge:
            print(
                "📦 Initializing MXBAI-Edge embedding generator (downloads the Core ML model on first use)..."
            )
            generator = try await MXBAIEdgeColbertEmbeddingGenerator.download(tokenizer: tokenizer)
        }

        // Use SentenceBoundarySplitter for better semantic coherence in chunks
        // Preserves complete sentences instead of splitting mid-sentence
        let chunker = SentenceBoundarySplitter(withTokenizer: tokenizer)

        // Set embedding dimension based on model
        self.embeddingDim = model.embeddingDimension

        self.colbert = ColbertModel(
            generator: generator,
            configuration: .init(
                batchSize: SearchEngine.encodeBatchSize,
                embeddingDimension: model.embeddingDimension,
                queryLength: model.querySequenceLength,
                documentLength: model.documentSequenceLength
            ),
            chunker: chunker
        )

        self.currentModel = model
        UserDefaults.standard.set(model.rawValue, forKey: SearchEngine.currentModelKey)

        print("✅ Model loaded successfully (embedding dim: \(model.embeddingDimension))")

        // Check for existing index
        self.hasIndex = checkForExistingIndex()

        if hasIndex {
            do {
                self.indexState = try loadIndexState()
                print("📚 Loaded existing index with \(indexState?.totalDocuments ?? 0) documents")
            } catch {
                print("⚠️  Error loading index state: \(error)")
                self.hasIndex = false
            }
        }
    }

    /// Initialize with saved model or default
    func initialize() async throws {
        // Load saved model preference or default to MXBAI-Edge
        let model: ModelType
        if let savedRawValue = UserDefaults.standard.string(forKey: SearchEngine.currentModelKey),
            let savedModel = ModelType(rawValue: savedRawValue)
        {
            model = savedModel
        } else {
            model = .mxbaiEdge
        }

        try await initialize(with: model)
    }

    /// Check if an index exists on disk
    private func checkForExistingIndex() -> Bool {
        let metadataPath = indexURL.appendingPathComponent("metadata.json")
        let statePath = indexURL.appendingPathComponent("index_state.json")
        return FileManager.default.fileExists(atPath: metadataPath.path)
            && FileManager.default.fileExists(atPath: statePath.path)
    }

    /// One encoded chunk handed from the encoder stage to the indexer stage.
    private struct EncodedChunkBatchItem: Sendable {
        let documentName: String
        let chunkIndex: Int
        let text: String
        let embeddings: [[Float]]
    }

    /// Create a new index from documents
    /// Each document is chunked and each chunk becomes a separate searchable unit
    func createIndex(documents: [Document]) async throws {
        guard var colbert = colbert else {
            throw SearchEngineError.modelNotInitialized
        }
        // Pick up the current encode batch-size setting so a change made in
        // Settings takes effect on the next index without reloading the model.
        colbert.batchSize = SearchEngine.encodeBatchSize

        print("🏗️  Creating index from \(documents.count) documents...")

        isIndexing = true
        indexingProgress = 0.0
        errorMessage = nil

        // Capture the pieces the background pipeline needs so no heavy work runs
        // on the main actor (it stays free to publish progress + keep the UI live).
        let backend = self.backend
        let metadataProvider = self.metadataProvider
        let indexURL = self.indexURL
        let indexName = self.indexName
        let embeddingDim = self.embeddingDim
        let nbits = self.nbits

        do {
            let totals = try await Self.runIndexingPipeline(
                documents: documents,
                colbert: colbert,
                backend: backend,
                metadataProvider: metadataProvider,
                indexURL: indexURL,
                indexName: indexName,
                embeddingDim: embeddingDim,
                nbits: nbits,
                batchChunkThreshold: 1024,
                onProgress: { [weak self] docIndex, docCount, filename in
                    Task { @MainActor in
                        guard let self else { return }
                        self.currentDocument = filename
                        self.indexingProgress = Double(docIndex) / Double(max(docCount, 1))
                    }
                }
            )

            // The SQLite metadata store is the source of truth for per-chunk
            // text, so the persisted state carries only the summary — keeping
            // this dict empty avoids holding every chunk's text in RAM.
            self.indexState = IndexState(
                documents: [:],
                createdAt: Date(),
                lastModified: Date(),
                totalDocuments: totals.totalChunks,
                totalEmbeddings: totals.totalEmbeddings
            )
            try saveIndexState()

            isIndexing = false
            hasIndex = true
            indexingProgress = 1.0

            print("✅ Index created successfully!")
            print(
                "   📊 \(documents.count) documents → \(totals.totalChunks) chunks → \(totals.totalEmbeddings) embeddings"
            )
        } catch {
            isIndexing = false
            throw error
        }
    }

    /// Runs the encode → index → store pipeline off the main actor as two
    /// overlapping stages joined by a bounded FIFO:
    ///
    ///  - **Producer (encoder):** walks documents in order, chunks + encodes each
    ///    (CoreML / ANE), accumulates a batch, and `enqueue`s it. Backpressure
    ///    from the bounded queue caps peak memory to ~2 batches.
    ///  - **Consumer (indexer):** dequeues batches FIFO, assigns the running
    ///    `plaidDocId` (so ids stay contiguous and in lockstep with the
    ///    `create`/`update` order), builds the index (Rust FFI, CPU) and writes
    ///    the per-chunk text to the SQLite metadata store.
    ///
    /// Overlapping the two stages lets ANE/GPU encoding run while the previous
    /// batch is quantized + written to disk. Streaming keeps peak memory bounded:
    /// the Rust engine rebuilds from retained raw embeddings only while the index
    /// has ≤ `start_from_scratch` (999) chunks — a constant window — then switches
    /// to incremental appends.
    private nonisolated static func runIndexingPipeline(
        documents: [Document],
        colbert: ColbertModel,
        backend: SearchBackend,
        metadataProvider: any PlaidMetadataProvider,
        indexURL: URL,
        indexName: String,
        embeddingDim: Int,
        nbits: Int,
        batchChunkThreshold: Int,
        onProgress:
            @Sendable @escaping (_ docIndex: Int, _ docCount: Int, _ filename: String) -> Void
    ) async throws -> (totalChunks: Int, totalEmbeddings: Int) {
        let queue = BoundedBatchQueue<[EncodedChunkBatchItem]>(capacity: 2)

        return try await withThrowingTaskGroup(
            of: (totalChunks: Int, totalEmbeddings: Int)?.self
        ) { group in
            // Producer: encode documents into bounded batches.
            group.addTask {
                do {
                    var batch: [EncodedChunkBatchItem] = []
                    batch.reserveCapacity(batchChunkThreshold)

                    for (docIndex, doc) in documents.enumerated() {
                        onProgress(docIndex, documents.count, doc.filename)
                        print(
                            "📄 Encoding document [\(docIndex + 1)/\(documents.count)]: \(doc.filename)"
                        )

                        let chunked = try colbert.encodeDocument(doc.text)
                        print(
                            "  ✅ \(chunked.chunks.count) chunks, \(chunked.totalEmbeddingCount) total embeddings"
                        )

                        for chunk in chunked.chunks {
                            batch.append(
                                EncodedChunkBatchItem(
                                    documentName: doc.filename,
                                    chunkIndex: chunk.chunkIndex,
                                    text: chunk.text,
                                    embeddings: chunk.embeddings
                                ))

                            if batch.count >= batchChunkThreshold {
                                let outgoing = batch
                                batch.removeAll(keepingCapacity: true)
                                let accepted = await queue.enqueue(outgoing)
                                if !accepted { return nil }  // consumer failed; stop early
                            }
                        }
                    }

                    if !batch.isEmpty {
                        _ = await queue.enqueue(batch)
                    }
                    await queue.finish()
                    return nil
                } catch {
                    await queue.fail(error)
                    throw error
                }
            }

            // Consumer: build the index + write metadata in dequeue order.
            group.addTask {
                do {
                    var didCreate = false
                    var currentPlaidDocId = 0
                    var totalChunks = 0
                    var totalEmbeddings = 0

                    while let batch = await queue.dequeue() {
                        let embeddings = batch.map { $0.embeddings }

                        if !didCreate {
                            print(
                                "💾 Creating Plaid index (first batch of \(batch.count) chunks)...")
                            // The Rust engine computes its own k-means centroids.
                            try backend.create(
                                indexURL: indexURL,
                                embeddingDim: embeddingDim,
                                nbits: nbits,
                                embeddings: embeddings,
                                batchSize: 64,
                                seed: 42
                            )
                            didCreate = true
                        } else {
                            print("➕ Appending \(batch.count) chunks to Plaid index...")
                            _ = try backend.update(
                                indexURL: indexURL, embeddings: embeddings, batchSize: 64)
                        }

                        var metadata:
                            [(
                                plaidDocId: Int, documentName: String, chunkText: String,
                                chunkIndex: Int, embeddingCount: Int, filePath: String?
                            )] = []
                        metadata.reserveCapacity(batch.count)
                        for item in batch {
                            metadata.append(
                                (
                                    plaidDocId: currentPlaidDocId,
                                    documentName: item.documentName,
                                    chunkText: item.text,
                                    chunkIndex: item.chunkIndex,
                                    embeddingCount: item.embeddings.count,
                                    filePath: nil
                                ))
                            currentPlaidDocId += 1
                            totalChunks += 1
                            totalEmbeddings += item.embeddings.count
                        }
                        try await metadataProvider.registerDocuments(metadata, indexName: indexName)
                    }

                    if let failure = await queue.failure { throw failure }
                    guard didCreate else { throw SearchEngineError.noEmbeddings }
                    return (totalChunks, totalEmbeddings)
                } catch {
                    await queue.fail(error)
                    throw error
                }
            }

            var outcome: (totalChunks: Int, totalEmbeddings: Int)?
            for try await value in group {
                if let value { outcome = value }
            }
            guard let outcome else { throw SearchEngineError.noEmbeddings }
            return outcome
        }
    }

    /// Search the index using Plaid's native MaxSim (late interaction) scoring
    /// This is the proper ColBERT approach - each query token finds its best matching document token
    func search(query: String, topK: Int = 5) async throws -> [SearchResult] {
        guard let colbert = colbert else {
            throw SearchEngineError.modelNotInitialized
        }

        guard let indexState = indexState else {
            throw SearchEngineError.noIndex
        }

        print("🔍 Searching for: \(query)")

        // Encode query
        let queryEmbedding = try colbert.encode(sentence: query, isQuery: true)
        print("  ✅ Query encoded: \(queryEmbedding.count) embeddings")

        // Use Plaid's native MaxSim scoring - this is the correct ColBERT late interaction
        return try await maxSimSearch(
            queryEmbedding: queryEmbedding,
            indexState: indexState,
            topK: topK
        )
    }

    /// MaxSim search using Plaid's native late interaction scoring
    /// This is the proper ColBERT approach where each query token finds its best document token match
    private func maxSimSearch(
        queryEmbedding: [[Float]],
        indexState: IndexState,
        topK: Int
    ) async throws -> [SearchResult] {
        let startTime = DispatchTime.now()

        // Adaptive scoring parameters based on index size
        // IVF pre-filtering identifies candidates, then we do full MaxSim on top candidates
        let totalChunks = indexState.totalDocuments
        let (nFullScores, nIvfProbe) = adaptiveSearchParams(totalChunks: totalChunks, topK: topK)

        let params = SearchParameters(
            batchSize: 1,
            nFullScores: nFullScores,
            topK: topK,
            nIvfProbe: nIvfProbe,
            logTiming: false
        )

        let results = try backend.loadAndSearch(
            indexURL: indexURL,
            queries: [queryEmbedding],
            searchParameters: params,
            showProgress: false,
            preloadIndex: true,
            subset: nil
        )

        let searchTime =
            Double(DispatchTime.now().uptimeNanoseconds - startTime.uptimeNanoseconds) / 1_000_000

        guard let firstResult = results.first else {
            print("  ❌ No results returned")
            return []
        }

        print(
            "  🎯 MaxSim scored \(nFullScores)/\(totalChunks) chunks in \(String(format: "%.1f", searchTime))ms (nIvfProbe=\(nIvfProbe))"
        )

        // Enrich results with metadata
        let enrichedResults = try await firstResult.enriched(
            from: metadataProvider,
            indexName: indexName
        )

        print("  ✅ Found \(enrichedResults.count) results")

        // The engine's raw score is ColBERT MaxSim: a *sum* of the best per-token
        // cosine over the query tokens, so it scales with query length and is not
        // bounded to [0, 1]. Divide by the query-token count to get the average
        // best-per-token cosine (~[0, 1]) — a genuine, query-length-independent
        // relevance that the UI can render as 0–100%. Ranking is unchanged: every
        // score for this query is divided by the same constant.
        //
        // Queries are zero-padded to `queryLength` (see ColbertModel.normalizeAndPadQueries),
        // and those padding rows are zero vectors that contribute 0 to the MaxSim sum.
        // Dividing by the padded row count would deflate the score, so count only the
        // real (non-zero) query tokens that actually contribute.
        let realTokenCount = queryEmbedding.reduce(into: 0) { count, row in
            if row.contains(where: { $0 != 0 }) { count += 1 }
        }
        let queryTokenCount = Float(max(realTokenCount, 1))

        return enrichedResults.map { enriched in
            SearchResult(
                documentId: enriched.plaidDocId,
                filename: enriched.documentName,
                chunkIndex: enriched.chunkIndex,
                score: enriched.score / queryTokenCount,
                text: enriched.chunkText
            )
        }
    }

    /// Compute adaptive search parameters based on index size
    /// Balances search quality vs performance for different corpus sizes
    ///
    /// Note: With parallel decompression + intelligent candidate selection:
    /// - Decompression is 3.5x faster (parallel on 6 cores)
    /// - Centroid pre-scoring adds ~20ms overhead but ensures quality
    ///
    /// Performance characteristics (measured on 1679-chunk index with parallel):
    /// - 150 chunks → ~400ms (fast but some quality loss)
    /// - 200 chunks → ~530ms (good quality/speed balance) ✅
    /// - 250 chunks → ~660ms (excellent quality)
    /// - 300 chunks → ~800ms (near-perfect quality)
    private func adaptiveSearchParams(totalChunks: Int, topK: Int) -> (
        nFullScores: Int, nIvfProbe: Int
    ) {
        // For small indices, score everything for perfect recall
        if totalChunks <= 100 {
            return (totalChunks, min(16, totalChunks))
        }

        // For medium indices, balance quality and speed
        if totalChunks <= 500 {
            let nFullScores = min(200, totalChunks)
            let nIvfProbe = min(24, totalChunks / 4)
            return (max(topK * 40, nFullScores), max(8, nIvfProbe))
        }

        // For large indices (500-2000 chunks)
        // Use 250 chunks: excellent quality with sub-second search (~660ms)
        if totalChunks <= 2000 {
            let nFullScores = max(topK * 60, 250)  // Score 250 candidates for quality
            let nIvfProbe = min(32, max(24, totalChunks / 80))  // Probe 24-32 clusters
            return (nFullScores, nIvfProbe)
        }

        // For very large indices, cap at 300 chunks
        let nFullScores = max(topK * 80, 300)
        let nIvfProbe = min(48, max(32, totalChunks / 100))
        return (nFullScores, nIvfProbe)
    }

    /// Save index state to disk
    private func saveIndexState() throws {
        guard let indexState = indexState else { return }
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = .prettyPrinted
        let data = try encoder.encode(indexState)
        let url = indexURL.appendingPathComponent("index_state.json")
        try data.write(to: url)
        print("💾 Index state saved")
    }

    /// Load index state from disk
    private func loadIndexState() throws -> IndexState {
        let url = indexURL.appendingPathComponent("index_state.json")
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode(IndexState.self, from: data)
    }

    /// Index all text files in a directory
    /// Reads .txt, .md, .swift, .json, and other text files from the directory
    func indexDirectory(at directoryURL: URL) async throws {
        guard colbert != nil else {
            throw SearchEngineError.modelNotInitialized
        }

        print("📁 Indexing directory: \(directoryURL.path)")

        // Find all text files in the directory
        let supportedExtensions = [
            "txt", "md", "swift", "json", "xml", "html", "css", "js", "ts", "py", "rs", "go",
            "java", "kt", "c", "h", "cpp", "hpp",
        ]

        let fileManager = FileManager.default
        var documents: [Document] = []

        // Enumerate files in directory (non-recursive for now)
        let contents = try fileManager.contentsOfDirectory(
            at: directoryURL,
            includingPropertiesForKeys: [.isRegularFileKey, .contentTypeKey],
            options: [.skipsHiddenFiles]
        )

        for fileURL in contents {
            let resourceValues = try? fileURL.resourceValues(forKeys: [.isRegularFileKey])
            guard resourceValues?.isRegularFile == true else { continue }

            let ext = fileURL.pathExtension.lowercased()
            guard supportedExtensions.contains(ext) else { continue }

            do {
                let text = try String(contentsOf: fileURL, encoding: .utf8)
                guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                    continue
                }

                let filename = fileURL.lastPathComponent
                documents.append(Document(filename: filename, text: text))
                print("  📄 Found: \(filename) (\(text.count) chars)")
            } catch {
                print("  ⚠️  Skipping \(fileURL.lastPathComponent): \(error.localizedDescription)")
            }
        }

        guard !documents.isEmpty else {
            throw NSError(
                domain: "PlaidDemo",
                code: 2,
                userInfo: [
                    NSLocalizedDescriptionKey: "No supported text files found in directory"
                ]
            )
        }

        print("📚 Found \(documents.count) documents to index")

        // Create the index
        try await createIndex(documents: documents)
    }

    /// Index the demo/sample text files bundled in the app's Resources.
    ///
    /// Mirrors `indexDirectory(at:)` but sources documents from `Bundle.main`
    /// instead of a user-selected folder, so the app is usable out of the box
    /// without importing anything. The sample `.txt` files are flattened into
    /// the app bundle at build time, so we enumerate every bundled `.txt`.
    func indexBundledSamples() async throws {
        guard colbert != nil else {
            throw SearchEngineError.modelNotInitialized
        }

        print("📦 Indexing bundled sample documents")

        // Enumerate bundled .txt resources (sorted for deterministic ordering)
        let sampleURLs =
            (Bundle.main.urls(forResourcesWithExtension: "txt", subdirectory: nil) ?? [])
            .sorted { $0.lastPathComponent < $1.lastPathComponent }

        var documents: [Document] = []
        for fileURL in sampleURLs {
            do {
                let text = try String(contentsOf: fileURL, encoding: .utf8)
                guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                    continue
                }

                let filename = fileURL.lastPathComponent
                documents.append(Document(filename: filename, text: text))
                print("  📄 Sample: \(filename) (\(text.count) chars)")
            } catch {
                print("  ⚠️  Skipping \(fileURL.lastPathComponent): \(error.localizedDescription)")
            }
        }

        guard !documents.isEmpty else {
            throw NSError(
                domain: "PlaidDemo",
                code: 3,
                userInfo: [
                    NSLocalizedDescriptionKey: "No bundled sample documents were found"
                ]
            )
        }

        print("📚 Found \(documents.count) sample documents to index")

        // Create the index
        try await createIndex(documents: documents)
    }

    /// Delete the entire index and reset state
    func deleteIndex() async throws {
        print("🗑️ Deleting index...")

        // Remove the index directory
        if FileManager.default.fileExists(atPath: indexURL.path) {
            try FileManager.default.removeItem(at: indexURL)
            print("  ✅ Index directory removed")
        }

        // Removing the directory already took `metadata.db` with it; this is a
        // defensive no-op that keeps the provider seam honest.
        try await metadataProvider.deleteIndex(indexName: indexName)
        print("  ✅ Metadata store deleted")

        // Recreate empty directory
        try FileManager.default.createDirectory(
            at: indexURL,
            withIntermediateDirectories: true
        )

        // Reset state
        await MainActor.run {
            hasIndex = false
            indexState = nil
        }

        print("✅ Index deleted successfully")
    }

    /// Delete specific chunks (by their current Plaid `doc_id`) from the index.
    ///
    /// `backend.delete` re-sequences the vectors AND the co-located SQLite
    /// metadata with the identical `new_id = old_id − count(deleted < old_id)`
    /// rule, so surviving text stays tied to its embedding. Callers must re-run
    /// their search afterward: every id at or above a deleted id shifts down.
    ///
    /// Returns the number of chunks actually removed.
    @discardableResult
    func deleteDocuments(plaidDocIds: [Int]) async throws -> Int {
        guard !plaidDocIds.isEmpty else { return 0 }
        guard let state = indexState else { throw SearchEngineError.noIndex }

        // A 0-document index is not a valid state for the engine: `create`
        // rejects an empty corpus, and emptying an index via delete fails when it
        // reloads ("No data to merge"). So removing every remaining chunk means
        // deleting the index itself, not emptying it. Restrict the targets to the
        // live corpus (in-range, unique) and compare against the authoritative
        // store count; a full wipe routes to `deleteIndex()`.
        let currentCount = try await metadataProvider.documentCount(indexName: indexName)
        let targets = Set(plaidDocIds.filter { $0 >= 0 && $0 < currentCount })
        guard !targets.isEmpty else { return 0 }

        if targets.count >= currentCount {
            print("🗑️ Deleting all \(targets.count) chunk(s) — removing the index")
            try await deleteIndex()
            return targets.count
        }

        print("🗑️ Deleting \(plaidDocIds.count) chunk(s): \(plaidDocIds.sorted())")

        // Capture the embedding counts of the chunks we're about to remove
        // *before* deleting — once the rows are gone and ids renumber, the exact
        // per-chunk counts are unrecoverable. Summing these gives a truthful
        // post-delete embedding total instead of a proportional estimate.
        let doomed = try await metadataProvider.getDocuments(
            plaidDocIds: plaidDocIds, indexName: indexName)
        let embeddingCountById = Dictionary(
            doomed.map { ($0.plaidDocId, $0.embeddingCount) },
            uniquingKeysWith: { first, _ in first })

        let outcome = try backend.delete(indexURL: indexURL, subset: plaidDocIds)
        let deletedCount = outcome.deletedIdsSorted.count
        guard deletedCount > 0 else { return 0 }

        // Reconcile the persisted summary with the store's authoritative
        // post-renumber chunk count, and subtract the exact embeddings removed.
        let removedEmbeddings = outcome.deletedIdsSorted.reduce(0) {
            $0 + (embeddingCountById[$1] ?? 0)
        }
        let remainingChunks = try await metadataProvider.documentCount(indexName: indexName)
        let remainingEmbeddings = max(0, state.totalEmbeddings - removedEmbeddings)
        indexState = IndexState(
            documents: state.documents,
            createdAt: state.createdAt,
            lastModified: Date(),
            totalDocuments: remainingChunks,
            totalEmbeddings: remainingChunks == 0 ? 0 : remainingEmbeddings
        )
        hasIndex = remainingChunks > 0
        try saveIndexState()

        print("  ✅ Deleted \(deletedCount) chunk(s); \(remainingChunks) remaining")
        return deletedCount
    }

    /// Delete every chunk belonging to a document, grouped by `documentName`
    /// (the source filename). A document is chunked into many `plaidDocId`s at
    /// index time; this collects them all and deletes them in one renumbering
    /// pass. Returns the number of chunks removed.
    @discardableResult
    func deleteDocument(named documentName: String) async throws -> Int {
        guard indexState != nil else { throw SearchEngineError.noIndex }

        let ids = try await chunkIds(forDocumentNamed: documentName)
        guard !ids.isEmpty else {
            print("🗑️ No chunks found for document '\(documentName)'")
            return 0
        }

        print("🗑️ Deleting document '\(documentName)' (\(ids.count) chunk(s))")
        return try await deleteDocuments(plaidDocIds: ids)
    }

    /// Every distinct document currently in the store, one entry per source
    /// filename, with its chunk and embedding totals. Backs the "indexed
    /// documents" browser in the UI.
    func indexedDocuments() async throws -> [IndexedDocument] {
        try await metadataProvider.indexedDocuments(indexName: indexName)
    }

    /// The current Plaid `doc_id`s of every chunk whose `documentName` matches.
    /// The provider pushes this filter into SQLite (see
    /// `NextPlaidMetadataProvider.documentChunks`), so no full-store scan.
    private func chunkIds(forDocumentNamed documentName: String) async throws -> [Int] {
        let chunks = try await metadataProvider.documentChunks(
            named: documentName, indexName: indexName)
        return chunks.map { $0.plaidDocId }
    }
}
