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

    /// ObjectBox metadata provider for document storage
    private let metadataProvider = ObjectBoxMetadataProvider.shared

    /// Index name for ObjectBox metadata
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

    init(backend: SearchBackend = RustSearchBackend()) {
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
                queryLength: tokenizer.maxSequenceLength,
                documentLength: tokenizer.maxSequenceLength
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

            // ObjectBox is the source of truth for per-chunk metadata, so the
            // persisted state carries only the summary — keeping this dict empty
            // avoids holding every chunk's text for the whole corpus in RAM.
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
    ///    ObjectBox metadata.
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
        metadataProvider: ObjectBoxMetadataProvider,
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
                            // The Rust engine computes its own k-means and ignores
                            // `centroids`; the legacy engine consumes them.
                            let centroids = try generateCentroids(
                                from: embeddings, nbits: nbits, embeddingDim: embeddingDim)
                            try backend.create(
                                indexURL: indexURL,
                                embeddingDim: embeddingDim,
                                nbits: nbits,
                                embeddings: embeddings,
                                centroids: centroids,
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
                                chunkIndex: Int, filePath: String?
                            )] = []
                        metadata.reserveCapacity(batch.count)
                        for item in batch {
                            metadata.append(
                                (
                                    plaidDocId: currentPlaidDocId,
                                    documentName: item.documentName,
                                    chunkText: item.text,
                                    chunkIndex: item.chunkIndex,
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

        return enrichedResults.map { enriched in
            SearchResult(
                documentId: enriched.plaidDocId,
                filename: enriched.documentName,
                chunkIndex: enriched.chunkIndex,
                score: enriched.score,
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

    /// Generate centroids from embeddings using uniform sampling.
    ///
    /// `nonisolated static` so the background indexing pipeline can call it
    /// off the main actor. Only the legacy engine uses these; the Rust engine
    /// computes its own k-means and ignores them.
    nonisolated static func generateCentroids(
        from embeddings: [[[Float]]], nbits: Int, embeddingDim: Int
    ) throws -> [[Float]] {
        let numCentroids = 1 << nbits  // 4 centroids for nbits=2

        var allVectors: [[Float]] = []
        for docEmbedding in embeddings {
            allVectors.append(contentsOf: docEmbedding)
        }

        guard !allVectors.isEmpty else {
            throw SearchEngineError.noEmbeddings
        }

        var centroids: [[Float]] = []
        if allVectors.count <= numCentroids {
            centroids = allVectors
            // Pad with random vectors if needed
            while centroids.count < numCentroids {
                let randomVector = (0 ..< embeddingDim).map { _ in Float.random(in: -1 ... 1) }
                centroids.append(normalize(randomVector))
            }
        } else {
            // Sample uniformly
            let stride = allVectors.count / numCentroids
            for i in 0 ..< numCentroids {
                let index = min(i * stride, allVectors.count - 1)
                centroids.append(allVectors[index])
            }
        }

        return centroids
    }

    /// Normalize a vector to unit length
    nonisolated static func normalize(_ vector: [Float]) -> [Float] {
        let norm = sqrt(vector.reduce(0) { $0 + $1 * $1 })
        guard norm > 0 else { return vector }
        return vector.map { $0 / norm }
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

    /// Delete the entire index and reset state
    func deleteIndex() async throws {
        print("🗑️ Deleting index...")

        // Remove the index directory
        if FileManager.default.fileExists(atPath: indexURL.path) {
            try FileManager.default.removeItem(at: indexURL)
            print("  ✅ Index directory removed")
        }

        // Delete ObjectBox metadata for this index
        try await metadataProvider.deleteIndex(indexName: indexName)
        print("  ✅ ObjectBox metadata deleted")

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
}
