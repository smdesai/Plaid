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

    private static let currentModelKey = "currentModel"

    init() {
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
            print("📦 Initializing LFM2 embedding generator...")
            generator = try LFM2ColbertEmbeddingGenerator(tokenizer: tokenizer)
        case .mxbaiEdge:
            print("📦 Initializing MXBAI-Edge embedding generator...")
            generator = try MXBAIEdgeColbertEmbeddingGenerator(tokenizer: tokenizer)
        }

        // Use SentenceBoundarySplitter for better semantic coherence in chunks
        // Preserves complete sentences instead of splitting mid-sentence
        let chunker = SentenceBoundarySplitter(withTokenizer: tokenizer)

        // Set embedding dimension based on model
        self.embeddingDim = model.embeddingDimension

        self.colbert = ColbertModel(
            generator: generator,
            configuration: .init(
                batchSize: 32,
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
        // Load saved model preference or default to LFM2
        let model: ModelType
        if let savedRawValue = UserDefaults.standard.string(forKey: SearchEngine.currentModelKey),
            let savedModel = ModelType(rawValue: savedRawValue)
        {
            model = savedModel
        } else {
            model = .lfm2
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

    /// Create a new index from documents
    /// Each document is chunked and each chunk becomes a separate searchable unit
    func createIndex(documents: [Document]) async throws {
        guard let colbert = colbert else {
            throw SearchEngineError.modelNotInitialized
        }

        print("🏗️  Creating index from \(documents.count) documents...")

        await MainActor.run {
            isIndexing = true
            indexingProgress = 0.0
            errorMessage = nil
        }

        // Each chunk gets its own plaidDocId, stored as separate embedding array
        var allChunkEmbeddings: [[[Float]]] = []
        var chunksForObjectBox:
            [(
                plaidDocId: Int, documentName: String, chunkText: String, chunkIndex: Int,
                filePath: String?
            )] = []
        var documentMetadata: [Int: DocumentMetadata] = [:]
        var currentPlaidDocId = 0

        // Encode each document using chunk-aware encoding
        for (docIndex, doc) in documents.enumerated() {
            await MainActor.run {
                currentDocument = doc.filename
                indexingProgress = Double(docIndex) / Double(documents.count)
            }

            print("📄 Encoding document [\(docIndex + 1)/\(documents.count)]: \(doc.filename)")

            // Use encodeDocument to get individual chunks with their text
            let chunkedResult = try colbert.encodeDocument(doc.text)

            print(
                "  ✅ \(chunkedResult.chunks.count) chunks, \(chunkedResult.totalEmbeddingCount) total embeddings"
            )

            // Each chunk becomes a separate entry in Plaid index
            for chunk in chunkedResult.chunks {
                // Store chunk embeddings for Plaid
                allChunkEmbeddings.append(chunk.embeddings)

                // Store chunk metadata for ObjectBox
                chunksForObjectBox.append(
                    (
                        plaidDocId: currentPlaidDocId,
                        documentName: doc.filename,
                        chunkText: chunk.text,
                        chunkIndex: chunk.chunkIndex,
                        filePath: nil
                    ))

                // Track in document metadata
                documentMetadata[currentPlaidDocId] = DocumentMetadata(
                    id: currentPlaidDocId,
                    filename: doc.filename,
                    addedAt: Date(),
                    characterCount: chunk.text.count,
                    embeddingCount: chunk.embeddings.count,
                    text: chunk.text,
                    originalDocId: docIndex
                )

                currentPlaidDocId += 1
            }
        }

        // Generate centroids from all chunk embeddings
        print("🎯 Generating centroids...")
        let centroids = try generateCentroids(from: allChunkEmbeddings)
        print("  ✅ Generated \(centroids.count) centroids")

        // Create Plaid index - each chunk is a separate "document"
        print("💾 Creating Plaid index with \(allChunkEmbeddings.count) chunks...")
        try Plaid.create(
            indexURL: indexURL,
            embeddingDim: embeddingDim,
            nbits: nbits,
            embeddings: allChunkEmbeddings,
            centroids: centroids,
            batchSize: 64
        )

        // Store chunk metadata in ObjectBox
        print("📦 Storing \(chunksForObjectBox.count) chunk metadata entries in ObjectBox...")
        try await metadataProvider.registerDocuments(chunksForObjectBox, indexName: indexName)

        // Save index state
        let totalEmbeddings = allChunkEmbeddings.reduce(0) { $0 + $1.count }
        self.indexState = IndexState(
            documents: documentMetadata,
            createdAt: Date(),
            lastModified: Date(),
            totalDocuments: allChunkEmbeddings.count,
            totalEmbeddings: totalEmbeddings
        )
        try saveIndexState()

        await MainActor.run {
            isIndexing = false
            hasIndex = true
            indexingProgress = 1.0
        }

        print("✅ Index created successfully!")
        print(
            "   📊 \(documents.count) documents → \(allChunkEmbeddings.count) chunks → \(totalEmbeddings) embeddings"
        )
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

        let results = try Plaid.loadAndSearch(
            indexURL: indexURL,
            queries: [queryEmbedding],
            searchParameters: params,
            showProgress: false,
            preloadIndex: true
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
    /// Note: Decompression is the bottleneck (~40-50ms per chunk on iOS).
    /// The parameters below balance quality vs search time, given this constraint.
    ///
    /// Performance characteristics (measured on 1679-chunk index):
    /// - 80 chunks → ~3.5s (degraded quality for some queries)
    /// - 150 chunks → ~6.5s (good quality/speed balance)
    /// - 200 chunks → ~8.5s (excellent quality, slower)
    /// - 300 chunks → ~13s (overkill for topK=3)
    private func adaptiveSearchParams(totalChunks: Int, topK: Int) -> (
        nFullScores: Int, nIvfProbe: Int
    ) {
        // For small indices, score everything for perfect recall
        if totalChunks <= 100 {
            return (totalChunks, min(16, totalChunks))
        }

        // For medium indices, balance quality and speed
        if totalChunks <= 500 {
            let nFullScores = min(150, totalChunks)
            let nIvfProbe = min(24, totalChunks / 4)
            return (max(topK * 30, nFullScores), max(8, nIvfProbe))
        }

        // For large indices (500-2000 chunks)
        // Use 150 chunks as sweet spot: good quality, reasonable speed (~6-7s)
        if totalChunks <= 2000 {
            let nFullScores = max(topK * 40, 150)  // Score 150 candidates for quality
            let nIvfProbe = min(32, max(24, totalChunks / 80))  // Probe 24-32 clusters
            return (nFullScores, nIvfProbe)
        }

        // For very large indices, cap at 200 chunks
        let nFullScores = max(topK * 50, 200)
        let nIvfProbe = min(48, max(32, totalChunks / 100))
        return (nFullScores, nIvfProbe)
    }

    /// Generate centroids from embeddings using uniform sampling
    private func generateCentroids(from embeddings: [[[Float]]]) throws -> [[Float]] {
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
    private func normalize(_ vector: [Float]) -> [Float] {
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
