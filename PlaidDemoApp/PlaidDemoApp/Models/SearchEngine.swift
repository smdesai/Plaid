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

    private let indexURL: URL
    private var tokenizer: PreTrainedColbertTokenizer?
    private var colbert: ColbertModel?

    private let modelId = "LiquidAI/LFM2-ColBERT-350M"
    private let embeddingDim = 128
    private let nbits = 2

    init() {
        // Set up index directory in Application Support
        let appSupport = FileManager.default.urls(
            //            for: .applicationSupportDirectory,
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

    /// Initialize the ColBERT model and check for existing index
    func initialize() async throws {
        print("🔧 Initializing SearchEngine...")

        // Load tokenizer and model
        print("📥 Loading ColBERT model...")
        self.tokenizer = try await PreTrainedColbertTokenizer.from(pretrained: modelId)

        guard let tokenizer = tokenizer else {
            throw SearchEngineError.modelLoadFailed
        }

        let generator = try LFM2ColbertEmbeddingGenerator(tokenizer: tokenizer)
        let chunker = TokenSplitter(withTokenizer: tokenizer)

        self.colbert = ColbertModel(
            generator: generator,
            configuration: .init(
                batchSize: 32,
                embeddingDimension: embeddingDim,
                queryLength: tokenizer.maxSequenceLength,
                documentLength: tokenizer.maxSequenceLength
            ),
            chunker: chunker
        )

        print("✅ Model loaded successfully")

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

    /// Check if an index exists on disk
    private func checkForExistingIndex() -> Bool {
        let metadataPath = indexURL.appendingPathComponent("metadata.json")
        let statePath = indexURL.appendingPathComponent("index_state.json")
        return FileManager.default.fileExists(atPath: metadataPath.path)
            && FileManager.default.fileExists(atPath: statePath.path)
    }

    /// Create a new index from documents
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

        var allEmbeddings: [[[Float]]] = []
        var documentMetadata: [Int: DocumentMetadata] = [:]

        // Encode each document
        for (index, doc) in documents.enumerated() {
            await MainActor.run {
                currentDocument = doc.filename
                indexingProgress = Double(index) / Double(documents.count)
            }

            print("📄 Encoding document [\(index + 1)/\(documents.count)]: \(doc.filename)")
            let embeddings = try colbert.encode(sentence: doc.text, isQuery: false)
            allEmbeddings.append(embeddings)

            documentMetadata[index] = DocumentMetadata(
                id: index,
                filename: doc.filename,
                addedAt: Date(),
                characterCount: doc.text.count,
                embeddingCount: embeddings.count,
                text: doc.text
            )

            print("  ✅ \(embeddings.count) embeddings generated")
        }

        // Generate centroids
        print("🎯 Generating centroids...")
        let centroids = try generateCentroids(from: allEmbeddings)
        print("  ✅ Generated \(centroids.count) centroids")

        // Create index
        print("💾 Creating Plaid index...")
        try Plaid.create(
            indexURL: indexURL,
            embeddingDim: embeddingDim,
            nbits: nbits,
            embeddings: allEmbeddings,
            centroids: centroids,
            batchSize: 64
        )

        // Save metadata
        let totalEmbeddings = allEmbeddings.reduce(0) { $0 + $1.count }
        self.indexState = IndexState(
            documents: documentMetadata,
            createdAt: Date(),
            lastModified: Date(),
            totalDocuments: documents.count,
            totalEmbeddings: totalEmbeddings
        )
        try saveIndexState()

        await MainActor.run {
            isIndexing = false
            hasIndex = true
            indexingProgress = 1.0
        }

        print("✅ Index created successfully! \(totalEmbeddings) total embeddings")
    }

    /// Search the index with a query
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

        // Search index
        let params = SearchParameters(
            batchSize: 1,
            nFullScores: indexState.totalDocuments,
            topK: topK,
            nIvfProbe: 8,
            logTiming: true
        )

        let results = try Plaid.loadAndSearch(
            indexURL: indexURL,
            queries: [queryEmbedding],
            searchParameters: params,
            showProgress: false,
            preloadIndex: false
        )

        guard let firstResult = results.first else {
            return []
        }

        print("  ✅ Found \(firstResult.passageIds.count) results")

        // Map results to documents
        let searchResults = zip(firstResult.passageIds, firstResult.scores).compactMap {
            docId, score -> SearchResult? in
            guard let metadata = indexState.documents[docId] else {
                print("  ⚠️  Warning: Document ID \(docId) not found in metadata")
                return nil
            }
            return SearchResult(
                documentId: docId,
                filename: metadata.filename,
                score: score,
                text: metadata.text
            )
        }

        return searchResults
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
}
