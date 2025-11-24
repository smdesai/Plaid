import Foundation

/// Represents the state and metadata of the search index
struct IndexState: Codable {
    /// Chunk metadata keyed by plaidDocId (chunk-level)
    var documents: [Int: DocumentMetadata]

    var createdAt: Date
    var lastModified: Date

    /// Total number of chunks in the index
    var totalDocuments: Int

    /// Total number of embeddings
    var totalEmbeddings: Int

    init(
        documents: [Int: DocumentMetadata],
        createdAt: Date,
        lastModified: Date,
        totalDocuments: Int,
        totalEmbeddings: Int
    ) {
        self.documents = documents
        self.createdAt = createdAt
        self.lastModified = lastModified
        self.totalDocuments = totalDocuments
        self.totalEmbeddings = totalEmbeddings
    }
}

/// Metadata for a single chunk in the index
struct DocumentMetadata: Codable {
    let id: Int  // plaidDocId (unique per chunk)
    let filename: String  // Original document filename
    let addedAt: Date
    let characterCount: Int  // Character count of this chunk
    let embeddingCount: Int  // Number of token embeddings in this chunk
    let text: String  // The chunk text
    let originalDocId: Int  // Reference to original document

    init(
        id: Int, filename: String, addedAt: Date, characterCount: Int, embeddingCount: Int,
        text: String, originalDocId: Int = 0
    ) {
        self.id = id
        self.filename = filename
        self.addedAt = addedAt
        self.characterCount = characterCount
        self.embeddingCount = embeddingCount
        self.text = text
        self.originalDocId = originalDocId
    }
}

/// Represents a document before indexing
struct Document {
    let filename: String
    let text: String
}

/// Represents a search result (at chunk level)
struct SearchResult: Identifiable {
    let id = UUID()
    let documentId: Int  // plaidDocId (unique per chunk)
    let filename: String  // Original document name
    let chunkIndex: Int  // Which chunk within the document (0-indexed)
    let score: Float
    let text: String  // The chunk text (not full document)

    init(documentId: Int, filename: String, chunkIndex: Int = 0, score: Float, text: String) {
        self.documentId = documentId
        self.filename = filename
        self.chunkIndex = chunkIndex
        self.score = score
        self.text = text
    }

    var scorePercentage: Int {
        Int(score * 100)
    }

    /// The chunk text is already a snippet - return it directly
    var snippet: String {
        text
    }

    /// Display name including chunk info if multiple chunks
    var displayName: String {
        if chunkIndex > 0 {
            return "\(filename) (chunk \(chunkIndex + 1))"
        }
        return filename
    }
}

/// Errors that can occur in the search engine
enum SearchEngineError: LocalizedError {
    case modelNotInitialized
    case modelLoadFailed
    case noIndex
    case noEmbeddings
    case invalidIndexState

    var errorDescription: String? {
        switch self {
        case .modelNotInitialized:
            return "Model not initialized"
        case .modelLoadFailed:
            return "Failed to load ColBERT model"
        case .noIndex:
            return "No index found"
        case .noEmbeddings:
            return "No embeddings generated"
        case .invalidIndexState:
            return "Index state is corrupted or invalid"
        }
    }
}
