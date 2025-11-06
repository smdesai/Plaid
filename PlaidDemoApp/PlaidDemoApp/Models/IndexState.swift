import Foundation

/// Represents the state and metadata of the search index
struct IndexState: Codable {
    var documents: [Int: DocumentMetadata]
    var createdAt: Date
    var lastModified: Date
    var totalDocuments: Int
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

/// Metadata for a single document in the index
struct DocumentMetadata: Codable {
    let id: Int
    let filename: String
    let addedAt: Date
    let characterCount: Int
    let embeddingCount: Int
    let text: String  // Store full text for display
}

/// Represents a document before indexing
struct Document {
    let filename: String
    let text: String
}

/// Represents a search result
struct SearchResult: Identifiable {
    let id = UUID()
    let documentId: Int
    let filename: String
    let score: Float
    let text: String

    var scorePercentage: Int {
        Int(score * 100)
    }

    var snippet: String {
        String(text.prefix(200))
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
