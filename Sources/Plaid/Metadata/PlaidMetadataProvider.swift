import Foundation

// MARK: - Document Metadata Types

/// Metadata for a document stored in Plaid's index
public struct PlaidDocumentMetadata: Codable, Sendable {
    /// The document ID used by Plaid (maps to passageId in QueryResult)
    public let plaidDocId: Int

    /// Human-readable document name (e.g., filename)
    public let documentName: String

    /// The actual text content of this document/chunk
    public let chunkText: String

    /// If the document was chunked, which chunk is this (0-indexed)
    public let chunkIndex: Int

    /// Optional original file path
    public let filePath: String?

    /// Name of the Plaid index this document belongs to
    public let indexName: String

    /// When this document was indexed
    public let createdAt: Date

    /// Optional JSON-encoded metadata for extensibility
    public let metadataJson: String?

    public init(
        plaidDocId: Int,
        documentName: String,
        chunkText: String,
        chunkIndex: Int = 0,
        filePath: String? = nil,
        indexName: String,
        createdAt: Date = Date(),
        metadataJson: String? = nil
    ) {
        self.plaidDocId = plaidDocId
        self.documentName = documentName
        self.chunkText = chunkText
        self.chunkIndex = chunkIndex
        self.filePath = filePath
        self.indexName = indexName
        self.createdAt = createdAt
        self.metadataJson = metadataJson
    }
}

/// A search result enriched with document metadata
public struct EnrichedSearchResult: Codable, Sendable, Identifiable {
    public var id: Int { plaidDocId }

    /// The document ID from Plaid's QueryResult
    public let plaidDocId: Int

    /// ColBERT similarity score
    public let score: Float

    /// Human-readable document name
    public let documentName: String

    /// The actual text content
    public let chunkText: String

    /// Chunk index within the original document
    public let chunkIndex: Int

    /// Optional file path
    public let filePath: String?

    public init(
        plaidDocId: Int,
        score: Float,
        documentName: String,
        chunkText: String,
        chunkIndex: Int = 0,
        filePath: String? = nil
    ) {
        self.plaidDocId = plaidDocId
        self.score = score
        self.documentName = documentName
        self.chunkText = chunkText
        self.chunkIndex = chunkIndex
        self.filePath = filePath
    }

    /// Preview of the chunk text (first N characters)
    public var preview: String {
        let limit = 200
        if chunkText.count <= limit {
            return chunkText
        }
        return String(chunkText.prefix(limit)) + "..."
    }

    /// Score as a percentage (0-100)
    public var scorePercentage: Int {
        Int(score * 100)
    }
}

// MARK: - Metadata Provider Protocol

/// Protocol for providing document metadata storage
/// Implement this to enable rich search results with document names and text
public protocol PlaidMetadataProvider: Sendable {

    // MARK: - Registration (called during index creation)

    /// Register a single document with metadata
    /// - Parameters:
    ///   - plaidDocId: The document ID that will be used by Plaid (0, 1, 2, ...)
    ///   - documentName: Human-readable name for the document
    ///   - chunkText: The actual text content
    ///   - chunkIndex: Index of chunk within original document (default 0)
    ///   - filePath: Optional original file path
    ///   - indexName: Name of the Plaid index
    func registerDocument(
        plaidDocId: Int,
        documentName: String,
        chunkText: String,
        chunkIndex: Int,
        filePath: String?,
        indexName: String
    ) async throws

    /// Register multiple documents at once (more efficient for batch operations)
    /// - Parameters:
    ///   - documents: Array of document data tuples
    ///   - indexName: Name of the Plaid index
    func registerDocuments(
        _ documents: [(
            plaidDocId: Int, documentName: String, chunkText: String, chunkIndex: Int,
            filePath: String?
        )],
        indexName: String
    ) async throws

    // MARK: - Retrieval (called after search)

    /// Get metadata for a single document by its Plaid ID
    func getDocument(plaidDocId: Int, indexName: String) async throws -> PlaidDocumentMetadata?

    /// Get metadata for multiple documents (preserves order of input IDs)
    func getDocuments(plaidDocIds: [Int], indexName: String) async throws -> [PlaidDocumentMetadata]

    // MARK: - Management

    /// Delete all metadata for a specific index
    func deleteIndex(indexName: String) async throws

    /// Get the count of documents for an index
    func documentCount(indexName: String) async throws -> Int
}

// MARK: - QueryResult Extension

extension QueryResult {
    /// Enrich search results with document metadata from a provider
    /// - Parameters:
    ///   - provider: The metadata provider to use
    ///   - indexName: Name of the Plaid index
    /// - Returns: Array of enriched results with document names and text
    public func enriched(
        from provider: PlaidMetadataProvider,
        indexName: String
    ) async throws -> [EnrichedSearchResult] {
        let documents = try await provider.getDocuments(
            plaidDocIds: passageIds,
            indexName: indexName
        )

        // Create a lookup dictionary for O(1) access
        let docLookup = Dictionary(uniqueKeysWithValues: documents.map { ($0.plaidDocId, $0) })

        // Map results preserving order of passageIds/scores
        return zip(passageIds, scores).compactMap { docId, score -> EnrichedSearchResult? in
            guard let doc = docLookup[docId] else { return nil }
            return EnrichedSearchResult(
                plaidDocId: docId,
                score: score,
                documentName: doc.documentName,
                chunkText: doc.chunkText,
                chunkIndex: doc.chunkIndex,
                filePath: doc.filePath
            )
        }
    }
}
