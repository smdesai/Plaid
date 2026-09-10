import Foundation
import ObjectBox
import Plaid

// MARK: - ObjectBox Entity

/// ObjectBox entity for storing Plaid document metadata.
///
/// Maps `plaidDocId` (the engine's internal passage id, i.e. `QueryResult.passageId`)
/// to the document name and chunk text — the bridge between search results and
/// human-readable metadata.
///
/// The persistence bindings (entity model, `Property` accessors, and the
/// `Store(directoryPath:)` convenience initializer) are produced by the ObjectBox
/// code generator into `generated/EntityInfo-PlaidDemoApp.generated.swift`. After
/// changing this entity, re-run the **ObjectBoxGeneratorCommand** plugin
/// (right-click the project in Xcode, or
/// `swift package plugin objectbox-generator`).
// objectbox: entity
class PlaidDocumentEntity {
    // objectbox: id
    var id: Id = 0

    /// The document ID used by Plaid (maps to passageId in QueryResult).
    /// This is the critical bridge between Plaid search results and document metadata.
    // objectbox: index
    var plaidDocId: Int = 0

    /// Human-readable document name (e.g., filename)
    var documentName: String = ""

    /// The actual text content of this document/chunk
    var chunkText: String = ""

    /// If the document was chunked, which chunk is this (0-indexed)
    var chunkIndex: Int = 0

    /// Optional original file path
    var filePath: String = ""

    /// Name of the Plaid index this document belongs to
    // objectbox: index
    var indexName: String = ""

    /// Unix timestamp when this document was indexed
    var createdAt: Int64 = 0

    /// Optional JSON-encoded metadata for extensibility
    var metadataJson: String = ""

    required init() {}
}

// MARK: - ObjectBox Metadata Provider

/// ObjectBox-based implementation of `PlaidMetadataProvider`.
///
/// Stores document metadata in an embedded ObjectBox database. This lives in the
/// app (not the `Plaid` package) precisely because storage is a caller concern:
/// the package only defines the `PlaidMetadataProvider` protocol, so a different
/// consumer can back it with any store.
actor ObjectBoxMetadataProvider: PlaidMetadataProvider {
    static let shared = ObjectBoxMetadataProvider()

    private var store: Store?
    private var box: Box<PlaidDocumentEntity>?

    private init() {}

    deinit {
        store?.close()
    }

    // MARK: - PlaidMetadataProvider Implementation

    func registerDocument(
        plaidDocId: Int,
        documentName: String,
        chunkText: String,
        chunkIndex: Int,
        filePath: String?,
        indexName: String
    ) async throws {
        let box = try ensureBox()

        let entity = PlaidDocumentEntity()
        entity.plaidDocId = plaidDocId
        entity.documentName = documentName
        entity.chunkText = chunkText
        entity.chunkIndex = chunkIndex
        entity.filePath = filePath ?? ""
        entity.indexName = indexName
        entity.createdAt = Int64(Date().timeIntervalSince1970)

        try box.put(entity)
    }

    func registerDocuments(
        _ documents: [(
            plaidDocId: Int, documentName: String, chunkText: String, chunkIndex: Int,
            filePath: String?
        )],
        indexName: String
    ) async throws {
        let box = try ensureBox()

        var entities: [PlaidDocumentEntity] = []
        entities.reserveCapacity(documents.count)

        let timestamp = Int64(Date().timeIntervalSince1970)

        for doc in documents {
            let entity = PlaidDocumentEntity()
            entity.plaidDocId = doc.plaidDocId
            entity.documentName = doc.documentName
            entity.chunkText = doc.chunkText
            entity.chunkIndex = doc.chunkIndex
            entity.filePath = doc.filePath ?? ""
            entity.indexName = indexName
            entity.createdAt = timestamp
            entities.append(entity)
        }

        try box.put(entities)
        print("📦 ObjectBox: Registered \(documents.count) documents for index '\(indexName)'")
    }

    func getDocument(plaidDocId: Int, indexName: String) async throws -> PlaidDocumentMetadata? {
        let box = try ensureBox()

        let query = try box.query {
            PlaidDocumentEntity.plaidDocId == plaidDocId
                && PlaidDocumentEntity.indexName == indexName
        }.build()

        guard let entity = try query.findFirst() else {
            return nil
        }

        return entityToMetadata(entity)
    }

    func getDocuments(plaidDocIds: [Int], indexName: String) async throws -> [PlaidDocumentMetadata]
    {
        guard !plaidDocIds.isEmpty else { return [] }

        let box = try ensureBox()

        // Query all documents for this index that match the IDs
        let query = try box.query {
            PlaidDocumentEntity.indexName == indexName
        }.build()

        let allEntities = try query.find()

        // Filter to requested IDs and create lookup
        let entityLookup = Dictionary(
            uniqueKeysWithValues:
                allEntities
                .filter { plaidDocIds.contains($0.plaidDocId) }
                .map { ($0.plaidDocId, $0) }
        )

        // Return in the order of requested IDs
        return plaidDocIds.compactMap { docId in
            guard let entity = entityLookup[docId] else { return nil }
            return entityToMetadata(entity)
        }
    }

    func deleteIndex(indexName: String) async throws {
        let box = try ensureBox()

        let query = try box.query {
            PlaidDocumentEntity.indexName == indexName
        }.build()

        let count = try query.remove()
        print("🗑️ ObjectBox: Deleted \(count) documents from index '\(indexName)'")
    }

    func documentCount(indexName: String) async throws -> Int {
        let box = try ensureBox()

        let query = try box.query {
            PlaidDocumentEntity.indexName == indexName
        }.build()

        return try query.count()
    }

    // MARK: - Additional Utility Methods

    /// Get total count of all documents across all indexes
    func totalDocumentCount() throws -> Int {
        let box = try ensureBox()
        return try box.count()
    }

    /// Get database file size as human-readable string
    func databaseSize() throws -> String {
        let directory = try databaseDirectory(createIfNeeded: false)
        guard FileManager.default.fileExists(atPath: directory.path) else {
            return "0 bytes"
        }

        let size = try directorySize(at: directory)
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        return formatter.string(fromByteCount: size)
    }

    /// Delete the entire database
    func deleteDatabase() throws {
        if let store = store {
            try store.closeAndDeleteAllFiles()
            self.store = nil
            self.box = nil
        } else {
            let directory = try databaseDirectory(createIfNeeded: false)
            if FileManager.default.fileExists(atPath: directory.path) {
                try FileManager.default.removeItem(at: directory)
            }
        }
        print("🗑️ ObjectBox: Database deleted")
    }

    /// Close the database connection
    func close() {
        store?.close()
        store = nil
        box = nil
    }

    // MARK: - Private Helpers

    private func ensureBox() throws -> Box<PlaidDocumentEntity> {
        if let box = box {
            return box
        }

        let directory = try databaseDirectory()
        // `Store(directoryPath:)` is the generated convenience initializer that
        // bakes in the entity model; if it's missing ("Missing argument for
        // parameter 'model'"), run the ObjectBoxGeneratorCommand plugin.
        let store = try Store(directoryPath: directory.path)
        let box: Box<PlaidDocumentEntity> = store.box(for: PlaidDocumentEntity.self)

        self.store = store
        self.box = box

        return box
    }

    private func databaseDirectory(createIfNeeded: Bool = true) throws -> URL {
        let fileManager = FileManager.default

        #if os(macOS)
            let baseDirectory = try fileManager.url(
                for: .applicationSupportDirectory,
                in: .userDomainMask,
                appropriateFor: nil,
                create: true
            )
        #else
            let baseDirectory = try fileManager.url(
                for: .documentDirectory,
                in: .userDomainMask,
                appropriateFor: nil,
                create: true
            )
        #endif

        let appDirectory = baseDirectory.appendingPathComponent("PlaidDemoApp", isDirectory: true)
        let storeDirectory = appDirectory.appendingPathComponent(
            "ObjectBoxStore", isDirectory: true)

        if createIfNeeded {
            try fileManager.createDirectory(at: storeDirectory, withIntermediateDirectories: true)
        }

        return storeDirectory
    }

    private func directorySize(at url: URL) throws -> Int64 {
        let fileManager = FileManager.default
        guard
            let enumerator = fileManager.enumerator(
                at: url,
                includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey],
                options: [.skipsHiddenFiles]
            )
        else {
            return 0
        }

        var total: Int64 = 0
        for case let fileURL as URL in enumerator {
            let resourceValues = try fileURL.resourceValues(forKeys: [
                .isRegularFileKey, .fileSizeKey,
            ])
            if resourceValues.isRegularFile == true {
                total += Int64(resourceValues.fileSize ?? 0)
            }
        }

        return total
    }

    private func entityToMetadata(_ entity: PlaidDocumentEntity) -> PlaidDocumentMetadata {
        PlaidDocumentMetadata(
            plaidDocId: entity.plaidDocId,
            documentName: entity.documentName,
            chunkText: entity.chunkText,
            chunkIndex: entity.chunkIndex,
            filePath: entity.filePath.isEmpty ? nil : entity.filePath,
            indexName: entity.indexName,
            createdAt: Date(timeIntervalSince1970: TimeInterval(entity.createdAt)),
            metadataJson: entity.metadataJson.isEmpty ? nil : entity.metadataJson
        )
    }
}
