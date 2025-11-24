import Foundation
import ObjectBox
import Plaid

// MARK: - ObjectBox Entity

/// ObjectBox entity for storing Plaid document metadata
/// Maps plaidDocId to document name and text content
// objectbox: entity
final class PlaidDocumentEntity: Entity, @unchecked Sendable {
    var id: Id = 0

    /// The document ID used by Plaid (maps to passageId in QueryResult)
    /// This is the critical bridge between Plaid search results and document metadata
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
    var indexName: String = ""

    /// Unix timestamp when this document was indexed
    var createdAt: Int64 = 0

    /// Optional JSON-encoded metadata for extensibility
    var metadataJson: String = ""

    required init() {}
}

// MARK: - ObjectBox Entity Conformances

extension PlaidDocumentEntity: ObjectBox.__EntityRelatable {
    typealias EntityType = PlaidDocumentEntity

    var _id: EntityId<PlaidDocumentEntity> {
        EntityId<PlaidDocumentEntity>(self.id)
    }
}

extension PlaidDocumentEntity: ObjectBox.EntityInspectable {
    typealias EntityBindingType = PlaidDocumentEntityBinding

    static var entityInfo = ObjectBox.EntityInfo(name: "PlaidDocumentEntity", id: 1)
    static var entityBinding = EntityBindingType()

    fileprivate static func buildEntity(modelBuilder: ObjectBox.ModelBuilder) throws {
        let entityBuilder = try modelBuilder.entityBuilder(
            for: PlaidDocumentEntity.self, id: 1, uid: 5001
        )
        try entityBuilder.addProperty(
            name: "id", type: PropertyType.long, flags: [.id], id: 1, uid: 5101
        )
        try entityBuilder.addProperty(
            name: "plaidDocId", type: PropertyType.int, flags: [.indexed], id: 2, uid: 5102,
            indexId: 1, indexUid: 6001
        )
        try entityBuilder.addProperty(
            name: "documentName", type: PropertyType.string, id: 3, uid: 5103
        )
        try entityBuilder.addProperty(
            name: "chunkText", type: PropertyType.string, id: 4, uid: 5104
        )
        try entityBuilder.addProperty(
            name: "chunkIndex", type: PropertyType.int, id: 5, uid: 5105
        )
        try entityBuilder.addProperty(
            name: "filePath", type: PropertyType.string, id: 6, uid: 5106
        )
        try entityBuilder.addProperty(
            name: "indexName", type: PropertyType.string, flags: [.indexed], id: 7, uid: 5107,
            indexId: 2, indexUid: 6002
        )
        try entityBuilder.addProperty(
            name: "createdAt", type: PropertyType.long, id: 8, uid: 5108
        )
        try entityBuilder.addProperty(
            name: "metadataJson", type: PropertyType.string, id: 9, uid: 5109
        )
        try entityBuilder.lastProperty(id: 9, uid: 5109)
    }
}

extension PlaidDocumentEntity {
    fileprivate func __setId(identifier: ObjectBox.Id) {
        self.id = identifier
    }
}

// MARK: - Property Definitions for Queries

extension PlaidDocumentEntity {
    static var id: Property<PlaidDocumentEntity, Id, Id> {
        Property<PlaidDocumentEntity, Id, Id>(propertyId: 1, isPrimaryKey: true)
    }
    static var plaidDocId: Property<PlaidDocumentEntity, Int, Void> {
        Property<PlaidDocumentEntity, Int, Void>(propertyId: 2, isPrimaryKey: false)
    }
    static var documentName: Property<PlaidDocumentEntity, String, Void> {
        Property<PlaidDocumentEntity, String, Void>(propertyId: 3, isPrimaryKey: false)
    }
    static var chunkText: Property<PlaidDocumentEntity, String, Void> {
        Property<PlaidDocumentEntity, String, Void>(propertyId: 4, isPrimaryKey: false)
    }
    static var chunkIndex: Property<PlaidDocumentEntity, Int, Void> {
        Property<PlaidDocumentEntity, Int, Void>(propertyId: 5, isPrimaryKey: false)
    }
    static var filePath: Property<PlaidDocumentEntity, String, Void> {
        Property<PlaidDocumentEntity, String, Void>(propertyId: 6, isPrimaryKey: false)
    }
    static var indexName: Property<PlaidDocumentEntity, String, Void> {
        Property<PlaidDocumentEntity, String, Void>(propertyId: 7, isPrimaryKey: false)
    }
    static var createdAt: Property<PlaidDocumentEntity, Int64, Void> {
        Property<PlaidDocumentEntity, Int64, Void>(propertyId: 8, isPrimaryKey: false)
    }
    static var metadataJson: Property<PlaidDocumentEntity, String, Void> {
        Property<PlaidDocumentEntity, String, Void>(propertyId: 9, isPrimaryKey: false)
    }
}

extension ObjectBox.Property where E == PlaidDocumentEntity {
    static var id: Property<PlaidDocumentEntity, Id, Id> {
        Property<PlaidDocumentEntity, Id, Id>(propertyId: 1, isPrimaryKey: true)
    }
    static var plaidDocId: Property<PlaidDocumentEntity, Int, Void> {
        Property<PlaidDocumentEntity, Int, Void>(propertyId: 2, isPrimaryKey: false)
    }
    static var documentName: Property<PlaidDocumentEntity, String, Void> {
        Property<PlaidDocumentEntity, String, Void>(propertyId: 3, isPrimaryKey: false)
    }
    static var chunkText: Property<PlaidDocumentEntity, String, Void> {
        Property<PlaidDocumentEntity, String, Void>(propertyId: 4, isPrimaryKey: false)
    }
    static var chunkIndex: Property<PlaidDocumentEntity, Int, Void> {
        Property<PlaidDocumentEntity, Int, Void>(propertyId: 5, isPrimaryKey: false)
    }
    static var filePath: Property<PlaidDocumentEntity, String, Void> {
        Property<PlaidDocumentEntity, String, Void>(propertyId: 6, isPrimaryKey: false)
    }
    static var indexName: Property<PlaidDocumentEntity, String, Void> {
        Property<PlaidDocumentEntity, String, Void>(propertyId: 7, isPrimaryKey: false)
    }
    static var createdAt: Property<PlaidDocumentEntity, Int64, Void> {
        Property<PlaidDocumentEntity, Int64, Void>(propertyId: 8, isPrimaryKey: false)
    }
    static var metadataJson: Property<PlaidDocumentEntity, String, Void> {
        Property<PlaidDocumentEntity, String, Void>(propertyId: 9, isPrimaryKey: false)
    }
}

// MARK: - Entity Binding

final class PlaidDocumentEntityBinding: ObjectBox.EntityBinding, Sendable {
    typealias EntityType = PlaidDocumentEntity
    typealias IdType = Id

    required init() {}

    func generatorBindingVersion() -> Int { 1 }

    func setEntityIdUnlessStruct(of entity: EntityType, to entityId: ObjectBox.Id) {
        entity.__setId(identifier: entityId)
    }

    func entityId(of entity: EntityType) -> ObjectBox.Id {
        entity.id
    }

    func collect(
        fromEntity entity: EntityType, id: ObjectBox.Id,
        propertyCollector: ObjectBox.FlatBufferBuilder,
        store: ObjectBox.Store
    ) throws {
        let offsetDocumentName = propertyCollector.prepare(string: entity.documentName)
        let offsetChunkText = propertyCollector.prepare(string: entity.chunkText)
        let offsetFilePath = propertyCollector.prepare(string: entity.filePath)
        let offsetIndexName = propertyCollector.prepare(string: entity.indexName)
        let offsetMetadataJson = propertyCollector.prepare(string: entity.metadataJson)

        propertyCollector.collect(id, at: 2 + 2 * 1)
        propertyCollector.collect(entity.plaidDocId, at: 2 + 2 * 2)
        propertyCollector.collect(dataOffset: offsetDocumentName, at: 2 + 2 * 3)
        propertyCollector.collect(dataOffset: offsetChunkText, at: 2 + 2 * 4)
        propertyCollector.collect(entity.chunkIndex, at: 2 + 2 * 5)
        propertyCollector.collect(dataOffset: offsetFilePath, at: 2 + 2 * 6)
        propertyCollector.collect(dataOffset: offsetIndexName, at: 2 + 2 * 7)
        propertyCollector.collect(entity.createdAt, at: 2 + 2 * 8)
        propertyCollector.collect(dataOffset: offsetMetadataJson, at: 2 + 2 * 9)
    }

    func createEntity(
        entityReader: ObjectBox.FlatBufferReader,
        store: ObjectBox.Store
    ) -> EntityType {
        let entity = PlaidDocumentEntity()
        entity.id = entityReader.read(at: 2 + 2 * 1)
        entity.plaidDocId = entityReader.read(at: 2 + 2 * 2)
        entity.documentName = entityReader.read(at: 2 + 2 * 3)
        entity.chunkText = entityReader.read(at: 2 + 2 * 4)
        entity.chunkIndex = entityReader.read(at: 2 + 2 * 5)
        entity.filePath = entityReader.read(at: 2 + 2 * 6)
        entity.indexName = entityReader.read(at: 2 + 2 * 7)
        entity.createdAt = entityReader.read(at: 2 + 2 * 8)
        entity.metadataJson = entityReader.read(at: 2 + 2 * 9)
        return entity
    }
}

// MARK: - Model Builder

private func plaidDocumentEntityModel() throws -> OpaquePointer {
    let modelBuilder = try ObjectBox.ModelBuilder()
    try PlaidDocumentEntity.buildEntity(modelBuilder: modelBuilder)
    modelBuilder.lastEntity(id: 1, uid: 5001)
    modelBuilder.lastIndex(id: 2, uid: 6002)
    return modelBuilder.finish()
}

// MARK: - ObjectBox Metadata Provider

/// ObjectBox-based implementation of PlaidMetadataProvider
/// Stores document metadata in an embedded ObjectBox database
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
        let model = try plaidDocumentEntityModel()
        let store = try Store(model: model, directory: directory.path)
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
