import Foundation
import NextPlaidBindings

/// `PlaidMetadataProvider` backed by the SQLite `metadata.db` that every
/// next-plaid index directory already carries (the bundled SQLite compiled into
/// `NextPlaidFFI`). Per-chunk text is stored keyed by the engine's own passage
/// id (`_subset_` == `plaidDocId`), so a search hit resolves straight back to
/// its text with no second database.
///
/// This replaces the app's former ObjectBox store. It lives inside the `Plaid`
/// package (not the app target) because the FFI free functions it calls
/// (`storeDocuments` / `getDocuments` / `documentCount`) are only visible
/// through the internal `NextPlaidBindings` target, which the app can't import.
///
/// The provider is bound to a single index directory (`indexURL`); the app is
/// 1:1 (one index, `indexName == "default"`), so `indexName` is accepted for
/// protocol conformance but every call targets `indexURL/metadata.db`. All FFI
/// calls take `indexURL.path`; the engine appends `metadata.db` itself.
///
/// Delete is handled entirely by the vector path: `NextPlaidBackend.delete` →
/// `PlaidIndex.remove` → `MmapIndex::delete` already re-sequences `metadata.db`
/// with the same `new_id = old_id − count(deleted < old_id)` rule it applies to
/// the vectors, so stored text stays tied to its embedding. This provider never
/// deletes individual rows.
public final class NextPlaidMetadataProvider: PlaidMetadataProvider {
    /// JSON keys for the per-chunk payload. `_subset_` is reserved by the engine
    /// (it carries the passage id) and is never written here — it comes back on
    /// read via `getDocuments`.
    private enum Key {
        static let documentName = "documentName"
        static let chunkText = "chunkText"
        static let chunkIndex = "chunkIndex"
        static let embeddingCount = "embeddingCount"
        static let filePath = "filePath"
        static let createdAt = "createdAt"
        static let subset = "_subset_"
    }

    private let indexURL: URL

    public init(indexURL: URL) {
        self.indexURL = indexURL
    }

    private var path: String { indexURL.standardizedFileURL.path }

    // MARK: - Registration

    public func registerDocument(
        plaidDocId: Int,
        documentName: String,
        chunkText: String,
        chunkIndex: Int,
        embeddingCount: Int,
        filePath: String?,
        indexName: String
    ) async throws {
        try await registerDocuments(
            [(plaidDocId, documentName, chunkText, chunkIndex, embeddingCount, filePath)],
            indexName: indexName
        )
    }

    public func registerDocuments(
        _ documents: [(
            plaidDocId: Int, documentName: String, chunkText: String, chunkIndex: Int,
            embeddingCount: Int, filePath: String?
        )],
        indexName: String
    ) async throws {
        guard !documents.isEmpty else { return }

        let createdAt = Int64(Date().timeIntervalSince1970)
        var docIds: [Int64] = []
        var payloads: [String] = []
        docIds.reserveCapacity(documents.count)
        payloads.reserveCapacity(documents.count)

        for doc in documents {
            docIds.append(Int64(doc.plaidDocId))
            // filePath is stored as "" when absent (a stable, non-null column)
            // and mapped back to nil on read, matching the prior ObjectBox shape.
            let object: [String: Any] = [
                Key.documentName: doc.documentName,
                Key.chunkText: doc.chunkText,
                Key.chunkIndex: doc.chunkIndex,
                Key.embeddingCount: doc.embeddingCount,
                Key.filePath: doc.filePath ?? "",
                Key.createdAt: createdAt,
            ]
            payloads.append(try Self.encode(object))
        }

        let written = try storeDocuments(path: path, docIds: docIds, metadataJson: payloads)
        print("🗃️ SQLite: stored \(written) documents for index '\(indexName)'")
    }

    // MARK: - Retrieval

    public func getDocument(plaidDocId: Int, indexName: String) async throws
        -> PlaidDocumentMetadata?
    {
        let docs = try await getDocuments(plaidDocIds: [plaidDocId], indexName: indexName)
        return docs.first
    }

    public func getDocuments(plaidDocIds: [Int], indexName: String) async throws
        -> [PlaidDocumentMetadata]
    {
        guard !plaidDocIds.isEmpty else { return [] }

        let ids = plaidDocIds.map { Int64($0) }
        let rows = try NextPlaidBindings.getDocuments(path: path, docIds: ids)

        // Rows come back in requested order, each carrying its `_subset_`
        // (== plaidDocId); missing ids are silently dropped by the engine, which
        // `enriched(...)` tolerates by dict-mapping on plaidDocId.
        return rows.compactMap { try? Self.decode($0, indexName: indexName) }
    }

    /// Push the `documentName` filter into SQLite instead of scanning the store.
    /// The engine validates the condition against the schema and binds `?` to a
    /// JSON-encoded parameter, so this is injection-safe.
    ///
    /// `documentName` is a *fat* column: the engine's v2 layout routes every
    /// column outside its fixed thin allowlist (`file`, `name`, `line`, …) into
    /// the `METADATA_CONTENT` table, and `documentName` isn't on that list. The
    /// engine's `get` JOINs the thin and fat tables and resolves the unqualified
    /// `documentName` against the join, so filtering a fat-routed column works
    /// exactly like a thin one. `NextPlaidMetadataProviderTests`'
    /// `testDeleteEntireDocumentByNameRemovesAllItsChunks` exercises this path
    /// end-to-end (store → `documentName = ?` filter → correct rows).
    public func documentChunks(named documentName: String, indexName: String) async throws
        -> [PlaidDocumentMetadata]
    {
        let param = try Self.encodeScalar(documentName)
        let rows = try NextPlaidBindings.queryDocuments(
            path: path,
            condition: "documentName = ?",
            params: [param]
        )
        return rows.compactMap { try? Self.decode($0, indexName: indexName) }
    }

    /// List every distinct document by grouping the store's chunks on
    /// `documentName`. The engine keeps `_subset_` ids dense (`0..<count`) even
    /// after deletes, so we read the whole id space in one batch and fold it in
    /// Swift — enumerating every document is inherently a full pass, and this
    /// keeps it to a single FFI round-trip. Results are ordered by name.
    public func indexedDocuments(indexName: String) async throws -> [IndexedDocument] {
        let count = try await documentCount(indexName: indexName)
        guard count > 0 else { return [] }

        let chunks = try await getDocuments(plaidDocIds: Array(0 ..< count), indexName: indexName)

        // Fold chunks into per-document aggregates, preserving first-seen order.
        var order: [String] = []
        var byName: [String: IndexedDocument] = [:]
        for chunk in chunks {
            if let existing = byName[chunk.documentName] {
                byName[chunk.documentName] = IndexedDocument(
                    documentName: existing.documentName,
                    chunkCount: existing.chunkCount + 1,
                    embeddingCount: existing.embeddingCount + chunk.embeddingCount,
                    filePath: existing.filePath ?? chunk.filePath,
                    createdAt: min(existing.createdAt, chunk.createdAt)
                )
            } else {
                order.append(chunk.documentName)
                byName[chunk.documentName] = IndexedDocument(
                    documentName: chunk.documentName,
                    chunkCount: 1,
                    embeddingCount: chunk.embeddingCount,
                    filePath: chunk.filePath,
                    createdAt: chunk.createdAt
                )
            }
        }

        return order.compactMap { byName[$0] }.sorted { $0.documentName < $1.documentName }
    }

    // MARK: - Management

    public func deleteIndex(indexName: String) async throws {
        // `indexName` is intentionally ignored: this provider is bound to a
        // single `indexURL` (the app is 1:1, `indexName == "default"`), so every
        // call already targets `indexURL/metadata.db`. The parameter exists only
        // for protocol conformance.
        //
        // The caller (`SearchEngine.deleteIndex`) removes the whole index
        // directory, which takes `metadata.db` with it. Remove the db defensively
        // in case this is called on its own; ignore "already gone".
        let db = indexURL.appendingPathComponent("metadata.db")
        try? FileManager.default.removeItem(at: db)
    }

    public func documentCount(indexName: String) async throws -> Int {
        Int(try NextPlaidBindings.documentCount(path: path))
    }

    // MARK: - JSON

    /// JSON-encode a scalar for use as a `queryDocuments` `?` parameter (a
    /// string becomes `"foo"`, quotes included), which the engine binds safely.
    private static func encodeScalar(_ value: String) throws -> String {
        let data = try JSONEncoder().encode(value)
        guard let string = String(data: data, encoding: .utf8) else {
            throw PlaidError.metadataEncodingFailed
        }
        return string
    }

    private static func encode(_ object: [String: Any]) throws -> String {
        let data = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        guard let string = String(data: data, encoding: .utf8) else {
            throw PlaidError.metadataEncodingFailed
        }
        return string
    }

    private static func decode(_ json: String, indexName: String) throws -> PlaidDocumentMetadata {
        guard
            let data = json.data(using: .utf8),
            let object = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            throw PlaidError.metadataDecodingFailed
        }

        // `_subset_` is the engine's passage id; JSON numbers may decode as any
        // NSNumber, so normalize through Int/Double.
        guard let plaidDocId = intValue(object[Key.subset]) else {
            throw PlaidError.metadataDecodingFailed
        }

        let filePath = object[Key.filePath] as? String
        let createdAtSeconds = intValue(object[Key.createdAt]) ?? 0

        return PlaidDocumentMetadata(
            plaidDocId: plaidDocId,
            documentName: object[Key.documentName] as? String ?? "",
            chunkText: object[Key.chunkText] as? String ?? "",
            chunkIndex: intValue(object[Key.chunkIndex]) ?? 0,
            embeddingCount: intValue(object[Key.embeddingCount]) ?? 0,
            filePath: (filePath?.isEmpty ?? true) ? nil : filePath,
            indexName: indexName,
            createdAt: Date(timeIntervalSince1970: TimeInterval(createdAtSeconds)),
            metadataJson: nil
        )
    }

    private static func intValue(_ any: Any?) -> Int? {
        switch any {
        case let n as NSNumber: return n.intValue
        case let i as Int: return i
        case let d as Double: return Int(d)
        case let s as String: return Int(s)
        default: return nil
        }
    }
}
