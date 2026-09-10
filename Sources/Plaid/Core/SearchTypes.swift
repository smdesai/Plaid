import Foundation

/// Errors surfaced by the vector-engine seam and its callers. Shared by every
/// `SearchBackend` implementation; not tied to any one engine.
public enum PlaidError: Error, LocalizedError {
    case invalidEmbeddingDimensions(expected: Int, actual: Int)
    case emptyEmbeddingSet
    case mismatchedQueryDimension(expected: Int, actual: Int)
    case indexNotFound(URL)
    case invalidSubset(String)
    case invalidDocumentId(Int, totalDocuments: Int)

    public var errorDescription: String? {
        switch self {
        case .invalidEmbeddingDimensions(let expected, let actual):
            return "Embedding dimension mismatch. Expected \(expected), got \(actual)."
        case .emptyEmbeddingSet:
            return "At least one embedding row is required."
        case .mismatchedQueryDimension(let expected, let actual):
            return "Query dimension mismatch. Expected \(expected), got \(actual)."
        case .indexNotFound(let url):
            return "No index materialized at \(url.path)."
        case .invalidSubset(let reason):
            return "Subset validation failed: \(reason)."
        case .invalidDocumentId(let docId, let totalDocuments):
            return "Document ID \(docId) is out of range. Valid range is 0..<\(totalDocuments)."
        }
    }
}

/// Search knobs passed across the `SearchBackend` seam. `Codable` so they can be
/// persisted or read from fixtures; `logTiming` is omitted from output when false
/// to keep fixtures stable.
public struct SearchParameters: Codable, Sendable {
    public var batchSize: Int
    public var nFullScores: Int
    public var topK: Int
    public var nIvfProbe: Int
    public var logTiming: Bool

    enum CodingKeys: String, CodingKey {
        case batchSize
        case nFullScores
        case topK
        case nIvfProbe
        case logTiming
    }

    public init(
        batchSize: Int, nFullScores: Int, topK: Int, nIvfProbe: Int, logTiming: Bool = false
    ) {
        self.batchSize = batchSize
        self.nFullScores = nFullScores
        self.topK = topK
        self.nIvfProbe = nIvfProbe
        self.logTiming = logTiming
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        batchSize = try container.decode(Int.self, forKey: .batchSize)
        nFullScores = try container.decode(Int.self, forKey: .nFullScores)
        topK = try container.decode(Int.self, forKey: .topK)
        nIvfProbe = try container.decode(Int.self, forKey: .nIvfProbe)
        logTiming = try container.decodeIfPresent(Bool.self, forKey: .logTiming) ?? false
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(batchSize, forKey: .batchSize)
        try container.encode(nFullScores, forKey: .nFullScores)
        try container.encode(topK, forKey: .topK)
        try container.encode(nIvfProbe, forKey: .nIvfProbe)
        if logTiming {
            try container.encode(logTiming, forKey: .logTiming)
        }
    }
}

/// One query's ranked results: `passageIds` best-first, aligned with `scores`.
public struct QueryResult: Codable, Sendable {
    public let queryId: Int
    public let passageIds: [Int]
    public let scores: [Float]

    public init(queryId: Int, passageIds: [Int], scores: [Float]) {
        self.queryId = queryId
        self.passageIds = passageIds
        self.scores = scores
    }
}
