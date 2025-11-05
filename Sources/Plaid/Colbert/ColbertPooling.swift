import Foundation

public enum HierarchicalPoolingError: Error, LocalizedError {
    case invalidTensorRank
    case inconsistentDimensions

    public var errorDescription: String? {
        switch self {
        case .invalidTensorRank:
            return "Input must have shape [batch, tokens, embeddingDim]."
        case .inconsistentDimensions:
            return "All embeddings must share the same dimensionality."
        }
    }
}

/// Clusters document tokens to shrink the number of embeddings per passage.
public func hierarchicalPooling(
    documents embeddings: [[[Float]]],
    poolFactor: Int
) throws -> [[[Float]]] {
    guard poolFactor > 1 else { return embeddings }
    guard let embeddingDim = embeddings.first?.first?.count else {
        throw HierarchicalPoolingError.invalidTensorRank
    }

    var pooled: [[[Float]]] = []
    pooled.reserveCapacity(embeddings.count)

    for document in embeddings {
        guard let firstToken = document.first else {
            pooled.append(document)
            continue
        }

        let tokensToPool = Array(document.dropFirst())
        let numTokens = tokensToPool.count

        if numTokens <= 1 {
            pooled.append(document)
            continue
        }

        for token in tokensToPool {
            guard token.count == embeddingDim else {
                throw HierarchicalPoolingError.inconsistentDimensions
            }
        }

        let desiredClusters = max(1, numTokens / poolFactor)
        if desiredClusters >= numTokens {
            pooled.append(document)
            continue
        }

        let clusters = performClustering(
            tokens: tokensToPool,
            targetClusterCount: desiredClusters
        )

        var pooledVectors: [[Float]] = clusters.map { cluster in
            let divisor = Float(max(cluster.count, 1))
            var mean = Array(repeating: Float(0), count: embeddingDim)
            for vector in cluster {
                mean = zip(mean, vector).map(+)
            }
            return mean.map { $0 / divisor }
        }

        pooledVectors.append(firstToken)
        pooled.append(pooledVectors)
    }

    return pooled
}

private func performClustering(
    tokens: [[Float]],
    targetClusterCount: Int
) -> [[[Float]]] {
    struct Cluster {
        var vectors: [[Float]]

        var centroid: [Float] {
            let count = Float(max(vectors.count, 1))
            return vectors.reduce(Array(repeating: Float(0), count: vectors.first?.count ?? 0)) {
                zip($0, $1).map(+)
            }.map { $0 / count }
        }
    }

    var clusters: [Cluster] = tokens.map { Cluster(vectors: [$0]) }
    guard let dimension = tokens.first?.count else { return [] }

    while clusters.count > targetClusterCount {
        var bestPair: (Int, Int)?
        var bestDistance: Float = .infinity

        for i in 0 ..< clusters.count {
            let centroidI = clusters[i].centroid
            for j in i + 1 ..< clusters.count {
                let centroidJ = clusters[j].centroid
                let distance = 1 - cosineSimilarity(centroidI, centroidJ, dimension: dimension)
                if distance < bestDistance {
                    bestDistance = distance
                    bestPair = (i, j)
                }
            }
        }

        guard let pair = bestPair else { break }
        var merged = clusters[pair.0]
        merged.vectors.append(contentsOf: clusters[pair.1].vectors)

        clusters.remove(at: pair.1)
        clusters.remove(at: pair.0)
        clusters.append(merged)
    }

    return clusters.map(\.vectors)
}

private func cosineSimilarity(_ lhs: [Float], _ rhs: [Float], dimension: Int) -> Float {
    precondition(lhs.count == dimension && rhs.count == dimension, "Vector dimension mismatch")
    var dot: Float = 0
    var lhsNorm: Float = 0
    var rhsNorm: Float = 0

    for i in 0 ..< dimension {
        let l = lhs[i]
        let r = rhs[i]
        dot += l * r
        lhsNorm += l * l
        rhsNorm += r * r
    }

    let denom = sqrt(lhsNorm) * sqrt(rhsNorm)
    guard denom > 0 else { return 0 }
    return dot / denom
}
