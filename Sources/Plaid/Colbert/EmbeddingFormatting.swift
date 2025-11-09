import Foundation

/// Utilities for formatting embeddings for debugging and logging
public enum EmbeddingFormatting {
    /// Formats a preview of embeddings showing head and tail tokens
    /// - Parameters:
    ///   - embeddings: Array of token embeddings
    ///   - headCount: Number of tokens to show from the beginning
    ///   - tailCount: Number of tokens to show from the end
    /// - Returns: Formatted string preview
    public static func formatEmbeddingsPreview(
        _ embeddings: [[Float]],
        headCount: Int = 5,
        tailCount: Int = 5
    ) -> String {
        guard !embeddings.isEmpty else { return "(empty)" }
        return embeddings.enumerated().map { index, vector in
            let formatted = formatVector(vector, headCount: headCount, tailCount: tailCount)
            return "  [token #\(index)] \(formatted)"
        }.joined(separator: "\n")
    }

    /// Formats a single embedding vector showing head and tail dimensions
    /// - Parameters:
    ///   - values: Embedding vector values
    ///   - headCount: Number of dimensions to show from the beginning
    ///   - tailCount: Number of dimensions to show from the end
    /// - Returns: Formatted string representation
    public static func formatVector(
        _ values: [Float],
        headCount: Int,
        tailCount: Int
    ) -> String {
        let formatter: (Float) -> String = { String(format: "%.9f", $0) }

        if values.count <= headCount + tailCount {
            return "[" + values.map(formatter).joined(separator: ", ") + "]"
        }

        let head = values.prefix(headCount).map(formatter)
        let tail = values.suffix(tailCount).map(formatter)
        return "[" + head.joined(separator: ", ") + ", …, " + tail.joined(separator: ", ") + "]"
    }
}
