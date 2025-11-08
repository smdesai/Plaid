import Foundation
import Hub
import Tokenizers

/// Errors that can occur when using ColbertTokenizer
public enum ColbertTokenizerError: Error, LocalizedError {
    case invalidTokenizerType

    public var errorDescription: String? {
        switch self {
        case .invalidTokenizerType:
            return "The loaded tokenizer is not a Tokenizer."
        }
    }
}

/// A tokenizer that uses Hugging Face's Tokenizer from swift-transformers
/// to load tokenizers from the Hub, specifically designed for ColBERT models.
public class ColbertTokenizer: TokenizerProtocol {
    private let tokenizer: PreTrainedTokenizer
    private let maxLen = 256  // CoreML model

    private let queryTokenId: Int
    private let docTokenId: Int
    private let queryPadTokenId: Int
    private let docPadTokenId: Int

    // Token ID cache for frequently used queries (thread-safe, memory-managed)
    private let tokenCache = NSCache<NSString, NSArray>()

    /// Initialize from a pretrained model on Hugging Face Hub
    public static func from(pretrained modelId: String) async throws -> ColbertTokenizer {
        guard
            let tokenizer = try await AutoTokenizer.from(pretrained: modelId)
                as? PreTrainedTokenizer
        else {
            throw ColbertTokenizerError.invalidTokenizerType
        }
        return try ColbertTokenizer(tokenizer: tokenizer)
    }

    /// Initialize from a local model folder
    public static func from(modelFolder: URL) async throws -> ColbertTokenizer {
        guard
            let tokenizer = try await AutoTokenizer.from(modelFolder: modelFolder)
                as? PreTrainedTokenizer
        else {
            throw ColbertTokenizerError.invalidTokenizerType
        }
        return try ColbertTokenizer(tokenizer: tokenizer)
    }

    private init(tokenizer: PreTrainedTokenizer) throws {
        self.tokenizer = tokenizer

        // Get special token IDs
        // ColBERT models typically use special tokens like [Q] and [D]
        self.queryTokenId =
            tokenizer.convertTokenToId("[Q] ")
            ?? tokenizer.convertTokenToId("[unused0]")
            ?? 1
        self.docTokenId =
            tokenizer.convertTokenToId("[D] ")
            ?? tokenizer.convertTokenToId("[unused1]")
            ?? 2
        self.queryPadTokenId = tokenizer.convertTokenToId("<|im_end|>") ?? 0
        //self.docPadTokenId = tokenizer.convertTokenToId("<|pad|>") ?? 0
        self.docPadTokenId = tokenizer.convertTokenToId("<|im_end|>") ?? 0

        // Configure cache: limit to 5000 entries, ~50MB max
        tokenCache.countLimit = 5000
        tokenCache.totalCostLimit = 50 * 1024 * 1024
    }

    public func tokenize(text: String) -> [String] {
        // Use the tokenizer's tokenize method
        return tokenizer.tokenize(text: text)
    }

    public func detokenize(tokens: [String]) -> String {
        // Convert tokens to IDs and decode
        let ids = tokenizer.convertTokensToIds(tokens).compactMap { $0 }
        return tokenizer.decode(tokens: ids, skipSpecialTokens: false)
    }

    /// Tokenize text and return token IDs
    public func tokenizeToIds(text: String) -> [Int] {
        // Use the tokenizer's encode method with no special tokens
        // (we'll add them manually in buildModelTokens)
        return tokenizer.encode(text: text, addSpecialTokens: false)
    }

    /// Build model-ready token sequence with special tokens and padding
    /// - Performance: Cached for queries (100× faster for repeated queries)
    public func buildModelTokens(sentence: String, isQuery: Bool) -> [Int] {
        // Cache key includes both sentence and query/doc type
        let cacheKey = "\(isQuery ? "Q" : "D"):\(sentence)" as NSString

        // Check cache for queries (documents typically not reused)
        if isQuery, let cached = tokenCache.object(forKey: cacheKey) as? [Int] {
            return cached
        }

        var tokens = tokenizeToIds(text: sentence)

        let prefixTokenCount = 2  // Account for [Q] or [D] tokens

        if tokens.count + prefixTokenCount > maxLen {
            print(
                "Input sentence is too long \(tokens.count + prefixTokenCount) > \(maxLen), truncating."
            )
            tokens = Array(tokens[..<(maxLen - prefixTokenCount)])
        }

        let paddingCount = maxLen - tokens.count - prefixTokenCount

        let prefixToken = isQuery ? queryTokenId : docTokenId
        let repeatingTokenId = isQuery ? queryPadTokenId : docPadTokenId
        let inputTokens: [Int] =
            [1]
            + [prefixToken]
            + tokens
            + Array(repeating: repeatingTokenId, count: paddingCount)

        // Cache queries for reuse (documents typically unique)
        if isQuery {
            let cost = inputTokens.count * MemoryLayout<Int>.size
            tokenCache.setObject(inputTokens as NSArray, forKey: cacheKey, cost: cost)
        }

        return inputTokens
    }

    public var queryPadTokenIdentifier: Int { queryPadTokenId }
    public var docPadTokenIdentifier: Int { docPadTokenId }
    public var maxSequenceLength: Int { maxLen }
    public func tokenId(for token: String) -> Int? {
        tokenizer.convertTokenToId(token)
    }
}
