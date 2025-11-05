import Foundation
import Hub
import Tokenizers

/// Errors that can occur when using PreTrainedColbertTokenizer
public enum PreTrainedColbertTokenizerError: Error, LocalizedError {
    case invalidTokenizerType

    public var errorDescription: String? {
        switch self {
        case .invalidTokenizerType:
            return "The loaded tokenizer is not a PreTrainedTokenizer."
        }
    }
}

/// A tokenizer that uses Hugging Face's PreTrainedTokenizer from swift-transformers
/// to load tokenizers from the Hub, specifically designed for ColBERT models.
public class PreTrainedColbertTokenizer: TokenizerProtocol {
    private let tokenizer: PreTrainedTokenizer
    private let maxLen = 256  // CoreML model

    private let queryTokenId: Int
    private let docTokenId: Int
    private let queryPadTokenId: Int
    private let docPadTokenId: Int

    /// Initialize from a pretrained model on Hugging Face Hub
    public static func from(pretrained modelId: String) async throws -> PreTrainedColbertTokenizer {
        guard
            let tokenizer = try await AutoTokenizer.from(pretrained: modelId)
                as? PreTrainedTokenizer
        else {
            throw PreTrainedColbertTokenizerError.invalidTokenizerType
        }
        return try PreTrainedColbertTokenizer(tokenizer: tokenizer)
    }

    /// Initialize from a local model folder
    public static func from(modelFolder: URL) async throws -> PreTrainedColbertTokenizer {
        guard
            let tokenizer = try await AutoTokenizer.from(modelFolder: modelFolder)
                as? PreTrainedTokenizer
        else {
            throw PreTrainedColbertTokenizerError.invalidTokenizerType
        }
        return try PreTrainedColbertTokenizer(tokenizer: tokenizer)
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
        self.docPadTokenId = tokenizer.convertTokenToId("<|pad|>") ?? 0
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
    public func buildModelTokens(sentence: String, isQuery: Bool) -> [Int] {
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

        return inputTokens
    }

    public var queryPadTokenIdentifier: Int { queryPadTokenId }
    public var docPadTokenIdentifier: Int { docPadTokenId }
    public var maxSequenceLength: Int { maxLen }
    public func tokenId(for token: String) -> Int? {
        tokenizer.convertTokenToId(token)
    }
}
