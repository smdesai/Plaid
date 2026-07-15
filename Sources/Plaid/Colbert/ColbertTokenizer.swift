import Foundation
import HuggingFace
import Tokenizers

/// Errors that can occur when using ColbertTokenizer
public enum ColbertTokenizerError: Error, LocalizedError {
    case invalidTokenizerType
    case invalidRepositoryId(String)

    public var errorDescription: String? {
        switch self {
        case .invalidTokenizerType:
            return "The loaded tokenizer is not a Tokenizer."
        case .invalidRepositoryId(let id):
            return "Invalid repository ID: \(id)"
        }
    }
}

/// A tokenizer that uses Hugging Face's Tokenizer from swift-tokenizers
/// to load tokenizers from the Hub, specifically designed for ColBERT models.
public class ColbertTokenizer: TokenizerProtocol {
    private let tokenizer: any Tokenizer
    private let maxLen = 256  // CoreML model

    private let queryTokenId: Int
    private let docTokenId: Int
    private let queryPadTokenId: Int
    private let docPadTokenId: Int

    /// Initialize from a pretrained model on Hugging Face Hub
    public static func from(pretrained modelId: String) async throws -> ColbertTokenizer {
        guard let repoId = Repo.ID(rawValue: modelId) else {
            throw ColbertTokenizerError.invalidRepositoryId(modelId)
        }
        let client = HubClient.default
        let directory = try await client.downloadSnapshot(
            of: repoId,
            matching: ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
        )
        return try await from(modelFolder: directory)
    }

    /// Initialize from a local model folder
    public static func from(modelFolder: URL) async throws -> ColbertTokenizer {
        // swift-tokenizers 0.7.x returns the public `Tokenizer` protocol directly;
        // the concrete `PreTrainedTokenizer` is now `package`-internal and not visible here.
        let tokenizer = try await AutoTokenizer.from(modelFolder: modelFolder)
        return try ColbertTokenizer(tokenizer: tokenizer)
    }

    private init(tokenizer: any Tokenizer) throws {
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
    }

    public func tokenize(text: String) -> [String] {
        // Use the tokenizer's tokenize method. swift-tokenizers 0.7.x throws on
        // failure; TokenizerProtocol is non-throwing, so fall back to empty.
        return (try? tokenizer.tokenize(text: text)) ?? []
    }

    public func detokenize(tokens: [String]) -> String {
        // Convert tokens to IDs and decode
        let ids = tokenizer.convertTokensToIds(tokens).compactMap { $0 }
        return (try? tokenizer.decode(tokens: ids, skipSpecialTokens: false)) ?? ""
    }

    /// Tokenize text and return token IDs
    public func tokenizeToIds(text: String) -> [Int] {
        // Use the tokenizer's encode method with no special tokens
        // (we'll add them manually in buildModelTokens)
        return (try? tokenizer.encode(text: text, addSpecialTokens: false)) ?? []
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

    /// Build model-ready token sequence from pre-tokenized IDs (performance optimized)
    /// Skips tokenization and directly adds special tokens and padding
    public func buildModelTokensFromIds(tokenIds: [Int], isQuery: Bool) -> [Int] {
        var tokens = tokenIds

        let prefixTokenCount = 2  // Account for [BOS] and [Q]/[D] tokens

        if tokens.count + prefixTokenCount > maxLen {
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
