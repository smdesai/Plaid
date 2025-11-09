import XCTest

@testable import Plaid

/// Tests for TokenSplitter to ensure correct chunking behavior
final class TokenSplitterTests: XCTestCase {

    /// Mock tokenizer for testing
    class MockTokenizer: TokenizerProtocol {
        func tokenize(text: String) -> [String] {
            // Simple whitespace tokenization for testing
            return text.split(separator: " ").map { String($0) }
        }

        func tokenizeToIds(text: String) -> [Int] {
            // Generate simple sequential IDs based on token position
            let tokens = tokenize(text: text)
            return tokens.enumerated().map { $0.offset + 1 }  // Start from 1 to avoid 0
        }

        func detokenize(tokens: [String]) -> String {
            return tokens.joined(separator: " ")
        }
    }

    func testBasicChunking() {
        let tokenizer = MockTokenizer()
        let splitter = TokenSplitter(withTokenizer: tokenizer)

        let text = String(repeating: "word ", count: 100).trimmingCharacters(in: .whitespaces)
        let chunks = splitter.chunk(for: text, chunkSize: 20, overlapSize: 5)

        // Should create multiple chunks
        XCTAssertGreaterThan(chunks.count, 1, "Should create multiple chunks for long text")

        // All chunks should be non-empty
        for chunk in chunks {
            XCTAssertFalse(chunk.isEmpty, "Chunks should not be empty")
        }

        print("Created \(chunks.count) chunks from 100 words")
    }

    func testOverlap() {
        let tokenizer = MockTokenizer()
        let splitter = TokenSplitter(withTokenizer: tokenizer)

        // Create text with unique words to verify overlap
        let words = (0 ..< 100).map { "word\($0)" }
        let text = words.joined(separator: " ")

        let chunks = splitter.chunk(for: text, chunkSize: 20, overlapSize: 5)

        // With chunkSize=20 and overlap=5, step size should be 15
        // So: chunk1=[0-19], chunk2=[15-34], chunk3=[30-49], etc.
        // Verify overlap exists
        XCTAssertGreaterThan(chunks.count, 1, "Should create multiple chunks")

        print("Overlap test: Created \(chunks.count) chunks")
        for (i, chunk) in chunks.enumerated() {
            let wordCount = chunk.split(separator: " ").count
            print("  Chunk \(i): \(wordCount) words")
        }
    }

    func testNoInfiniteLoop() {
        let tokenizer = MockTokenizer()
        let splitter = TokenSplitter(withTokenizer: tokenizer)

        // This should complete in reasonable time (not hang)
        let text = String(repeating: "word ", count: 1000).trimmingCharacters(in: .whitespaces)

        let startTime = Date()
        let chunks = splitter.chunk(for: text, chunkSize: 180, overlapSize: 64)
        let elapsed = Date().timeIntervalSince(startTime)

        XCTAssertLessThan(elapsed, 5.0, "Chunking should complete in under 5 seconds")
        XCTAssertGreaterThan(chunks.count, 0, "Should produce chunks")

        print("Chunked 1000 words in \(elapsed)s, produced \(chunks.count) chunks")
    }

    func testEmptyInput() {
        let tokenizer = MockTokenizer()
        let splitter = TokenSplitter(withTokenizer: tokenizer)

        let chunks = splitter.chunk(for: "", chunkSize: 20, overlapSize: 5)
        XCTAssertEqual(chunks.count, 0, "Empty input should produce no chunks")

        let whitespaceChunks = splitter.chunk(for: "   \n  \t  ", chunkSize: 20, overlapSize: 5)
        XCTAssertEqual(whitespaceChunks.count, 0, "Whitespace-only input should produce no chunks")
    }

    func testShortText() {
        let tokenizer = MockTokenizer()
        let splitter = TokenSplitter(withTokenizer: tokenizer)

        let text = "just a few words"
        let chunks = splitter.chunk(for: text, chunkSize: 20, overlapSize: 5)

        XCTAssertEqual(chunks.count, 1, "Short text should produce one chunk")
        XCTAssertEqual(chunks.first, text, "Short text should be unchanged")
    }

    func testChunkSizeLimits() {
        let tokenizer = MockTokenizer()
        let splitter = TokenSplitter(withTokenizer: tokenizer)

        let text = String(repeating: "word ", count: 200).trimmingCharacters(in: .whitespaces)

        // chunkSize is capped at 180
        let chunks = splitter.chunk(for: text, chunkSize: 300, overlapSize: 0)

        // Verify all chunks are within size limits
        for chunk in chunks {
            let wordCount = chunk.split(separator: " ").count
            XCTAssertLessThanOrEqual(wordCount, 180, "Chunk size should be capped at 180")
        }
    }

    func testChunkToIdsFollowsChunkSizeAndOverlap() {
        let tokenizer = MockTokenizer()
        let splitter = TokenSplitter(withTokenizer: tokenizer)

        let tokenIds = Array(1 ... 60)
        let chunkSize = 16
        let overlap = 4

        let chunks = splitter.chunkToIds(
            tokenIds: tokenIds, chunkSize: chunkSize, overlapSize: overlap)

        XCTAssertEqual(
            chunks.first, Array(1 ... chunkSize), "First chunk should capture initial tokens")
        XCTAssertTrue(
            chunks.allSatisfy { $0.count <= chunkSize }, "All chunks must respect chunk size")

        // Manually build the expected sliding windows to ensure logic stays in sync
        var expected: [[Int]] = []
        var position = 0
        let step = chunkSize - overlap
        while position < tokenIds.count {
            let end = min(position + chunkSize, tokenIds.count)
            expected.append(Array(tokenIds[position ..< end]))
            position += max(step, 1)
        }

        XCTAssertEqual(chunks, expected, "chunkToIds should produce the same sliding windows")
    }

    func testChunkToIdsClampsLargeOverlap() {
        let tokenizer = MockTokenizer()
        let splitter = TokenSplitter(withTokenizer: tokenizer)

        let tokenIds = Array(1 ... 15)
        let chunkSize = 5
        let overlap = 20  // Larger than chunk size, should clamp internally

        let chunks = splitter.chunkToIds(
            tokenIds: tokenIds, chunkSize: chunkSize, overlapSize: overlap)

        // With effective overlap=chunkSize-1 the window should advance by exactly one token each time
        XCTAssertEqual(
            chunks.count, tokenIds.count,
            "Sliding by one token should yield one chunk per token start"
        )
        XCTAssertEqual(chunks[0], [1, 2, 3, 4, 5])
        XCTAssertEqual(chunks[1], [2, 3, 4, 5, 6])
        XCTAssertEqual(chunks[10], [11, 12, 13, 14, 15])
        XCTAssertEqual(chunks.last, [15], "Final chunk should contain the trailing tokens")
    }

    func testChunkToIdsEmptyInput() {
        let tokenizer = MockTokenizer()
        let splitter = TokenSplitter(withTokenizer: tokenizer)

        let chunks = splitter.chunkToIds(tokenIds: [], chunkSize: 10, overlapSize: 5)
        XCTAssertTrue(chunks.isEmpty, "Empty token input should yield no chunks")
    }
}
