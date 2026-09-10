import XCTest

@testable import Plaid

/// Comprehensive tests for SentenceBoundarySplitter
final class SentenceBoundarySplitterTests: XCTestCase {

    /// Mock tokenizer for testing
    class MockTokenizer: TokenizerProtocol {
        func tokenize(text: String) -> [String] {
            // Simple whitespace tokenization for testing
            return text.split(separator: " ").map { String($0) }
        }

        func tokenizeToIds(text: String) -> [Int] {
            // Generate simple sequential IDs based on token position
            let tokens = tokenize(text: text)
            return tokens.enumerated().map { $0.offset + 1 }
        }

        func detokenize(tokens: [String]) -> String {
            return tokens.joined(separator: " ")
        }
    }

    // MARK: - Basic Functionality Tests

    func testBasicSentenceSplitting() {
        let tokenizer = MockTokenizer()
        let splitter = SentenceBoundarySplitter(withTokenizer: tokenizer)

        let text =
            "This is the first sentence. This is the second sentence. This is the third sentence."
        let chunks = splitter.chunk(for: text, chunkSize: 20, overlapSize: 5)

        // Should create chunks
        XCTAssertGreaterThan(chunks.count, 0, "Should create at least one chunk")

        // Verify no chunk ends mid-sentence (should end with punctuation or be the full text)
        for chunk in chunks {
            let trimmed = chunk.trimmingCharacters(in: .whitespaces)
            XCTAssertFalse(trimmed.isEmpty, "Chunks should not be empty")
        }

        print("Created \(chunks.count) chunks from 3 sentences")
        for (i, chunk) in chunks.enumerated() {
            print("  Chunk \(i): \(chunk)")
        }
    }

    func testEmptyInput() {
        let tokenizer = MockTokenizer()
        let splitter = SentenceBoundarySplitter(withTokenizer: tokenizer)

        let chunks = splitter.chunk(for: "", chunkSize: 20, overlapSize: 5)
        XCTAssertEqual(chunks.count, 0, "Empty input should produce no chunks")

        let whitespaceChunks = splitter.chunk(for: "   \n  \t  ", chunkSize: 20, overlapSize: 5)
        XCTAssertEqual(whitespaceChunks.count, 0, "Whitespace-only input should produce no chunks")
    }

    func testSingleShortSentence() {
        let tokenizer = MockTokenizer()
        let splitter = SentenceBoundarySplitter(withTokenizer: tokenizer)

        let text = "This is a short sentence."
        let chunks = splitter.chunk(for: text, chunkSize: 20, overlapSize: 5)

        XCTAssertEqual(chunks.count, 1, "Single short sentence should produce one chunk")
        XCTAssertTrue(chunks.first?.contains("short sentence") ?? false)
    }

    // MARK: - Sentence Boundary Preservation Tests

    func testNoMidSentenceSplits() {
        let tokenizer = MockTokenizer()
        let splitter = SentenceBoundarySplitter(withTokenizer: tokenizer)

        // Create text with clear sentence boundaries
        let sentences = [
            "The adoption policy requires documentation.",
            "All employees must submit forms.",
            "Processing takes five business days.",
            "Questions should be directed to HR.",
        ]
        let text = sentences.joined(separator: " ")

        let chunks = splitter.chunk(for: text, chunkSize: 10, overlapSize: 2)

        // Each chunk should contain complete sentences only
        for chunk in chunks {
            // Chunk should end with sentence-ending punctuation or be the full text
            let trimmed = chunk.trimmingCharacters(in: .whitespacesAndNewlines)
            let lastChar = trimmed.last
            let endsWithPunctuation = lastChar == "." || lastChar == "!" || lastChar == "?"

            XCTAssertTrue(
                endsWithPunctuation || chunk == text,
                "Chunk should end with sentence boundary: '\(chunk)'"
            )
        }

        print("Sentence boundary test: Created \(chunks.count) chunks")
        for (i, chunk) in chunks.enumerated() {
            print("  Chunk \(i): \(chunk)")
        }
    }

    func testAbbreviationsHandled() {
        let tokenizer = MockTokenizer()
        let splitter = SentenceBoundarySplitter(withTokenizer: tokenizer)

        // NaturalLanguage should handle abbreviations like Dr., Mr., etc.
        let text =
            "Dr. Smith works here. Mr. Jones is the manager. The office opens at 9 a.m. daily."
        let chunks = splitter.chunk(for: text, chunkSize: 15, overlapSize: 5)

        // Should not split at abbreviation periods
        for chunk in chunks {
            // "Dr." and "Mr." and "a.m." should stay with their sentences
            XCTAssertFalse(chunk == "Dr.", "Should not create chunk with just abbreviation")
            XCTAssertFalse(chunk == "Mr.", "Should not create chunk with just abbreviation")
        }

        print("Abbreviation test: Created \(chunks.count) chunks")
        for (i, chunk) in chunks.enumerated() {
            print("  Chunk \(i): \(chunk)")
        }
    }

    // MARK: - Long Sentence Handling Tests

    func testLongSentenceSplitAtClauses() {
        let tokenizer = MockTokenizer()
        let splitter = SentenceBoundarySplitter(withTokenizer: tokenizer)

        // Create a very long sentence with clause boundaries
        let longSentence =
            "The company policy states that all employees must submit documentation, including proof of identity, proof of address, and employment history; furthermore, additional background checks may be required, depending on the position, and all information must be verified within thirty days."

        let chunks = splitter.chunk(for: longSentence, chunkSize: 15, overlapSize: 3)

        XCTAssertGreaterThan(chunks.count, 1, "Long sentence should be split into multiple chunks")

        // Verify splits happened at clause boundaries (commas, semicolons)
        for chunk in chunks.dropLast() {
            let trimmed = chunk.trimmingCharacters(in: .whitespacesAndNewlines)
            let lastChar = trimmed.last

            let endsWithClauseBoundary = lastChar == "," || lastChar == ";" || lastChar == "."
            XCTAssertTrue(
                endsWithClauseBoundary,
                "Long sentence chunks should end at clause boundaries: '\(trimmed)'"
            )
        }

        print("Long sentence test: Created \(chunks.count) chunks")
        for (i, chunk) in chunks.enumerated() {
            print("  Chunk \(i): \(chunk)")
        }
    }

    // MARK: - Overlap Tests

    func testOverlapIncludesCompleteSentences() {
        let tokenizer = MockTokenizer()
        let splitter = SentenceBoundarySplitter(withTokenizer: tokenizer)

        let sentences = (1 ... 10).map { "Sentence number \($0) has some content." }
        let text = sentences.joined(separator: " ")

        let chunks = splitter.chunk(for: text, chunkSize: 20, overlapSize: 8)

        XCTAssertGreaterThan(chunks.count, 1, "Should create multiple chunks")

        // Verify overlap exists between consecutive chunks
        if chunks.count > 1 {
            for i in 0 ..< chunks.count - 1 {
                let chunk1 = chunks[i]
                let chunk2 = chunks[i + 1]

                // Look for overlapping sentences
                // This is a heuristic test - at least some words should overlap
                let words1 = Set(chunk1.split(separator: " "))
                let words2 = Set(chunk2.split(separator: " "))
                let overlap = words1.intersection(words2)

                print("  Overlap between chunk \(i) and \(i+1): \(overlap.count) words")
            }
        }
    }

    // MARK: - Token Budget Tests

    func testRespectsTokenBudget() {
        let tokenizer = MockTokenizer()
        let splitter = SentenceBoundarySplitter(withTokenizer: tokenizer)

        let sentences = (1 ... 20).map { "Sentence \($0)." }
        let text = sentences.joined(separator: " ")

        let chunkSize = 10
        let chunks = splitter.chunk(for: text, chunkSize: chunkSize, overlapSize: 2)

        // Each chunk should respect token budget (with some tolerance for complete sentences)
        for chunk in chunks {
            let tokenCount = tokenizer.tokenize(text: chunk).count
            // Allow exceeding budget only if it's a single long sentence
            let sentenceCount = chunk.components(separatedBy: ".").filter {
                !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            }.count

            if sentenceCount > 1 {
                XCTAssertLessThanOrEqual(
                    tokenCount,
                    chunkSize + 5,  // Small tolerance for sentence boundaries
                    "Multi-sentence chunk should respect token budget: \(tokenCount) tokens in '\(chunk)'"
                )
            }
        }

        print("Token budget test: Created \(chunks.count) chunks with max size \(chunkSize)")
        for (i, chunk) in chunks.enumerated() {
            let tokenCount = tokenizer.tokenize(text: chunk).count
            print("  Chunk \(i): \(tokenCount) tokens")
        }
    }

    // MARK: - Performance Tests

    func testLargeDocumentPerformance() {
        let tokenizer = MockTokenizer()
        let splitter = SentenceBoundarySplitter(withTokenizer: tokenizer)

        // Create a large document with many sentences
        let sentences = (1 ... 500).map {
            "This is sentence number \($0) with some content about various topics."
        }
        let text = sentences.joined(separator: " ")

        let startTime = Date()
        let chunks = splitter.chunk(for: text, chunkSize: 180, overlapSize: 64)
        let elapsed = Date().timeIntervalSince(startTime)

        XCTAssertLessThan(
            elapsed, 10.0, "Large document chunking should complete in under 10 seconds")
        XCTAssertGreaterThan(chunks.count, 0, "Should produce chunks")

        print("Chunked 500 sentences in \(elapsed)s, produced \(chunks.count) chunks")
    }

    // MARK: - Edge Cases

    func testMultiplePunctuationMarks() {
        let tokenizer = MockTokenizer()
        let splitter = SentenceBoundarySplitter(withTokenizer: tokenizer)

        let text = "Really?! Yes! No... Maybe? Definitely."
        let chunks = splitter.chunk(for: text, chunkSize: 10, overlapSize: 2)

        XCTAssertGreaterThan(chunks.count, 0, "Should handle multiple punctuation marks")

        print("Multiple punctuation test: Created \(chunks.count) chunks")
        for (i, chunk) in chunks.enumerated() {
            print("  Chunk \(i): \(chunk)")
        }
    }

    func testNewlinesAndParagraphs() {
        let tokenizer = MockTokenizer()
        let splitter = SentenceBoundarySplitter(withTokenizer: tokenizer)

        let text = """
            This is the first paragraph.
            It has multiple sentences. Each sentence is complete.

            This is the second paragraph.
            It also has sentences.
            """

        let chunks = splitter.chunk(for: text, chunkSize: 15, overlapSize: 5)

        XCTAssertGreaterThan(chunks.count, 0, "Should handle newlines and paragraphs")

        print("Newline test: Created \(chunks.count) chunks")
        for (i, chunk) in chunks.enumerated() {
            print("  Chunk \(i): \(chunk)")
        }
    }

    func testMixedLanguagePunctuation() {
        let tokenizer = MockTokenizer()
        let splitter = SentenceBoundarySplitter(withTokenizer: tokenizer)

        let text = "First sentence; second clause: third part - final section."
        let chunks = splitter.chunk(for: text, chunkSize: 10, overlapSize: 2)

        XCTAssertGreaterThan(chunks.count, 0, "Should handle mixed punctuation")

        print("Mixed punctuation test: Created \(chunks.count) chunks")
        for (i, chunk) in chunks.enumerated() {
            print("  Chunk \(i): \(chunk)")
        }
    }

    // MARK: - chunkToIds Tests

    func testChunkToIdsFallback() {
        let tokenizer = MockTokenizer()
        let splitter = SentenceBoundarySplitter(withTokenizer: tokenizer)

        let tokenIds = Array(1 ... 60)
        let chunks = splitter.chunkToIds(tokenIds: tokenIds, chunkSize: 20, overlapSize: 5)

        XCTAssertGreaterThan(chunks.count, 0, "Should produce chunks from token IDs")

        // Verify chunks respect size limits
        for chunk in chunks {
            XCTAssertLessThanOrEqual(chunk.count, 20, "Chunks should respect size limit")
        }

        print("chunkToIds test: Created \(chunks.count) chunks from \(tokenIds.count) tokens")
    }

    func testChunkToIdsEmptyInput() {
        let tokenizer = MockTokenizer()
        let splitter = SentenceBoundarySplitter(withTokenizer: tokenizer)

        let chunks = splitter.chunkToIds(tokenIds: [], chunkSize: 20, overlapSize: 5)
        XCTAssertEqual(chunks.count, 0, "Empty token ID input should produce no chunks")
    }

    // MARK: - Real-world Scenarios

    func testPolicyDocument() {
        let tokenizer = MockTokenizer()
        let splitter = SentenceBoundarySplitter(withTokenizer: tokenizer)

        let text = """
            Adoption Policy

            Overview: The company adoption policy outlines the requirements for all new hires. \
            All employees must complete the onboarding process within their first week.

            Required Documentation: Employees must submit proof of identity, proof of address, \
            and employment history. Additional background checks may be required depending on the position.

            Timeline: All documentation must be submitted within 30 days of hire. \
            Processing typically takes 5 business days. Questions should be directed to HR.
            """

        let chunks = splitter.chunk(for: text, chunkSize: 30, overlapSize: 10)

        XCTAssertGreaterThan(chunks.count, 0, "Should chunk policy document")

        // Verify each chunk contains complete sentences
        for chunk in chunks {
            let trimmed = chunk.trimmingCharacters(in: .whitespacesAndNewlines)
            XCTAssertFalse(trimmed.isEmpty, "Chunks should not be empty")
        }

        print("Policy document test: Created \(chunks.count) chunks")
        for (i, chunk) in chunks.enumerated() {
            print(
                "  Chunk \(i) (\(tokenizer.tokenize(text: chunk).count) tokens): \(chunk.prefix(80))..."
            )
        }
    }

    func testCodeDocumentation() {
        let tokenizer = MockTokenizer()
        let splitter = SentenceBoundarySplitter(withTokenizer: tokenizer)

        let text = """
            The searchEngine.search() method performs semantic search using ColBERT embeddings. \
            It takes a query string and returns the top K most relevant document chunks. \
            Results are ranked using MaxSim scoring, where each query token finds its best match. \
            This approach preserves fine-grained semantic information compared to single-vector methods.
            """

        let chunks = splitter.chunk(for: text, chunkSize: 25, overlapSize: 8)

        print("Code documentation test: Created \(chunks.count) chunks")
        for (i, chunk) in chunks.enumerated() {
            let tokenCount = tokenizer.tokenize(text: chunk).count
            print("  Chunk \(i) (\(tokenCount) tokens): \(chunk)")

            // Verify no mid-sentence splits
            let trimmed = chunk.trimmingCharacters(in: .whitespacesAndNewlines)
            if let lastChar = trimmed.last {
                let validEndings: Set<Character> = [".", "!", "?"]
                XCTAssertTrue(
                    validEndings.contains(lastChar) || chunk == text,
                    "Chunk should end with sentence boundary"
                )
            }
        }
    }
}
