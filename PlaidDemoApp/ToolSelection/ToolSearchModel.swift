import Foundation
import Plaid

/// A scored tool, ready for display.
struct ToolMatch: Identifiable {
    let tool: IndexedTool
    /// Raw ColBERT MaxSim score (used for ranking).
    let rawScore: Float
    /// `rawScore` normalized by the number of real query tokens, clamped to 0...1.
    let matchFraction: Float

    var id: UUID { tool.id }
    var matchPercent: Int { Int((matchFraction * 100).rounded()) }
}

/// Drives the ToolSelection screen: loads the Quant4 ColBERT model, builds an
/// in-memory embedding index over the tool descriptions, and scores queries with
/// late-interaction (MaxSim) against that index.
///
/// The corpus is small (151 short descriptions) so there is no need for the on-disk
/// Plaid IVF index or ObjectBox used by the main demo app — everything stays in memory.
@MainActor
final class ToolSearchModel: ObservableObject {
    enum Phase: Equatable {
        case loading
        case indexing(done: Int, total: Int)
        case ready
        case failed(String)
    }

    @Published var query: String = ""
    @Published private(set) var phase: Phase = .loading
    @Published private(set) var results: [ToolMatch] = []
    /// The full catalog grouped by domain, for browsing when there's no active query.
    @Published private(set) var catalog: [ToolDomainGroup] = []

    private var colbert: ColbertModel?
    private var index: [(tool: IndexedTool, embeddings: [[Float]])] = []
    private var searchTask: Task<Void, Never>?

    private let embeddingDimension = 128
    private let modelId = "LiquidAI/LFM2-ColBERT-350M"
    private let topK = 10

    /// Loads the tokenizer + Quant4 model and indexes every tool description.
    /// Safe to call repeatedly (no-op once initialized).
    func bootstrap() async {
        guard colbert == nil else { return }
        do {
            phase = .loading

            // Tokenizer is downloaded from Hugging Face on first launch, then cached.
            let tokenizer = try await ColbertTokenizer.from(pretrained: modelId)

            // Index with the 4-bit quantized LFM2 ColBERT model bundled in the Plaid package.
            let generator = try LFM2ColbertEmbeddingGenerator(
                tokenizer: tokenizer,
                modelResourceName: "LFM2ColbertQuant4"
            )

            let model = ColbertModel(
                generator: generator,
                configuration: .init(
                    batchSize: 16,
                    embeddingDimension: embeddingDimension,
                    queryLength: 32,
                    documentLength: 64
                ),
                chunker: nil  // descriptions are short — single-pass encode, no chunking
            )
            self.colbert = model

            try await buildIndex(using: model)
            phase = .ready
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    private func buildIndex(using model: ColbertModel) async throws {
        let tools = ToolCatalog.load()
        guard !tools.isEmpty else {
            throw NSError(
                domain: "ToolSelection", code: 1,
                userInfo: [
                    NSLocalizedDescriptionKey: "No tool definitions found in the app bundle."
                ])
        }

        catalog = ToolCatalog.grouped(tools)

        var built: [(tool: IndexedTool, embeddings: [[Float]])] = []
        built.reserveCapacity(tools.count)
        phase = .indexing(done: 0, total: tools.count)

        for (offset, tool) in tools.enumerated() {
            let embeddings = try model.encode(sentence: tool.description, isQuery: false)
            built.append((tool, embeddings))

            if offset % 5 == 0 || offset == tools.count - 1 {
                phase = .indexing(done: offset + 1, total: tools.count)
            }
            // Yield so SwiftUI can render progress between Core ML calls.
            await Task.yield()
        }

        self.index = built
    }

    /// Debounced entry point — call whenever `query` changes.
    func queryChanged() {
        searchTask?.cancel()

        let text = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else {
            results = []
            return
        }
        guard case .ready = phase, let model = colbert else { return }

        searchTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: 250_000_000)
            if Task.isCancelled { return }
            await self?.runSearch(text: text, model: model)
        }
    }

    func clear() {
        searchTask?.cancel()
        query = ""
        results = []
    }

    private func runSearch(text: String, model: ColbertModel) async {
        do {
            let queryEmbedding = try model.encode(sentence: text, isQuery: true)
            // Padding tokens encode to zero vectors and contribute 0 to MaxSim;
            // normalize the score by the count of real (non-zero) query tokens.
            let realTokens = max(
                queryEmbedding.filter { vector in vector.contains { $0 != 0 } }.count, 1)

            var scored: [ToolMatch] = []
            scored.reserveCapacity(index.count)

            for entry in index {
                if Task.isCancelled { return }
                let raw = try model.similarity(query: queryEmbedding, document: entry.embeddings)
                let fraction = min(max(raw / Float(realTokens), 0), 1)
                scored.append(
                    ToolMatch(tool: entry.tool, rawScore: raw, matchFraction: fraction))
            }

            scored.sort { $0.rawScore > $1.rawScore }
            if Task.isCancelled { return }
            results = Array(scored.prefix(topK))
        } catch {
            // Keep the previous results on a transient scoring error.
        }
    }
}
