# Plaid (Swift + MLX)

A Swift Package Manager module that reimplements the public FastPlaid indexing/search API without the PyO3/Torch dependency. The implementation targets iOS (17+) and macOS (13.3+) and can be used from Swift code. MLX tensor helpers are available on both platforms when `mlx-swift` is present.

## Highlights
- Same entry-point naming as the Python bridge (`create`, `update`, `preloadIndex`, `loadAndSearch`, `delete`, `initializeTorch`).
- Stores IVF/PQ artefacts (`centroids.bin`, `bucket_cutoffs.bin`, `bucket_weights.bin`, chunked codes/residuals, IVF postings) in a Swift-specific layout.
- Uses MLX for candidate selection and per-token residual reconstruction; falls back to legacy averages when PQ artefacts are missing.
- CLI demos (`PlaidCLI quickstart` and `update`) mirror the Python quick start flows.

## Building
```
cd swift/Plaid
swift build
```
The package declares a dependency on [`mlx-swift`](https://github.com/ml-explore/mlx-swift). SwiftPM will fetch it automatically during the first build on a machine with network access. On systems without MLX available the Array-based overloads remain usable.

## CLI Examples
Run the bundled executable to mirror the Python documentation flows:
```
swift run PlaidCLI quickstart
swift run PlaidCLI update
```

## Storage format
`Plaid.create` materializes:
- `plan.json`, `metadata.json`
- `centroids.bin`, `bucket_cutoffs.bin`, `bucket_weights.bin`, `avg_residual.bin`
- Per-chunk `chunk_X.codes.bin`, `chunk_X.residuals.bin`, `doclens.X.json`, `chunk_X.metadata.json`
- `ivf.bin`, `ivf_lengths.bin`
- A JSON summary (`plaid_index.json`) for environments that still rely on averaged embeddings.

## Behavioural notes
- `loadAndSearch` consumes the Ivy/PQ artefacts when available and falls back to the legacy JSON summary otherwise.
- `update` and `delete` are currently unimplemented on the Swift backend.
- Residual decoding uses a simplified bit-unpacking path; metrics may diverge slightly from the Rust reference until the MLX equivalent is finalized.

## ColBERT utilities
The package also exposes a Swift port of the ColBERT helpers from `pylate-rs`, allowing any embedding backend to feed Plaid.

- `ColbertEmbeddingGenerator` defines how token embeddings and attention masks are produced.
- `SentenceChunker` controls batching behaviour; `FixedSizeChunker` mirrors the Rust implementation.
- `ColbertModel` handles batching, normalization, padding, and similarity scoring.
- `hierarchicalPooling(documents:poolFactor:)` performs the hierarchical token pooling used by the Rust runtime.

```swift
struct MyGenerator: ColbertEmbeddingGenerator {
    func generateEmbeddings(for sentence: String, isQuery: Bool, maxLength: Int) throws -> ColbertEmbeddingBatch {
        // Produce token embeddings and attention masks using your preferred model.
    }
}

let colbert = ColbertModel(
    generator: MyGenerator(),
    configuration: .init(embeddingDimension: 128)
)

let query = try colbert.encode(sentence: "What is ColBERT?", isQuery: true)
let document = try colbert.encode(sentence: "ColBERT uses late interaction.", isQuery: false)
let score = try colbert.similarity(query: query, document: document)
```

When targeting macOS/iOS you can use the bundled Core ML model via `LFM2ColbertEmbeddingGenerator` and the Hugging Face powered tokenizer:

```swift
let tokenizer = try await PreTrainedColbertTokenizer.from(pretrained: "LiquidAI/LFM2-ColBERT-350M")
let generator = try LFM2ColbertEmbeddingGenerator(tokenizer: tokenizer)
let colbert = ColbertModel(
    generator: generator,
    configuration: .init(embeddingDimension: 128)
)

let query = try colbert.encode(sentence: "What is ColBERT?", isQuery: true)
```
