# Plaid Swift

**Fast, elegant semantic search powered by ColBERT embeddings and optimized indexing.**

Plaid Swift is a native Swift implementation of the Plaid indexing and search system, bringing high-performance semantic search to iOS and macOS applications. Built on MLX for efficient tensor operations and CoreML for on-device embeddings, it enables powerful neural search capabilities without external dependencies.

## Features

- 🚀 **Fast Neural Search** - ColBERT-style late interaction for accurate semantic matching
- 🤖 **Dual Model Support** - Choose between LFM2-ColBERT (128-dim) and MXBAI-Edge (64-dim)
- 📦 **Compact Indexes** - Product quantization with 1-8 bit compression
- 🔄 **Intelligent Chunking** - Automatic handling of large documents with overlap
- 💻 **Native Swift** - No Python or PyTorch dependencies
- 📱 **iOS & macOS** - Runs on-device with CoreML acceleration
- 🎯 **Easy CLI** - Index and search from the command line
- 🔧 **Flexible API** - Use as a library in your Swift projects

## Quick Start

### Installation

```bash
git clone https://github.com/your-org/plaid-swift.git
cd plaid-swift
swift build
```

Requires:
- **Swift 5.9+**
- **macOS 13.3+** or **iOS 17+**
- MLX Swift (fetched automatically by SwiftPM)

---

## Supported Models

Plaid Swift supports two ColBERT embedding models, each optimized for different use cases:

### LFM2-ColBERT (Default)
- **Model ID**: `LiquidAI/LFM2-ColBERT-350M`
- **Embedding Dimension**: 128
- **Best For**: Maximum accuracy, research-grade quality
- **Use Case**: When you need the best possible search quality

```swift
let generator = try LFM2ColbertEmbeddingGenerator(tokenizer: tokenizer)
let config = ColbertModel.Configuration(embeddingDimension: 128, ...)
```

### MXBAI-Edge
- **Model ID**: `mixedbread-ai/mxbai-edge-colbert-v0-32m`
- **Embedding Dimension**: 64
- **Best For**: Faster inference, lower memory usage
- **Use Case**: When you need speed and efficiency

```swift
let generator = try MXBAIEdgeColbertEmbeddingGenerator(tokenizer: tokenizer)
let config = ColbertModel.Configuration(embeddingDimension: 64, ...)
```

### Choosing a Model

| Factor | LFM2-ColBERT | MXBAI-Edge |
|--------|--------------|------------|
| **Accuracy** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Speed** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Memory** | ~512MB | ~256MB |
| **Index Size** | Larger (128-dim) | Smaller (64-dim) |
| **Recommended For** | Production search, Q&A systems | Mobile apps, edge devices |

**Important**: Once you create an index with a specific model, you must use the same model for searching. Mixing models will result in incorrect results.

---

## Swift API Usage

### Creating an Index

Index your documents with ColBERT embeddings:

```swift
import Plaid

// 1. Prepare your documents (each document is an array of token embeddings)
let documents: [[[Float]]] = [
    // Document 0: array of token vectors (shape: [num_tokens, embedding_dim])
    [[0.1, 0.2, ...], [0.3, 0.4, ...], ...],
    // Document 1
    [[0.5, 0.6, ...], [0.7, 0.8, ...], ...],
    // ... more documents
]

// 2. Generate or provide centroids for quantization
let centroids: [[Float]] = generateCentroids(embeddingDim: 128, nbits: 2)

// 3. Create the index
try Plaid.create(
    indexURL: URL(fileURLWithPath: "/path/to/index"),
    device: "cpu",
    embeddingDim: 128,
    nbits: 2,                    // Compression: 2^nbits clusters
    embeddings: documents,
    centroids: centroids,
    batchSize: 64,
    seed: 42
)
```

### Searching an Index

Search for semantically similar documents:

```swift
// 1. Prepare your query embeddings
let queries: [[[Float]]] = [
    // Query 0: array of token vectors
    [[0.2, 0.3, ...], [0.4, 0.5, ...], ...]
]

// 2. Configure search parameters
let searchParams = SearchParameters(
    batchSize: 32,
    nFullScores: 1024,          // Candidates to score fully
    topK: 10,                   // Number of results to return
    nIvfProbe: 8                // IVF partitions to probe
)

// 3. Search
let results = try Plaid.loadAndSearch(
    indexURL: URL(fileURLWithPath: "/path/to/index"),
    device: "cpu",
    queries: queries,
    searchParameters: searchParams,
    showProgress: false,
    preloadIndex: false
)

// 4. Process results
for result in results {
    print("Query \(result.queryId):")
    for (docId, score) in zip(result.passageIds, result.scores) {
        print("  Document \(docId): score \(score)")
    }
}
```

### Using ColBERT Embeddings

Generate embeddings using the built-in CoreML models:

#### Option 1: LFM2-ColBERT (128-dimensional, default)

```swift
import Plaid

// 1. Load tokenizer and model
let tokenizer = try await ColbertTokenizer.from(
    pretrained: "LiquidAI/LFM2-ColBERT-350M"
)
let generator = try LFM2ColbertEmbeddingGenerator(tokenizer: tokenizer)
let chunker = TokenSplitter(withTokenizer: tokenizer)

// 2. Create ColBERT model
let colbert = ColbertModel(
    generator: generator,
    configuration: .init(
        embeddingDimension: 128,  // LFM2 uses 128-dimensional embeddings
        queryLength: 32,
        documentLength: 180
    ),
    chunker: chunker  // Handles large documents automatically
)

// 3. Encode text
let queryEmbedding = try colbert.encode(sentence: "What is machine learning?", isQuery: true)
let docEmbedding = try colbert.encode(sentence: "Machine learning is a subset of AI...", isQuery: false)

// 4. Compute similarity
let score = try colbert.similarity(query: queryEmbedding, document: docEmbedding)
print("Similarity: \(score)")
```

#### Option 2: MXBAI-Edge (64-dimensional, faster)

```swift
import Plaid

// 1. Load tokenizer and model
let tokenizer = try await ColbertTokenizer.from(
    pretrained: "mixedbread-ai/mxbai-edge-colbert-v0-32m"
)
let generator = try MXBAIEdgeColbertEmbeddingGenerator(tokenizer: tokenizer)
let chunker = TokenSplitter(withTokenizer: tokenizer)

// 2. Create ColBERT model
let colbert = ColbertModel(
    generator: generator,
    configuration: .init(
        embeddingDimension: 64,  // MXBAI uses 64-dimensional embeddings
        queryLength: 32,
        documentLength: 180
    ),
    chunker: chunker
)

// 3. Encode text
let queryEmbedding = try colbert.encode(sentence: "What is machine learning?", isQuery: true)
let docEmbedding = try colbert.encode(sentence: "Machine learning is a subset of AI...", isQuery: false)

// 4. Compute similarity
let score = try colbert.similarity(query: queryEmbedding, document: docEmbedding)
print("Similarity: \(score)")
```

### MLX Integration

When using MLX arrays directly:

```swift
import MLX

let mlxEmbeddings: [MLXArray] = [...]  // Your embeddings as MLX arrays
let mlxCentroids = MLXArray(...)       // Centroids as MLX array

try Plaid.create(
    indexURL: indexURL,
    device: "cpu",
    embeddingDim: 128,
    nbits: 2,
    embeddings: mlxEmbeddings,     // MLXArray inputs
    centroids: mlxCentroids,
    batchSize: 64
)
```

---

## CLI Usage

The `PlaidCLI` tool provides a powerful command-line interface for indexing and searching.

### Available Commands

```bash
PlaidCLI <command> [options]

Commands:
  demo         End-to-end demo: create index and search
  update       Add new documents to an existing index
  delete       Remove documents from an existing index
  quickstart   Test with bundled fixtures
  tokenize     Tokenize text with ColBERT tokenizer
  similarity   Compute similarity between query and document

Available Models:
  lfm2         LFM2-ColBERT (128-dimensional, default)
  mxbai        MXBAI-Edge (64-dimensional, faster)
```

### Model Selection

All indexing and search commands support the `--model` flag to choose between models:

```bash
# Use LFM2 (default, higher accuracy)
PlaidCLI demo --query "..." --files doc.txt --model lfm2

# Use MXBAI (faster, lower memory)
PlaidCLI demo --query "..." --files doc.txt --model mxbai
```

**Important**: You must use the same model for both indexing and searching. If you created an index with `--model mxbai`, you must search it with `--model mxbai`.

---

### `demo` Command

**Create an index from documents and search it.**

#### Basic Usage

```bash
# Index inline text documents
PlaidCLI demo \
  --query "machine learning" \
  --docs "AI and deep learning" "Cooking pasta" "Python programming" \
  --top-k 3
```

#### Index Files

```bash
# Index files instead of inline text
PlaidCLI demo \
  --query "artificial intelligence" \
  --files doc1.txt doc2.md research.txt \
  --top-k 5
```

#### Mix Files and Inline Text

```bash
# Combine both approaches
PlaidCLI demo \
  --query "neural networks" \
  --docs "Inline document about AI" \
  --files article1.txt article2.md \
  --top-k 3
```

#### Save Index for Reuse

```bash
# Create and save index with LFM2 (default)
PlaidCLI demo \
  --query "test query" \
  --files *.txt \
  --keep-index \
  --index-name my_documents

# Create and save index with MXBAI (faster)
PlaidCLI demo \
  --query "test query" \
  --files *.txt \
  --model mxbai \
  --keep-index \
  --index-name my_documents_mxbai

# Output: 💾 Index saved at: /path/to/index/my_documents_mxbai
```

#### Search Existing Index

```bash
# Search previously created index (must match the model used during indexing)
PlaidCLI demo \
  --query "new search query" \
  --index-path /path/to/index/my_documents_mxbai \
  --model mxbai \
  --top-k 10
```

#### All Options

```bash
PlaidCLI demo \
  --query "your search query" \
  --docs "doc1" "doc2" ...              # Inline text documents
  --files "file1.txt" "file2.md" ...    # File paths to index
  --index-path /path/to/existing/index  # Search existing index
  --model [lfm2|mxbai]                   # Model to use (default: lfm2)
  --pretrained "MODEL_ID"                # HuggingFace model ID (overrides --model)
  --top-k 10                             # Number of results (default: 5)
  --nbits 2                              # Quantization bits (default: 2)
  --keep-index                           # Don't delete index after demo
  --index-name "my_index"                # Name for saved index
```

---

### `update` Command

**Add new documents to an existing Plaid index.**

The `update` command allows you to incrementally add new documents to an index without rebuilding from scratch. Documents are automatically encoded with ColBERT and quantized using the existing index's compression settings.

#### Basic Usage

```bash
# Add new documents to an existing index
PlaidCLI update \
  --index-path ~/.plaid/my_index \
  --files new_doc1.txt new_doc2.txt new_doc3.txt
```

#### Add Multiple Files

```bash
# Use glob patterns to add many files at once
PlaidCLI update \
  -i ~/.plaid/documentation \
  -f docs/new/*.md docs/updates/*.txt
```

#### Custom Model and Batch Size

```bash
# Use MXBAI model and larger batch size
PlaidCLI update \
  --index-path ~/.plaid/my_index_mxbai \
  --files new_articles/*.txt \
  --model mxbai \
  --batch-size 128
```

#### All Options

```bash
PlaidCLI update \
  --index-path PATH           # (Required) Path to existing index directory
  --files FILE1 FILE2 ...     # (Required) Files to add to index
  --model [lfm2|mxbai]        # Model to use (must match index, default: lfm2)
  --pretrained MODEL_ID       # HuggingFace model ID (overrides --model)
  --batch-size SIZE           # Batch size for encoding (default: 64)
  --help                      # Show help message
```

**Important**: The model used must match the model that was used to create the original index.

#### Example Workflow

```bash
# 1. Create initial index with some documents
PlaidCLI demo \
  --query "initial query" \
  --files docs/batch1/*.txt \
  --keep-index \
  --index-name my_knowledge_base

# 2. Later, add more documents as they arrive
PlaidCLI update \
  -i ~/.plaid/my_knowledge_base \
  -f docs/batch2/*.txt

# 3. Add even more documents
PlaidCLI update \
  -i ~/.plaid/my_knowledge_base \
  -f docs/batch3/*.txt

# 4. Search the complete index
PlaidCLI demo \
  --query "search across all documents" \
  --index-path ~/.plaid/my_knowledge_base
```

#### Output Example

```
╔══════════════════════════════════════════════════════════════════════╗
║  Plaid Index Update                                                  ║
╚══════════════════════════════════════════════════════════════════════╝

📂 Index: /Users/you/.plaid/my_knowledge_base
📄 Documents to add: 3

⚙️  Loading ColBERT model: LiquidAI/LFM2-ColBERT-350M...
✅ Model loaded

📚 Loading and encoding documents...

  [1/3] new_doc1.txt (1234 chars)
    ✅ Encoded: 156 embeddings

  [2/3] new_doc2.txt (2345 chars)
    ✅ Encoded: 234 embeddings

  [3/3] new_doc3.txt (3456 chars)
    ✅ Encoded: 312 embeddings

✅ Encoded 3 document(s)

📦 Updating index...
✅ Index updated successfully!

💾 Updated index: /Users/you/.plaid/my_knowledge_base
   Added 3 document(s)
```

#### Notes

- **Concurrency Safe**: Uses file locking to prevent concurrent modifications
- **Incremental**: No need to rebuild the entire index
- **Automatic Encoding**: Documents are automatically encoded with ColBERT
- **Large Documents**: Automatically chunks documents that exceed token limits
- **Preserves Settings**: Uses the existing index's quantization settings (nbits, centroids)

---

### `delete` Command

**Remove documents from an existing Plaid index by their document IDs.**

The `delete` command allows you to remove specific documents from an index. Documents are identified by their zero-based index (the order they were added to the index).

#### Basic Usage

```bash
# Delete specific documents by ID
PlaidCLI delete \
  --index-path ~/.plaid/my_index \
  --doc-ids 5 12 23 45
```

#### Delete from File

```bash
# Read document IDs from a file (one ID per line)
PlaidCLI delete \
  --index-path ~/.plaid/my_index \
  --ids-file documents_to_delete.txt
```

Where `documents_to_delete.txt` contains:
```
0
15
23
99
```

#### All Options

```bash
PlaidCLI delete \
  --index-path PATH          # (Required) Path to existing index directory
  --doc-ids ID1 ID2 ...      # Document IDs to delete (space-separated)
  --ids-file FILE            # File containing document IDs (one per line)
  --help                     # Show help message
```

**Note:** You must provide either `--doc-ids` or `--ids-file` (not both).

#### How to Find Document IDs

Document IDs are assigned sequentially starting from 0 in the order documents were added:

```bash
# If you indexed 3 documents initially:
# - doc1.txt → ID 0
# - doc2.txt → ID 1
# - doc3.txt → ID 2

# Then added 2 more documents:
# - doc4.txt → ID 3
# - doc5.txt → ID 4

# To delete doc2.txt and doc4.txt:
PlaidCLI delete -i ~/.plaid/my_index -d 1 3
```

#### Example Workflow

```bash
# 1. Create index with documents
PlaidCLI demo \
  --query "test" \
  --files doc0.txt doc1.txt doc2.txt doc3.txt doc4.txt \
  --keep-index \
  --index-name my_index

# 2. Remove outdated documents (IDs 1 and 3)
PlaidCLI delete \
  -i ~/.plaid/my_index \
  -d 1 3

# 3. Search updated index (now only has docs 0, 2, 4)
PlaidCLI demo \
  --query "updated search" \
  --index-path ~/.plaid/my_index
```

#### Batch Deletion with File

```bash
# Create a file with IDs to delete
cat > to_delete.txt << EOF
5
12
23
45
67
89
EOF

# Delete all at once
PlaidCLI delete \
  -i ~/.plaid/my_index \
  -f to_delete.txt
```

#### Output Example

```
╔══════════════════════════════════════════════════════════════════════╗
║  Plaid Index Deletion                                                ║
╚══════════════════════════════════════════════════════════════════════╝

📂 Index: /Users/you/.plaid/my_index
🗑️  Documents to delete: 4
   IDs: 5, 12, 23, 45

🗑️  Deleting documents...
✅ Deletion complete!

💾 Updated index: /Users/you/.plaid/my_index
   Removed 4 document(s)
```

#### Notes

- **Concurrency Safe**: Uses file locking to prevent concurrent modifications
- **ID Validation**: Invalid IDs are automatically filtered out
- **Deduplication**: Duplicate IDs are automatically removed
- **Rebuilds IVF**: The inverted file index is rebuilt for efficient search
- **Permanent**: Deletion cannot be undone - make sure you have backups!

#### Advanced: Maintaining Document Metadata

Since document IDs are positional, it's recommended to maintain a separate mapping file:

```bash
# Create a mapping file when building index
cat > ~/.plaid/my_index/doc_mapping.json << EOF
{
  "0": {"file": "doc1.txt", "date": "2024-01-01"},
  "1": {"file": "doc2.txt", "date": "2024-01-02"},
  "2": {"file": "doc3.txt", "date": "2024-01-03"}
}
EOF

# Use this to look up which ID to delete
# Then delete by ID
PlaidCLI delete -i ~/.plaid/my_index -d 1
```

---

### `tokenize` Command

**Tokenize text using the ColBERT tokenizer.**

```bash
# Tokenize as query with LFM2 (default)
PlaidCLI tokenize --query "What is AI?"

# Tokenize as query with MXBAI
PlaidCLI tokenize --query "What is AI?" --model mxbai

# Tokenize as document
PlaidCLI tokenize --doc "Artificial intelligence is..." --model lfm2
```

**Output:**
```
Input: What is AI?
Mode: Query

WordPiece tokens:
what, is, ai, ?

Token IDs:
2054, 2003, 9932, 1029

Encoded sequence (with [Q] prefix and padding):
101, 1000, 2054, 2003, 9932, 1029, 102, 0, 0, ...
```

---

### `similarity` Command

**Compute ColBERT similarity between a query and document.**

```bash
# Compute similarity with LFM2 (default)
PlaidCLI similarity \
  --query "machine learning applications" \
  --doc "Machine learning is used in healthcare, finance, and autonomous vehicles."

# Compute similarity with MXBAI
PlaidCLI similarity \
  --query "machine learning applications" \
  --doc "Machine learning is used in healthcare, finance, and autonomous vehicles." \
  --model mxbai
```

**Output:**
```
Loading tokenizer/model: LiquidAI/LFM2-ColBERT-350M (LFM2-ColBERT (128-dim))
=== query embedding ===
=== document embedding ===
=== similarity ===

Query: machine learning applications
Document: Machine learning is used in healthcare...
ColBERT score: 12.3456
```

---

### `quickstart` Command

**Run built-in demo with test fixtures.**

```bash
PlaidCLI quickstart
```

Loads bundled test data and demonstrates index creation and search.

---

## Real-World Examples

### Index Your Documentation

```bash
# Index all markdown files in docs/ with LFM2 (best accuracy)
PlaidCLI demo \
  --query "authentication setup" \
  --files docs/**/*.md \
  --model lfm2 \
  --keep-index \
  --index-name documentation \
  --top-k 5

# Or use MXBAI for faster indexing
PlaidCLI demo \
  --query "authentication setup" \
  --files docs/**/*.md \
  --model mxbai \
  --keep-index \
  --index-name documentation_mxbai \
  --top-k 5
```

### Search Code Comments

```bash
# Index Swift source files with MXBAI (faster for large codebases)
PlaidCLI demo \
  --query "error handling" \
  --files Sources/**/*.swift \
  --model mxbai \
  --keep-index \
  --index-name codebase \
  --top-k 10
```

### Build a Knowledge Base (with Incremental Updates)

```bash
# 1. Create initial index from knowledge base articles
PlaidCLI demo \
  --query "initial query" \
  --files kb/batch1/*.txt \
  --keep-index \
  --index-name knowledge_base

# 2. Add new articles as they're created
PlaidCLI update \
  -i ~/.plaid/knowledge_base \
  -f kb/batch2/*.txt

# 3. Add more articles later
PlaidCLI update \
  -i ~/.plaid/knowledge_base \
  -f kb/batch3/*.txt

# 4. Search the complete knowledge base
PlaidCLI demo --query "how to reset password" --index-path ~/.plaid/knowledge_base
PlaidCLI demo --query "billing questions" --index-path ~/.plaid/knowledge_base
PlaidCLI demo --query "API authentication" --index-path ~/.plaid/knowledge_base
```

### Research Paper Search

```bash
# Index academic papers
PlaidCLI demo \
  --query "transformer architectures" \
  --files papers/*.txt \
  --nbits 4 \
  --top-k 20 \
  --keep-index \
  --index-name research_papers
```

### Maintain a Living Documentation Index

```bash
# 1. Create index of current documentation
PlaidCLI demo \
  --query "initial" \
  --files docs/*.md \
  --keep-index \
  --index-name live_docs

# 2. Track document order in a mapping file
cat > ~/.plaid/live_docs/mapping.txt << EOF
0: docs/intro.md
1: docs/setup.md
2: docs/api.md
3: docs/faq.md
EOF

# 3. Add new documentation as it's written
PlaidCLI update \
  -i ~/.plaid/live_docs \
  -f docs/new_feature.md

# Update mapping: 4: docs/new_feature.md

# 4. Remove deprecated documentation (e.g., old FAQ doc ID 3)
PlaidCLI delete \
  -i ~/.plaid/live_docs \
  -d 3

# 5. Add updated version
PlaidCLI update \
  -i ~/.plaid/live_docs \
  -f docs/faq_v2.md

# 6. Search always gets latest content
PlaidCLI demo \
  --query "how to install" \
  --index-path ~/.plaid/live_docs
```

### Content Moderation Workflow

```bash
# 1. Index user-generated content
PlaidCLI demo \
  --query "test" \
  --files user_content/*.txt \
  --keep-index \
  --index-name user_content

# 2. Identify problematic content IDs (e.g., through manual review)
# Suppose documents 5, 12, and 23 violate guidelines

# 3. Remove them from the index
PlaidCLI delete \
  -i ~/.plaid/user_content \
  -d 5 12 23

# 4. Index is now clean and ready for search
PlaidCLI demo \
  --query "search approved content" \
  --index-path ~/.plaid/user_content
```

---

## Environment Variables

### Index Storage Location

By default, indexes are stored in `.plaid/` in the current directory. Customize with:

```bash
export PLAID_CLI_INDEX_DIR=/path/to/your/indexes

PlaidCLI demo --query "..." --files ... --keep-index --index-name myindex
# Index saved at: /path/to/your/indexes/myindex
```

---

## Advanced Features

### Automatic Document Chunking

Large documents are automatically split into chunks with overlap:

```bash
# Large document (1000s of words) automatically chunked
PlaidCLI demo \
  --query "key concepts" \
  --files very_large_document.txt \
  --top-k 5
```

**Output shows chunking:**
```
📄 Document chunking: split into 7 chunk(s)
  Chunk 1/7: "Beginning of document..."
    → Encoded 180 tokens, total: 180 embeddings
  Chunk 2/7: "...with overlap from previous chunk..."
    → Encoded 180 tokens, total: 360 embeddings
  ...
✅ Chunking complete: 1198 total embeddings from 7 chunk(s)
```

### Model Selection

Choose between built-in models or use a custom HuggingFace model:

```bash
# Use built-in LFM2 model (128-dim, higher accuracy)
PlaidCLI demo \
  --query "your query" \
  --files docs/*.txt \
  --model lfm2

# Use built-in MXBAI model (64-dim, faster)
PlaidCLI demo \
  --query "your query" \
  --files docs/*.txt \
  --model mxbai

# Or use any HuggingFace-compatible ColBERT model
PlaidCLI demo \
  --query "your query" \
  --files docs/*.txt \
  --pretrained "your-org/custom-colbert-model"
```

**Note**: When using `--pretrained` with a custom model, you must also ensure the embedding dimension matches your model's output by setting it appropriately in the code.

### Compression Control

Adjust quantization bits for size/accuracy tradeoff:

```bash
# Higher compression (smaller index, slightly lower accuracy)
PlaidCLI demo ... --nbits 1   # 2 clusters

# Balanced (default)
PlaidCLI demo ... --nbits 2   # 4 clusters

# Higher accuracy (larger index)
PlaidCLI demo ... --nbits 4   # 16 clusters
```

### Model Performance Comparison

Choosing the right model depends on your use case:

#### When to Use LFM2-ColBERT
- ✅ **Production search systems** requiring maximum accuracy
- ✅ **Q&A systems** where answer quality is critical
- ✅ **Research applications** needing state-of-the-art performance
- ✅ **Desktop/server applications** with sufficient memory
- ⚠️ Requires ~512MB memory
- ⚠️ Slower indexing (128-dimensional embeddings)

```bash
PlaidCLI demo --query "..." --files docs/*.txt --model lfm2
```

#### When to Use MXBAI-Edge
- ✅ **Mobile applications** with memory constraints
- ✅ **Large-scale indexing** where speed matters
- ✅ **Edge devices** with limited resources
- ✅ **Prototyping** for faster iteration
- ✅ Lower memory footprint (~256MB)
- ✅ Faster indexing (64-dimensional embeddings)

```bash
PlaidCLI demo --query "..." --files docs/*.txt --model mxbai
```

## Index File Structure

When you create an index, Plaid generates these files:

```
my_index/
├── metadata.json             # Index metadata
├── plaid_index.json          # Index configuration
├── plan.json                 # Execution plan
├── centroids.bin             # Quantization centroids
├── bucket_cutoffs.bin        # Quantization thresholds
├── bucket_weights.bin        # Quantization weights
├── avg_residual.bin          # Average residuals
├── ivf.bin                   # Inverted file index
├── ivf_lengths.bin           # IVF partition sizes
├── chunk_0.codes.bin         # Compressed codes
├── chunk_0.residuals.bin     # Compressed residuals
├── chunk_0.metadata.json     # Chunk metadata
└── doclens.0.json           # Document lengths
```

---

## Performance Tips

1. **Batch Size**: Larger batches are faster but use more memory
   ```swift
   SearchParameters(batchSize: 128, ...)  // Faster, more memory
   SearchParameters(batchSize: 32, ...)   // Slower, less memory
   ```

2. **IVF Probes**: More probes = better recall, slower search
   ```swift
   SearchParameters(nIvfProbe: 16, ...)   // Better recall
   SearchParameters(nIvfProbe: 4, ...)    // Faster search
   ```

3. **Candidate Selection**: Balance speed vs. accuracy
   ```swift
   SearchParameters(nFullScores: 2048, ...)  // More candidates
   SearchParameters(nFullScores: 512, ...)   // Faster
   ```

4. **Preloading**: Load index once for multiple searches
   ```swift
   Plaid.loadAndSearch(..., preloadIndex: true)  // Cache in memory
   ```

---

## Requirements

- **Swift**: 5.9 or later
- **macOS**: 13.3+ (Ventura)
- **iOS**: 17.0+
- **Xcode**: 15.0+ (for development)

### Dependencies

- [MLX Swift](https://github.com/ml-explore/mlx-swift) - Tensor operations
- CoreML - On-device inference
- Foundation - Core Swift functionality

All dependencies are managed by Swift Package Manager.

---

## Building from Source

```bash
# Clone repository
git clone https://github.com/your-org/plaid-swift.git
cd plaid-swift

# Build
swift build

# Run tests
swift test

# Build for release
swift build -c release
```

---

## License

[Your License Here]

---

## Citation

If you use Plaid Swift in your research, please cite:

```bibtex
@software{plaid,
  title = {Plaid: Native Semantic Search for iOS and macOS},
  author = {Sachin Desai},
  year = {2025},
  url = {https://github.com/smdesai/plaid}
}
```

---

**Built with ❤️ using Swift and MLX**
