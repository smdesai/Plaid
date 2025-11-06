#if canImport(MLX)
    import Foundation
    import MLX

    @available(iOS 17, macOS 13.3, *)
    extension Plaid {
        public static func create(
            indexURL: URL,
            torchPath: String? = nil,
            device: String,
            embeddingDim: Int,
            nbits: Int,
            embeddings: [MLXArray],
            centroids: MLXArray? = nil,
            batchSize: Int,
            seed: UInt64? = nil
        ) throws {
            let matrices = try embeddings.map { try arrayToMatrix($0) }
            let centroidMatrix = try centroids.map { try arrayToMatrix($0) } ?? []
            try create(
                indexURL: indexURL,
                torchPath: torchPath,
                device: device,
                embeddingDim: embeddingDim,
                nbits: nbits,
                embeddings: matrices,
                centroids: centroidMatrix,
                batchSize: batchSize,
                seed: seed
            )
        }

        public static func create(
            indexPath: String,
            torchPath: String? = nil,
            device: String,
            embeddingDim: Int,
            nbits: Int,
            embeddings: [MLXArray],
            centroids: MLXArray? = nil,
            batchSize: Int,
            seed: UInt64? = nil
        ) throws {
            try create(
                indexURL: URL(fileURLWithPath: indexPath),
                torchPath: torchPath,
                device: device,
                embeddingDim: embeddingDim,
                nbits: nbits,
                embeddings: embeddings,
                centroids: centroids,
                batchSize: batchSize,
                seed: seed
            )
        }

        public static func update(
            indexURL: URL,
            torchPath: String? = nil,
            device: String,
            embeddings: [MLXArray],
            batchSize: Int
        ) throws {
            let matrices = try embeddings.map { try arrayToMatrix($0) }
            try update(
                indexURL: indexURL,
                torchPath: torchPath,
                device: device,
                embeddings: matrices,
                batchSize: batchSize
            )
        }

        public static func update(
            indexPath: String,
            torchPath: String? = nil,
            device: String,
            embeddings: [MLXArray],
            batchSize: Int
        ) throws {
            try update(
                indexURL: URL(fileURLWithPath: indexPath),
                torchPath: torchPath,
                device: device,
                embeddings: embeddings,
                batchSize: batchSize
            )
        }

        public static func loadAndSearch(
            indexURL: URL,
            torchPath: String? = nil,
            device: String,
            queries: MLXArray,
            searchParameters: SearchParameters,
            showProgress: Bool,
            preloadIndex: Bool,
            subset: [[Int]]? = nil
        ) throws -> [QueryResult] {
            let queryArrays = try arrayToQueryBatches(queries)
            return try loadAndSearch(
                indexURL: indexURL,
                torchPath: torchPath,
                device: device,
                queries: queryArrays,
                searchParameters: searchParameters,
                showProgress: showProgress,
                preloadIndex: preloadIndex,
                subset: subset
            )
        }

        public static func loadAndSearch(
            indexPath: String,
            torchPath: String? = nil,
            device: String,
            queries: MLXArray,
            searchParameters: SearchParameters,
            showProgress: Bool,
            preloadIndex: Bool,
            subset: [[Int]]? = nil
        ) throws -> [QueryResult] {
            try loadAndSearch(
                indexURL: URL(fileURLWithPath: indexPath),
                torchPath: torchPath,
                device: device,
                queries: queries,
                searchParameters: searchParameters,
                showProgress: showProgress,
                preloadIndex: preloadIndex,
                subset: subset
            )
        }
    }

    private func arrayToMatrix(_ array: MLXArray) throws -> [[Float]] {
        let shape = array.shape
        guard shape.count == 2 else {
            throw PlaidError.invalidSubset("Expected rank 2 array, got rank \(shape.count)")
        }
        let rows = Int(shape[0])
        let cols = Int(shape[1])
        let values = array.asArray(Float.self)
        guard values.count == rows * cols else {
            throw PlaidError.invalidSubset("Array buffer size mismatch")
        }

        var result: [[Float]] = []
        result.reserveCapacity(rows)
        for row in 0 ..< rows {
            let start = row * cols
            let end = start + cols
            result.append(Array(values[start ..< end]))
        }
        return result
    }

    private func arrayToQueryBatches(_ array: MLXArray) throws -> [[[Float]]] {
        let shape = array.shape
        guard shape.count == 3 else {
            throw PlaidError.invalidSubset(
                "Expected rank 3 array for queries, got rank \(shape.count)")
        }
        let batches = Int(shape[0])
        let tokens = Int(shape[1])
        let dim = Int(shape[2])
        let values = array.asArray(Float.self)
        guard values.count == batches * tokens * dim else {
            throw PlaidError.invalidSubset("Array buffer size mismatch")
        }

        var result: [[[Float]]] = []
        result.reserveCapacity(batches)

        for batch in 0 ..< batches {
            var batchResult: [[Float]] = []
            batchResult.reserveCapacity(tokens)

            for token in 0 ..< tokens {
                let start = (batch * tokens + token) * dim
                let end = start + dim
                batchResult.append(Array(values[start ..< end]))
            }
            result.append(batchResult)
        }
        return result
    }
#endif
