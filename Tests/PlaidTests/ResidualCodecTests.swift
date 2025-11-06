import Foundation
import MLX
import XCTest

@testable import Plaid

/// Tests to verify that the ResidualCodec quantization matches the Rust implementation.
///
/// The critical fix ensures that quantiles are computed GLOBALLY across all residual values,
/// not per-dimension. This test verifies that behavior.
final class ResidualCodecTests: XCTestCase {

    /// Test that bucket weights are computed globally (same value broadcast across dimensions)
    func testGlobalQuantileComputation() throws {
        let nbits = 2
        let embeddingDim = 4
        let numSamples = 100

        // Create synthetic centroids
        var centroidValues: [Float] = []
        for i in 0 ..< 4 {
            for d in 0 ..< embeddingDim {
                centroidValues.append(d == i ? 1.0 : 0.0)
            }
        }
        let centroids = MLXArray(centroidValues, [4, embeddingDim]).asType(
            .float32, stream: .default)

        // Create synthetic samples with known distribution
        var sampleValues: [Float] = []
        for _ in 0 ..< numSamples {
            for d in 0 ..< embeddingDim {
                // Values range from -1 to 1, uniform across all dimensions
                let value = Float.random(in: -1.0 ... 1.0)
                sampleValues.append(value)
            }
        }
        let samples = MLXArray(sampleValues, [numSamples, embeddingDim]).asType(
            .float32, stream: .default)

        // Train codec
        let trainer = ResidualCodecTrainer(nbits: nbits, stream: .default)
        let codec = try trainer.train(centroids: centroids, samples: samples)

        // Verify bucket weights exist
        XCTAssertNotNil(codec.bucketWeights, "Bucket weights should be computed")
        guard let weights = codec.bucketWeights else {
            XCTFail("Bucket weights are nil")
            return
        }

        // Expected shape: [numBuckets, embeddingDim] where numBuckets = 2^nbits
        let numBuckets = 1 << nbits
        XCTAssertEqual(
            weights.shape, [numBuckets, embeddingDim],
            "Bucket weights should have shape [numBuckets, embeddingDim]")

        // The CRITICAL test: All dimensions should have the SAME bucket weights
        // because quantiles are computed globally, not per-dimension
        let weightArray = weights.asArray(Float32.self)

        for bucket in 0 ..< numBuckets {
            let firstDimValue = weightArray[bucket * embeddingDim + 0]

            // Check that all dimensions in this bucket have the same value
            for dim in 1 ..< embeddingDim {
                let dimValue = weightArray[bucket * embeddingDim + dim]
                XCTAssertEqual(
                    firstDimValue,
                    dimValue,
                    accuracy: 1e-6,
                    "Bucket \(bucket): All dimensions should have the same weight value (global quantiles). Found \(firstDimValue) vs \(dimValue)"
                )
            }
        }

        // Verify bucket cutoffs also use global quantiles
        XCTAssertNotNil(codec.bucketCutoffs, "Bucket cutoffs should be computed")
        guard let cutoffs = codec.bucketCutoffs else {
            XCTFail("Bucket cutoffs are nil")
            return
        }

        XCTAssertEqual(
            cutoffs.shape, [numBuckets - 1, embeddingDim],
            "Bucket cutoffs should have shape [numBuckets-1, embeddingDim]")

        let cutoffArray = cutoffs.asArray(Float32.self)
        for bucket in 0 ..< (numBuckets - 1) {
            let firstDimValue = cutoffArray[bucket * embeddingDim + 0]

            for dim in 1 ..< embeddingDim {
                let dimValue = cutoffArray[bucket * embeddingDim + dim]
                XCTAssertEqual(
                    firstDimValue,
                    dimValue,
                    accuracy: 1e-6,
                    "Cutoff \(bucket): All dimensions should have the same cutoff value (global quantiles). Found \(firstDimValue) vs \(dimValue)"
                )
            }
        }
    }

    /// Test that quantization buckets are correctly ordered
    func testBucketOrdering() throws {
        let nbits = 2
        let embeddingDim = 4
        let numSamples = 100

        var centroidValues: [Float] = []
        for i in 0 ..< 4 {
            for d in 0 ..< embeddingDim {
                centroidValues.append(d == i ? 1.0 : 0.0)
            }
        }
        let centroids = MLXArray(centroidValues, [4, embeddingDim]).asType(
            .float32, stream: .default)

        var sampleValues: [Float] = []
        for _ in 0 ..< numSamples {
            for d in 0 ..< embeddingDim {
                let value = Float.random(in: -1.0 ... 1.0)
                sampleValues.append(value)
            }
        }
        let samples = MLXArray(sampleValues, [numSamples, embeddingDim]).asType(
            .float32, stream: .default)

        let trainer = ResidualCodecTrainer(nbits: nbits, stream: .default)
        let codec = try trainer.train(centroids: centroids, samples: samples)

        guard let weights = codec.bucketWeights else {
            XCTFail("Bucket weights are nil")
            return
        }

        let numBuckets = 1 << nbits
        let weightArray = weights.asArray(Float32.self)

        // Extract the first dimension's weights (they should all be the same across dims)
        var bucketWeights: [Float] = []
        for bucket in 0 ..< numBuckets {
            bucketWeights.append(Float(weightArray[bucket * embeddingDim]))
        }

        // Bucket weights should be in ascending order (quantiles are ordered)
        for i in 0 ..< (bucketWeights.count - 1) {
            XCTAssertLessThanOrEqual(
                bucketWeights[i],
                bucketWeights[i + 1],
                "Bucket weights should be in ascending order (bucket \(i) = \(bucketWeights[i]), bucket \(i+1) = \(bucketWeights[i+1]))"
            )
        }
    }

    /// Test end-to-end index creation with the fixed quantization
    func testIndexCreationWithGlobalQuantization() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let indexURL = tempDir.appendingPathComponent("test_index")

        let embeddingDim = 8
        let nbits = 2

        // Create test documents
        let documents: [[[Float]]] = [
            [
                [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.8, 0.2, 0.0, 0.0, 0.0, 0.0],
            ],
        ]

        // Create centroids
        var centroidData: [[Float]] = []
        for i in 0 ..< embeddingDim {
            var centroid = [Float](repeating: 0.0, count: embeddingDim)
            centroid[i] = 1.0
            centroidData.append(centroid)
        }

        // Create index - this will use the fixed quantization
        try Plaid.create(
            indexURL: indexURL,
            device: "cpu",
            embeddingDim: embeddingDim,
            nbits: nbits,
            embeddings: documents,
            centroids: centroidData,
            batchSize: 2,
            seed: 42
        )

        // Verify index was created
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: indexURL.path), "Index should be created")

        // Load and verify artifacts
        let artifacts = try IndexStorage.loadArtifacts(from: indexURL)

        // Verify bucket weights use global quantization
        guard let bucketWeights = artifacts.bucketWeights else {
            XCTFail("Bucket weights should exist")
            return
        }

        let weightArray = bucketWeights.asArray(Float32.self)
        let numBuckets = 1 << nbits

        // Verify all dimensions have the same values (global quantization)
        for bucket in 0 ..< numBuckets {
            let firstValue = weightArray[bucket * embeddingDim]
            for dim in 1 ..< embeddingDim {
                XCTAssertEqual(
                    firstValue,
                    weightArray[bucket * embeddingDim + dim],
                    accuracy: 1e-6,
                    "All dimensions should have the same bucket weight (global quantization)"
                )
            }
        }

        // Verify search works
        let queries: [[[Float]]] = [
            [[0.85, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        ]

        let params = SearchParameters(
            batchSize: 1,
            nFullScores: 10,
            topK: 3,
            nIvfProbe: 2
        )

        let results = try Plaid.loadAndSearch(
            indexURL: indexURL,
            device: "cpu",
            queries: queries,
            searchParameters: params,
            showProgress: false,
            preloadIndex: false
        )

        XCTAssertEqual(results.count, 1, "Should have one result")
        XCTAssertFalse(results[0].passageIds.isEmpty, "Should have passage IDs")
        XCTAssertFalse(results[0].scores.isEmpty, "Should have scores")
    }
}
