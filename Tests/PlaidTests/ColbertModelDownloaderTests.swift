import Foundation
import XCTest

@testable import Plaid

/// The snapshot glob must reach every file inside a compiled Core ML bundle and
/// match none of the directory entries a recursive Hub tree listing returns —
/// `HubClient.downloadSnapshot` filters by glob only, and requesting a directory
/// as a file is the "404 Entry not found" that broke every encoder download.
final class ColbertModelDownloaderTests: XCTestCase {
    private func matches(_ path: String) -> Bool {
        ColbertModelDownloader.snapshotGlobs(modelName: "MXBAIEdgeColbert")
            .contains { fnmatch($0, path, 0) == 0 }
    }

    func testGlobMatchesEveryFileInTheBundle() {
        for file in [
            "MXBAIEdgeColbert.mlmodelc/coremldata.bin",
            "MXBAIEdgeColbert.mlmodelc/metadata.json",
            "MXBAIEdgeColbert.mlmodelc/model.mil",
            "MXBAIEdgeColbert.mlmodelc/weights/weight.bin",
            "MXBAIEdgeColbert.mlmodelc/analytics/coremldata.bin",
        ] {
            XCTAssertTrue(matches(file), file)
        }
    }

    func testGlobSkipsDirectoryEntriesAndRepoRootFiles() {
        for entry in [
            "MXBAIEdgeColbert.mlmodelc",
            "MXBAIEdgeColbert.mlmodelc/weights",
            "MXBAIEdgeColbert.mlmodelc/analytics",
            ".gitattributes",
            "README.md",
            "LFM2Colbert.mlmodelc/coremldata.bin",
        ] {
            XCTAssertFalse(matches(entry), entry)
        }
    }
}
