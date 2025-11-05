import Foundation
import XCTest

@testable import Plaid

private struct RustSearchOutput: Decodable {
    struct Entry: Decodable {
        let query_id: Int
        let passage_ids: [Int]
        let scores: [Double]
    }

    let results: [Entry]
}

private enum RustReferenceError: Error {
    case unavailable(String)
    case execution(String)
}

private struct RustQueries: Codable {
    let queries: [[[Float]]]
}

private struct RustParams: Codable {
    let batchSize: Int
    let nFullScores: Int
    let topK: Int
    let nIvfProbe: Int
}

final class PlaidTests: XCTestCase {
    private static let scoreTolerance: Float = 1e-2

    private static let fixturesRoot: URL = {
        packageRoot.appendingPathComponent("fixtures", isDirectory: true)
    }()

    private static let packageRoot: URL = {
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0 ..< 5 {
            url.deleteLastPathComponent()
        }
        return url
    }()

    private static let metallibSource: URL = {
        packageRoot.appendingPathComponent("swift/Plaid/default.metallib")
    }()

    private static let metallibDestination: URL = {
        packageRoot.appendingPathComponent("default.metallib")
    }()

    override class func setUp() {
        super.setUp()
        copyMetallibIfNeeded()
    }

    func testFixtureParity() throws {
        let indexURL = Self.fixturesRoot.appendingPathComponent("python_index", isDirectory: true)
        let docsURL = Self.fixturesRoot.appendingPathComponent("documents.json")
        let queriesURL = Self.fixturesRoot.appendingPathComponent("queries.json")
        let resultsURL = Self.fixturesRoot.appendingPathComponent("python_results.json")

        for url in [indexURL, docsURL, queriesURL, resultsURL] {
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw XCTSkip("Fixture missing: \(url.path)")
            }
        }

        let documentsData = try Data(contentsOf: docsURL)
        let documents = try JSONDecoder().decode([[[Float]]].self, from: documentsData)
        let queriesData = try Data(contentsOf: queriesURL)
        let queries = try JSONDecoder().decode([[[Float]]].self, from: queriesData)
        let expectedData = try Data(contentsOf: resultsURL)
        let expectedEntries = try JSONDecoder().decode(
            [RustSearchOutput.Entry].self, from: expectedData)

        let params = SearchParameters(
            batchSize: max(1, queries.count),
            nFullScores: 1024,
            topK: expectedEntries.first?.scores.count ?? 10,
            nIvfProbe: 8
        )

        let swiftResults = try Plaid.loadAndSearch(
            indexURL: indexURL,
            device: "cpu",
            queries: queries,
            searchParameters: params,
            showProgress: false,
            preloadIndex: false
        )

        XCTAssertEqual(swiftResults.count, expectedEntries.count)
        XCTAssertEqual(queries.count, expectedEntries.count)

        for (swift, reference) in zip(swiftResults, expectedEntries) {
            XCTAssertEqual(swift.queryId, reference.query_id)

            let swiftPairs: [(passageId: Int, score: Float)] = zip(swift.passageIds, swift.scores)
                .map { (passageId: $0.0, score: $0.1) }

            let referencePassages: [Int] = reference.passage_ids.map { Int($0) }
            let referenceScores: [Float] = reference.scores.map { Float($0) }
            let referencePairs: [(passageId: Int, score: Float)] = zip(
                referencePassages, referenceScores
            )
            .map { (passageId: $0.0, score: $0.1) }

            let swiftSorted = swiftPairs.sorted {
                (lhs: (passageId: Int, score: Float), rhs: (passageId: Int, score: Float)) -> Bool
                in
                let diff = lhs.score - rhs.score
                if abs(diff) > Self.scoreTolerance { return diff > 0 }
                return lhs.passageId < rhs.passageId
            }

            let referenceSorted = referencePairs.sorted {
                (lhs: (passageId: Int, score: Float), rhs: (passageId: Int, score: Float)) -> Bool
                in
                let diff = lhs.score - rhs.score
                if abs(diff) > Self.scoreTolerance { return diff > 0 }
                return lhs.passageId < rhs.passageId
            }

            XCTAssertEqual(swiftSorted.count, referenceSorted.count)
            XCTAssertEqual(swiftSorted.map { $0.passageId }, referenceSorted.map { $0.passageId })

            for (lhs, rhs) in zip(swiftSorted, referenceSorted) {
                XCTAssertLessThan(abs(lhs.score - rhs.score), Self.scoreTolerance)
            }
        }
        XCTAssertEqual(queries.count, expectedEntries.count)
    }

    func testSwiftMatchesRustReferenceAcrossMutations() throws {
        guard Self.hasCargo else {
            throw XCTSkip("cargo is not available on PATH; skipping parity test")
        }

        let tempDir = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let indexURL = tempDir.appendingPathComponent("index")
        let queries = Self.sampleQueries()
        let params = SearchParameters(batchSize: 2, nFullScores: 4, topK: 3, nIvfProbe: 2)

        try Self.encode(
            RustQueries(queries: queries), to: tempDir.appendingPathComponent("queries.json"))
        try Self.encode(
            RustParams(
                batchSize: params.batchSize,
                nFullScores: params.nFullScores,
                topK: params.topK,
                nIvfProbe: params.nIvfProbe
            ),
            to: tempDir.appendingPathComponent("params.json")
        )

        try Plaid.create(
            indexURL: indexURL,
            device: "cpu",
            embeddingDim: 4,
            nbits: 2,
            embeddings: Self.initialDocuments(),
            centroids: Self.sampleCentroids(),
            batchSize: 2,
            seed: 42
        )

        try assertParity(
            indexURL: indexURL,
            tempDir: tempDir,
            queries: queries,
            params: params,
            message: "initial index parity"
        )

        try Plaid.update(
            indexURL: indexURL,
            device: "cpu",
            embeddings: Self.updateDocuments(),
            batchSize: 1
        )

        try assertParity(
            indexURL: indexURL,
            tempDir: tempDir,
            queries: queries,
            params: params,
            message: "post-update parity"
        )

        try Plaid.delete(
            indexURL: indexURL,
            device: "cpu",
            subset: [1]
        )

        try assertParity(
            indexURL: indexURL,
            tempDir: tempDir,
            queries: queries,
            params: params,
            message: "post-delete parity"
        )
    }
}

extension PlaidTests {
    fileprivate static var hasCargo: Bool {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = ["which", "cargo"]
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = Pipe()
        try? process.run()
        process.waitUntilExit()
        return process.terminationStatus == 0
    }

    fileprivate static func copyMetallibIfNeeded() {
        let fm = FileManager.default
        guard fm.fileExists(atPath: metallibSource.path) else { return }
        if !fm.fileExists(atPath: metallibDestination.path) {
            try? fm.copyItem(at: metallibSource, to: metallibDestination)
        }
    }

    fileprivate func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    fileprivate static func encode<T: Encodable>(_ value: T, to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(value)
        try data.write(to: url, options: .atomic)
    }

    fileprivate func assertParity(
        indexURL: URL,
        tempDir: URL,
        queries: [[[Float]]],
        params: SearchParameters,
        message: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
        let swiftResults = try Plaid.loadAndSearch(
            indexURL: indexURL,
            device: "cpu",
            queries: queries,
            searchParameters: params,
            showProgress: false,
            preloadIndex: false
        )

        let rustOutputData: Data
        do {
            rustOutputData = try runRustReference(
                indexPath: indexURL.path,
                queriesPath: tempDir.appendingPathComponent("queries.json").path,
                paramsPath: tempDir.appendingPathComponent("params.json").path
            )
        } catch RustReferenceError.unavailable(let reason) {
            throw XCTSkip("Rust reference unavailable: \(reason)")
        } catch RustReferenceError.execution(let message) {
            throw XCTSkip("Rust reference failed: \(message)")
        }

        let rustOutput = try JSONDecoder().decode(RustSearchOutput.self, from: rustOutputData)
        XCTAssertEqual(
            swiftResults.count, rustOutput.results.count, message, file: file, line: line)

        for (swift, rust) in zip(swiftResults, rustOutput.results) {
            XCTAssertEqual(swift.queryId, rust.query_id, message, file: file, line: line)
            XCTAssertEqual(
                swift.passageIds, rust.passage_ids.map { Int($0) }, message, file: file, line: line)
            XCTAssertEqual(swift.scores.count, rust.scores.count, message, file: file, line: line)
            for (lhs, rhs) in zip(swift.scores, rust.scores) {
                XCTAssertLessThan(
                    abs(lhs - Float(rhs)),
                    Self.scoreTolerance,
                    "Score mismatch for query \(swift.queryId)",
                    file: file,
                    line: line
                )
            }
        }
    }

    fileprivate func runRustReference(indexPath: String, queriesPath: String, paramsPath: String)
        throws -> Data
    {
        let process = Process()
        process.currentDirectoryURL = PlaidTests.packageRoot
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        let referenceManifest = PlaidTests.packageRoot
            .appendingPathComponent("rust/swift_reference/Cargo.toml")
            .path
        process.arguments = [
            "cargo",
            "run",
            "--quiet",
            "--manifest-path",
            referenceManifest,
            "--",
            indexPath,
            queriesPath,
            paramsPath,
        ]

        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe
        let inheritedEnvironment = ProcessInfo.processInfo.environment
        var environment = inheritedEnvironment
        environment["CARGO_NET_OFFLINE"] = "true"

        if environment["LIBTORCH"] == nil {
            if let override = environment["LAID_LIBTORCH_PATH"] ?? environment["PYTORCH_HOME"] {
                environment["LIBTORCH"] = override
            }
        }
        if environment["LIBTORCH"] == nil,
            let dyldPaths = inheritedEnvironment["DYLD_LIBRARY_PATH"]?.split(separator: ":"),
            let torchLibPath = dyldPaths.first(where: { $0.contains("torch") })
        {
            let torchRoot = String(torchLibPath)
            environment["LIBTORCH"] = NSString(string: torchRoot).deletingLastPathComponent
        }

        if let libtorchRoot = environment["LIBTORCH"] {
            if environment["LIBTORCH_USE_PYTORCH"] == nil {
                environment["LIBTORCH_USE_PYTORCH"] = "1"
            }

            let libDirectory = PlaidTests.resolveTorchLibDirectory(root: libtorchRoot)
            environment["DYLD_LIBRARY_PATH"] = PlaidTests.prependSearchPath(
                libDirectory, existing: environment["DYLD_LIBRARY_PATH"])
            environment["DYLD_FALLBACK_LIBRARY_PATH"] = PlaidTests.prependSearchPath(
                libDirectory, existing: environment["DYLD_FALLBACK_LIBRARY_PATH"])
        }

        process.environment = environment

        try process.run()
        process.waitUntilExit()

        if process.terminationStatus != 0 {
            let stderrData = stderrPipe.fileHandleForReading.readDataToEndOfFile()
            let message = String(data: stderrData, encoding: .utf8) ?? "unknown error"
            let trimmed = message.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.contains("No module named 'torch'")
                || trimmed.contains("Cannot find a libtorch install")
                || trimmed.contains("Library not loaded: @rpath/libtorch_cpu.dylib")
            {
                throw RustReferenceError.unavailable(trimmed)
            }
            throw RustReferenceError.execution(trimmed)
        }

        return stdoutPipe.fileHandleForReading.readDataToEndOfFile()
    }

    fileprivate static func sampleCentroids() -> [[Float]] {
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    }

    fileprivate static func initialDocuments() -> [[[Float]]] {
        [
            [
                [0.95, 0.05, 0, 0],
                [0.8, 0.2, 0, 0],
            ],
            [
                [0.1, 0.9, 0, 0],
                [0.05, 0.85, 0.1, 0],
            ],
            [
                [0.05, 0.05, 0.9, 0],
                [0.05, 0.05, 0.8, 0.1],
            ],
        ]
    }

    fileprivate static func updateDocuments() -> [[[Float]]] {
        [
            [
                [0.2, 0.75, 0.05, 0],
                [0.1, 0.7, 0.2, 0],
            ],
            [
                [0.05, 0.05, 0.05, 0.85],
                [0.05, 0.05, 0.1, 0.8],
            ],
        ]
    }

    fileprivate static func sampleQueries() -> [[[Float]]] {
        [
            [
                [0.9, 0.1, 0, 0],
                [0.75, 0.25, 0, 0],
            ],
            [
                [0.05, 0.05, 0.85, 0.05],
                [0.05, 0.05, 0.8, 0.1],
            ],
        ]
    }

    fileprivate static func resolveTorchLibDirectory(root: String) -> String {
        let fm = FileManager.default
        var isDir: ObjCBool = false
        var adjustedRoot = root
        if fm.fileExists(atPath: adjustedRoot, isDirectory: &isDir) && !isDir.boolValue {
            adjustedRoot = NSString(string: adjustedRoot).deletingLastPathComponent
        }
        let libCandidate = NSString(string: adjustedRoot).appendingPathComponent("lib")
        if fm.fileExists(atPath: libCandidate, isDirectory: &isDir), isDir.boolValue {
            return libCandidate
        }
        return adjustedRoot
    }

    fileprivate static func prependSearchPath(_ newPath: String, existing: String?) -> String {
        guard let existing, !existing.isEmpty else { return newPath }
        return "\(newPath):\(existing)"
    }
}
