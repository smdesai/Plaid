import Foundation
import HuggingFace

/// Errors raised while fetching a Core ML ColBERT encoder from the Hugging Face Hub.
public enum ColbertModelDownloadError: Error, LocalizedError {
    case invalidRepositoryId(String)
    case modelNotFound(repoId: String, modelName: String)

    public var errorDescription: String? {
        switch self {
        case .invalidRepositoryId(let id):
            return "\"\(id)\" is not a valid Hugging Face repository id."
        case .modelNotFound(let repoId, let modelName):
            return
                "The snapshot of \(repoId) does not contain \(modelName).mlmodelc (or it is incomplete)."
        }
    }
}

/// Downloads Plaid's Core ML ColBERT encoders from the Hugging Face Hub.
///
/// Models are fetched with the same `HubClient` and on-disk cache that
/// `ColbertTokenizer.from(pretrained:)` uses, so a second call for the same
/// repository is a local lookup. Each repository is expected to hold a single
/// compiled `<modelName>.mlmodelc` directory at its root.
public enum ColbertModelDownloader {
    public typealias ProgressHandler = @MainActor @Sendable (Progress) -> Void

    /// Ensures `<modelName>.mlmodelc` from `repoId` is available locally and returns its URL.
    ///
    /// - Parameters:
    ///   - repoId: Hugging Face repository id, e.g. `"smdesai/LFM2Colbert"`.
    ///   - modelName: Name of the compiled model directory without the `.mlmodelc` suffix.
    ///   - revision: Git revision to fetch (branch, tag or commit).
    ///   - progressHandler: Download progress, delivered on the main actor.
    public static func download(
        repoId: String,
        modelName: String,
        revision: String = "main",
        progressHandler: ProgressHandler? = nil
    ) async throws -> URL {
        guard let repo = Repo.ID(rawValue: repoId) else {
            throw ColbertModelDownloadError.invalidRepositoryId(repoId)
        }
        // Files only. `HubClient.downloadSnapshot` lists the tree recursively and
        // filters the entries by glob alone — it never checks `entry.type`, so the
        // directory entries of a compiled Core ML bundle (`X.mlmodelc`, `X.mlmodelc/weights`,
        // `X.mlmodelc/analytics`) are requested as files and the Hub answers
        // "404 Entry not found" for the first one. An unfiltered snapshot therefore
        // never succeeds for these repos. `snapshotGlobs(modelName:)` selects the
        // files inside the bundle and nothing else; see its doc for why it is safe.
        let snapshot = try await HubClient.default.downloadSnapshot(
            of: repo,
            revision: revision,
            matching: snapshotGlobs(modelName: modelName),
            progressHandler: progressHandler
        )
        let modelURL = snapshot.appendingPathComponent("\(modelName).mlmodelc", isDirectory: true)
        let marker = modelURL.appendingPathComponent("coremldata.bin")
        guard FileManager.default.fileExists(atPath: marker.path) else {
            throw ColbertModelDownloadError.modelNotFound(repoId: repoId, modelName: modelName)
        }
        return modelURL
    }

    /// The glob that selects every file inside `<modelName>.mlmodelc` and no
    /// directory entry.
    ///
    /// `HubClient` matches with `fnmatch(_, _, 0)`, where `*` also matches `/`, so
    /// `<name>.mlmodelc/*.*` reaches nested files at any depth. It excludes the
    /// tree's directory entries because a directory inside a compiled bundle never
    /// has a dot in its name (`weights`, `analytics`), while every file in one does
    /// (`coremldata.bin`, `model.mil`, `metadata.json`, `weights/weight.bin`). The
    /// bundle directory itself does not match either — it lacks the trailing `/`.
    public static func snapshotGlobs(modelName: String) -> [String] {
        ["\(modelName).mlmodelc/*.*"]
    }
}
