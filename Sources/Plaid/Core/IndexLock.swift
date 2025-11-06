import Foundation

/// File-based locking mechanism for preventing concurrent modifications to a Plaid index.
///
/// This class implements a simple file-based lock using a `.lock` file in the index directory.
/// The lock file contains the process ID (PID) of the owning process, allowing for stale lock detection.
///
/// Usage:
/// ```swift
/// let lock = try IndexLock.acquire(for: indexURL)
/// defer { lock.release() }
/// // Perform index modifications
/// ```
public final class IndexLock {
    private let lockFileURL: URL
    private let processID: Int32
    private var isLocked: Bool = false

    private init(lockFileURL: URL, processID: Int32) {
        self.lockFileURL = lockFileURL
        self.processID = processID
    }

    /// Acquires a lock for the index at the specified URL.
    ///
    /// - Parameter indexURL: The URL of the index directory to lock
    /// - Throws: `PlaidError.indexLocked` if the index is already locked by another process
    /// - Returns: An `IndexLock` instance that must be released when done
    public static func acquire(for indexURL: URL) throws -> IndexLock {
        let lockFileURL = indexURL.appendingPathComponent(".lock")
        let currentPID = getpid()

        // Check if lock file exists
        if FileManager.default.fileExists(atPath: lockFileURL.path) {
            // Try to read existing lock
            if let lockData = try? Data(contentsOf: lockFileURL),
                let lockPIDString = String(data: lockData, encoding: .utf8),
                let lockPID = Int32(lockPIDString.trimmingCharacters(in: .whitespacesAndNewlines))
            {
                // Check if the process is still running
                if isProcessRunning(pid: lockPID) {
                    throw PlaidError.indexLocked(
                        "Index is locked by process \(lockPID). "
                            + "Wait for the operation to complete or manually remove \(lockFileURL.path) if the process has crashed."
                    )
                } else {
                    // Stale lock - remove it
                    try? FileManager.default.removeItem(at: lockFileURL)
                }
            } else {
                // Corrupted lock file - remove it
                try? FileManager.default.removeItem(at: lockFileURL)
            }
        }

        // Create new lock file
        let pidData = "\(currentPID)\n".data(using: .utf8)!
        try pidData.write(to: lockFileURL, options: [.atomic])

        let lock = IndexLock(lockFileURL: lockFileURL, processID: currentPID)
        lock.isLocked = true
        return lock
    }

    /// Releases the lock.
    ///
    /// This should be called when the index modification is complete.
    /// It's safe to call multiple times.
    public func release() {
        guard isLocked else { return }

        // Only remove lock if it still contains our PID
        if let lockData = try? Data(contentsOf: lockFileURL),
            let lockPIDString = String(data: lockData, encoding: .utf8),
            let lockPID = Int32(lockPIDString.trimmingCharacters(in: .whitespacesAndNewlines)),
            lockPID == processID
        {
            try? FileManager.default.removeItem(at: lockFileURL)
        }

        isLocked = false
    }

    deinit {
        release()
    }

    /// Checks if a process with the given PID is currently running.
    ///
    /// - Parameter pid: The process ID to check
    /// - Returns: `true` if the process is running, `false` otherwise
    private static func isProcessRunning(pid: Int32) -> Bool {
        // On Unix systems, kill(pid, 0) returns 0 if the process exists
        // without actually sending a signal
        return kill(pid, 0) == 0
    }
}

/// Extension to add lock-related errors
extension PlaidError {
    static func indexLocked(_ message: String) -> PlaidError {
        .invalidSubset(message)
    }
}
