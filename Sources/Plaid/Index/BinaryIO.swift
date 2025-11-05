import Foundation

enum BinaryIO {
    static func writeFloat32(_ values: [Float], to url: URL) throws {
        try ensureParentDirectoryExists(for: url)
        let data = values.withUnsafeBufferPointer { buffer -> Data in
            var data = Data(capacity: buffer.count * MemoryLayout<UInt32>.size)
            for value in buffer {
                var bits = value.bitPattern.littleEndian
                withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
            }
            return data
        }
        try data.write(to: url, options: .atomic)
    }

    static func writeInt32(_ values: [Int32], to url: URL) throws {
        try ensureParentDirectoryExists(for: url)
        let data = values.withUnsafeBufferPointer { buffer -> Data in
            var data = Data(capacity: buffer.count * MemoryLayout<Int32>.size)
            for value in buffer {
                var le = value.littleEndian
                withUnsafeBytes(of: &le) { data.append(contentsOf: $0) }
            }
            return data
        }
        try data.write(to: url, options: .atomic)
    }

    static func writeUInt8(_ values: [UInt8], to url: URL) throws {
        try ensureParentDirectoryExists(for: url)
        let data = Data(values)
        try data.write(to: url, options: .atomic)
    }

    static func readInt32(from url: URL) throws -> [Int32] {
        let data = try Data(contentsOf: url)
        let count = data.count / MemoryLayout<Int32>.size
        return data.withUnsafeBytes { raw in
            Array(
                UnsafeBufferPointer(
                    start: raw.baseAddress!.assumingMemoryBound(to: Int32.self),
                    count: count
                ))
        }
    }

    private static func ensureParentDirectoryExists(for url: URL) throws {
        let directory = url.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }
}
