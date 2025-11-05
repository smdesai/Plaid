import Foundation
import MLX

struct CodecArtifacts {
    let bitHelper: MLXArray
    let byteReversedBitsMap: MLXArray
    let bucketWeightLookup: MLXArray?
}

enum CodecSerialization {
    static func save(
        codec: ResidualCodec,
        paths: IndexBuilder.IndexPaths,
        stream: StreamOrDevice = .default
    ) throws {
        try BinaryIO.writeInt32(
            codec.bitHelper.asArray(Int32.self).map { Int32($0) },
            to: paths.bitHelper()
        )

        try BinaryIO.writeInt32(
            codec.byteReversedBitsMap.asArray(Int32.self).map { Int32($0) },
            to: paths.byteReversedBitsMap()
        )

        if let lookup = codec.bucketWeightIndicesLookup {
            try BinaryIO.writeInt32(
                lookup.asArray(Int32.self).map { Int32($0) },
                to: paths.bucketWeightLookup()
            )
        }
    }

    static func load(
        from paths: IndexStorage.IndexPaths,
        stream: StreamOrDevice = .default
    ) throws -> CodecArtifacts {
        let bitHelperData =
            FileManager.default.fileExists(atPath: paths.bitHelper().path)
            ? try BinaryIO.readInt32(from: paths.bitHelper())
            : []
        let bitHelper = MLXArray(bitHelperData.map { Float32($0) }, [bitHelperData.count]).asType(
            .int32, stream: stream)

        let byteMapData =
            FileManager.default.fileExists(atPath: paths.byteReversedBitsMap().path)
            ? try BinaryIO.readInt32(from: paths.byteReversedBitsMap())
            : []
        let byteMap = MLXArray(byteMapData.map { Float32($0) }, [byteMapData.count]).asType(
            .int32, stream: stream)

        let lookup: MLXArray? = {
            guard FileManager.default.fileExists(atPath: paths.bucketWeightLookup().path) else {
                return nil
            }
            guard let data = try? BinaryIO.readInt32(from: paths.bucketWeightLookup()),
                !data.isEmpty
            else {
                return nil
            }
            return MLXArray(data.map { Float32($0) }, [data.count]).asType(.int32, stream: stream)
        }()

        return CodecArtifacts(
            bitHelper: bitHelper,
            byteReversedBitsMap: byteMap,
            bucketWeightLookup: lookup
        )
    }
}
