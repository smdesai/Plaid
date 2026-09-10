// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "Plaid",
    platforms: [
        .iOS(.v17),
        .macOS(.v14),
    ],
    products: [
        .library(
            name: "Plaid",
            targets: ["Plaid"]
        ),
        .executable(
            name: "PlaidCLI",
            targets: ["PlaidCLI"]
        ),
    ],
    dependencies: [
        // Consumers that need the local experimental mlx-swift override this with a
        // root-level path dependency; a tagged Plaid must not depend on a path itself
        // (SwiftPM: stable-version packages cannot depend on unstable-version packages).
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.3"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.0.0"),
        .package(url: "https://github.com/huggingface/swift-huggingface.git", from: "0.8.1"),
    ],
    targets: [
        // Prebuilt Rust `next-plaid` engine (UniFFI), built by
        // next-plaid/next-plaid-ffi/xcframework-build.sh. Provides the clang
        // module `next_plaid_ffiFFI` consumed by NextPlaidBindings.
        .binaryTarget(
            name: "NextPlaidFFI",
            path: "Frameworks/NextPlaidFFI.xcframework"
        ),
        // UniFFI-generated Swift surface over NextPlaidFFI (PlaidIndex, records,
        // FfiError). Regenerated alongside the XCFramework; do not hand-edit.
        .target(
            name: "NextPlaidBindings",
            dependencies: ["NextPlaidFFI"],
            path: "Sources/NextPlaidBindings",
            // The Rust staticlib was built with the `accelerate` feature
            // (cargo emits a link directive that a .a cannot carry to SPM), so
            // the final product must link Accelerate for the engine's BLAS.
            linkerSettings: [.linkedFramework("Accelerate")]
        ),
        .target(
            name: "Plaid",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                "NextPlaidBindings",
            ],
            path: "Sources/Plaid",
            // Core ML encoders are downloaded from the Hugging Face Hub at runtime
            // (see ColbertModelDownloader); any locally converted models parked under
            // Sources/Plaid/Model are gitignored and must not be treated as sources.
            exclude: ["Model"]
        ),
        .testTarget(
            name: "PlaidTests",
            dependencies: ["Plaid"],
            path: "Tests/PlaidTests"
        ),
        .executableTarget(
            name: "PlaidCLI",
            dependencies: ["Plaid"],
            path: "Sources/PlaidCLI"
        ),
    ]
)
