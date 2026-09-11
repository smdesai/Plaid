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
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.0.0"),
        .package(url: "https://github.com/huggingface/swift-huggingface.git", from: "0.8.1"),
    ],
    targets: [
        // Prebuilt Rust `next-plaid` engine (UniFFI), built by
        // next-plaid/next-plaid-ffi/xcframework-build.sh. Provides the clang
        // module `next_plaid_ffiFFI` consumed by NextPlaidBindings.
        .binaryTarget(
            name: "NextPlaidFFI",
            url:
                "https://github.com/smdesai/Plaid/releases/download/v2.1.0/NextPlaidFFI.xcframework.zip",
            checksum: "a17d572a6ef39376a80a2b3c99f2e1d6101eaa3d2fef155aaf393686cc1a1ead"
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
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                "NextPlaidBindings",
            ],
            path: "Sources/Plaid",
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
