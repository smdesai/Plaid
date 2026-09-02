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
        .target(
            name: "Plaid",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
            ],
            path: "Sources/Plaid",
            resources: [
                .copy("Model/LFM2Colbert.mlmodelc"),
                .copy("Model/LFM2ColbertQuant4.mlmodelc"),
                .copy("Model/MXBAIEdgeColbert.mlmodelc"),
            ]
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
