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
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.29.1"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "0.1.17"),
    ],
    targets: [
        .target(
            name: "Plaid",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "Sources/Plaid",
            resources: [
                .copy("Model/LFM2Colbert.mlpackage")
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
