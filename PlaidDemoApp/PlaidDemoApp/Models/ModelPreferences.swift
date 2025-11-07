import Foundation
import SwiftUI

/// Represents the available ColBERT embedding models
enum ModelType: String, Codable, CaseIterable, Identifiable {
    case lfm2 = "LFM2"
    case mxbaiEdge = "MXBAIEdge"

    var id: String { rawValue }

    /// Human-readable display name
    var displayName: String {
        switch self {
        case .lfm2:
            return "LFM2-ColBERT"
        case .mxbaiEdge:
            return "MXBAI-Edge"
        }
    }

    /// Short description of the model's characteristics
    var description: String {
        switch self {
        case .lfm2:
            return "Powerful & Accurate"
        case .mxbaiEdge:
            return "Fast & Efficient"
        }
    }

    /// Detailed feature list for model selection
    var features: [String] {
        switch self {
        case .lfm2:
            return [
                "Best accuracy",
                "Deep semantic understanding",
                "Research-grade quality",
            ]
        case .mxbaiEdge:
            return [
                "Optimized size",
                "Quick indexing",
                "Low memory usage",
            ]
        }
    }

    /// SF Symbol icon name
    var iconName: String {
        switch self {
        case .lfm2:
            return "cpu.fill"
        case .mxbaiEdge:
            return "bolt.fill"
        }
    }

    /// Primary color for the model
    var color: Color {
        switch self {
        case .lfm2:
            return .blue
        case .mxbaiEdge:
            return .orange
        }
    }

    /// Hugging Face model ID for tokenizer loading
    var modelId: String {
        switch self {
        case .lfm2:
            return "LiquidAI/LFM2-ColBERT-350M"
        case .mxbaiEdge:
            return "mixedbread-ai/mxbai-edge-colbert-v0-32m"
        }
    }

    /// Embedding dimension for the model
    var embeddingDimension: Int {
        switch self {
        case .lfm2:
            return 128
        case .mxbaiEdge:
            return 64
        }
    }
}

/// Manages user preferences for model selection with persistence
@MainActor
class ModelPreferences: ObservableObject {
    @Published var selectedModel: ModelType {
        didSet {
            UserDefaults.standard.set(
                selectedModel.rawValue, forKey: ModelPreferences.selectedModelKey)
        }
    }

    private static let selectedModelKey = "selectedModel"

    init() {
        // Load persisted model preference or default to LFM2
        if let savedRawValue = UserDefaults.standard.string(
            forKey: ModelPreferences.selectedModelKey),
            let savedModel = ModelType(rawValue: savedRawValue)
        {
            self.selectedModel = savedModel
        } else {
            self.selectedModel = .lfm2
        }
    }

    /// Reset to default model
    func reset() {
        selectedModel = .lfm2
    }
}
