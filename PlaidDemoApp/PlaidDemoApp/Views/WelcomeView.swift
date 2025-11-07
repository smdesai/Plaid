import SwiftUI

struct WelcomeView: View {
    @ObservedObject var searchEngine: SearchEngine
    @StateObject private var modelPreferences = ModelPreferences()
    @State private var isLoading = false

    var body: some View {
        ScrollView {
            VStack(spacing: 32) {
                // Hero Section
                VStack(spacing: 16) {
                    Image(systemName: "magnifyingglass.circle.fill")
                        .font(.system(size: 70))
                        .foregroundColor(.blue)
                        .shadow(color: .blue.opacity(0.3), radius: 10, x: 0, y: 5)

                    Text("Plaid Search")
                        .font(.system(size: 36, weight: .bold))

                    Text("Intelligent document retrieval powered by ColBERT")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal)
                }
                .padding(.top, 40)

                // Indexing Progress or Model Selection
                if searchEngine.isIndexing {
                    indexingProgressView
                        .transition(.opacity.combined(with: .scale))
                } else {
                    VStack(spacing: 24) {
                        // Section Header
                        VStack(spacing: 8) {
                            Text("Choose Your Model")
                                .font(.title2)
                                .fontWeight(.semibold)

                            Text("Select the embedding model that best fits your needs")
                                .font(.caption)
                                .foregroundColor(.secondary)
                                .multilineTextAlignment(.center)
                        }

                        // Model Selection Cards
                        VStack(spacing: 16) {
                            ForEach(ModelType.allCases) { model in
                                ModelCard(
                                    model: model,
                                    isSelected: modelPreferences.selectedModel == model
                                ) {
                                    withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
                                        modelPreferences.selectedModel = model
                                    }
                                }
                            }
                        }
                        .padding(.horizontal)

                        // Create Index Button
                        Button(action: startIndexing) {
                            HStack(spacing: 8) {
                                Image(systemName: "sparkles")
                                Text("Create Index")
                                Text("(\(documentCount) docs)")
                                    .font(.subheadline)
                                    .opacity(0.8)
                            }
                            .font(.headline)
                            .foregroundColor(.white)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 16)
                            .background(
                                LinearGradient(
                                    colors: [.blue, .blue.opacity(0.8)],
                                    startPoint: .leading,
                                    endPoint: .trailing
                                )
                            )
                            .cornerRadius(12)
                            .shadow(color: .blue.opacity(0.3), radius: 8, x: 0, y: 4)
                        }
                        .disabled(isLoading)
                        .padding(.horizontal)
                        .padding(.top, 8)
                    }
                    .transition(.opacity.combined(with: .move(edge: .top)))
                }

                // Error message
                if let errorMessage = searchEngine.errorMessage {
                    Text(errorMessage)
                        .font(.caption)
                        .foregroundColor(.red)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal)
                        .padding(.vertical, 8)
                        .background(Color.red.opacity(0.1))
                        .cornerRadius(8)
                        .padding(.horizontal)
                }

                Spacer(minLength: 40)
            }
        }
    }

    private var indexingProgressView: some View {
        VStack(spacing: 24) {
            // Progress Circle
            ZStack {
                Circle()
                    .stroke(Color.blue.opacity(0.2), lineWidth: 8)
                    .frame(width: 120, height: 120)

                Circle()
                    .trim(from: 0, to: searchEngine.indexingProgress)
                    .stroke(
                        LinearGradient(
                            colors: [.blue, .cyan],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        ),
                        style: StrokeStyle(lineWidth: 8, lineCap: .round)
                    )
                    .frame(width: 120, height: 120)
                    .rotationEffect(.degrees(-90))

                VStack(spacing: 4) {
                    Text("\(Int(searchEngine.indexingProgress * 100))%")
                        .font(.title2)
                        .fontWeight(.bold)
                        .foregroundColor(.blue)
                        .contentTransition(.numericText())

                    Image(systemName: "doc.text")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }

            VStack(spacing: 8) {
                Text("Building Index")
                    .font(.title3)
                    .fontWeight(.semibold)

                Text(searchEngine.currentDocument)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .frame(maxWidth: 280)
                    .animation(.none, value: searchEngine.currentDocument)
            }
        }
        .padding(.vertical, 40)
    }

    private var documentCount: Int {
        // Count of bundled documents
        12
    }

    private func startIndexing() {
        isLoading = true

        Task {
            do {
                // Initialize with selected model
                try await searchEngine.initialize(with: modelPreferences.selectedModel)

                // Load bundled sample documents
                let documents = try loadBundledDocuments()
                print("📚 Loaded \(documents.count) bundled documents")

                // Create index
                try await searchEngine.createIndex(documents: documents)

                isLoading = false
            } catch {
                print("❌ Error during indexing: \(error)")
                await MainActor.run {
                    searchEngine.errorMessage = error.localizedDescription
                    isLoading = false
                }
            }
        }
    }

    private func loadBundledDocuments() throws -> [Document] {
        let filenames = [
            "swift_programming.txt",
            "health_tips.txt",
            "travel_guide.txt",
            "swift_programming_es.txt",
            "health_tips_es.txt",
            "travel_guide_es.txt",
            "swift_programming_ja.txt",
            "health_tips_ja.txt",
            "travel_guide_ja.txt",
            "swift_programming_fr.txt",
            "health_tips_fr.txt",
            "travel_guide_fr.txt",
        ]

        return try filenames.map { filename in
            guard
                let url = Bundle.main.url(
                    forResource: filename.replacingOccurrences(of: ".txt", with: ""),
                    withExtension: "txt"
                )
            else {
                throw NSError(
                    domain: "PlaidDemo",
                    code: 1,
                    userInfo: [
                        NSLocalizedDescriptionKey: "Could not find bundled document: \(filename)"
                    ]
                )
            }

            let text = try String(contentsOf: url, encoding: .utf8)
            print("  📄 Loaded \(filename): \(text.count) characters")
            return Document(filename: filename, text: text)
        }
    }
}

/// Beautiful card component for model selection
struct ModelCard: View {
    let model: ModelType
    let isSelected: Bool
    let onSelect: () -> Void

    var body: some View {
        Button(action: onSelect) {
            VStack(alignment: .leading, spacing: 16) {
                // Header with icon and name
                HStack(spacing: 12) {
                    Image(systemName: model.iconName)
                        .font(.system(size: 28))
                        .foregroundColor(model.color)
                        .frame(width: 44, height: 44)
                        .background(model.color.opacity(0.1))
                        .cornerRadius(10)

                    VStack(alignment: .leading, spacing: 4) {
                        Text(model.displayName)
                            .font(.headline)
                            .foregroundColor(.primary)

                        Text(model.description)
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }

                    Spacer()

                    // Selection indicator
                    if isSelected {
                        Image(systemName: "checkmark.circle.fill")
                            .font(.title2)
                            .foregroundColor(model.color)
                            .transition(.scale.combined(with: .opacity))
                    } else {
                        Image(systemName: "circle")
                            .font(.title2)
                            .foregroundColor(.secondary.opacity(0.3))
                    }
                }

                // Features
                VStack(alignment: .leading, spacing: 8) {
                    ForEach(model.features, id: \.self) { feature in
                        HStack(spacing: 8) {
                            Image(systemName: "checkmark")
                                .font(.caption)
                                .foregroundColor(model.color)

                            Text(feature)
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                }
            }
            .padding(20)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: 16)
                    .fill(Color(.systemBackground))
                    .shadow(
                        color: isSelected ? model.color.opacity(0.3) : Color.black.opacity(0.08),
                        radius: isSelected ? 12 : 8,
                        x: 0,
                        y: isSelected ? 6 : 4
                    )
            )
            .overlay(
                RoundedRectangle(cornerRadius: 16)
                    .strokeBorder(
                        isSelected ? model.color.opacity(0.5) : Color.clear,
                        lineWidth: 2
                    )
            )
            .scaleEffect(isSelected ? 1.02 : 1.0)
        }
        .buttonStyle(PlainButtonStyle())
        .animation(.spring(response: 0.3, dampingFraction: 0.7), value: isSelected)
    }
}

#Preview("Welcome View") {
    WelcomeView(searchEngine: SearchEngine())
}

#Preview("Model Card - Selected") {
    ModelCard(model: .lfm2, isSelected: true) {}
        .padding()
}

#Preview("Model Card - Not Selected") {
    ModelCard(model: .mxbaiEdge, isSelected: false) {}
        .padding()
}
