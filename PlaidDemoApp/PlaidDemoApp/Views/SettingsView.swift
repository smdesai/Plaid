import SwiftUI

struct SettingsView: View {
    @ObservedObject var searchEngine: SearchEngine
    @Environment(\.dismiss) private var dismiss
    @State private var selectedModel: ModelType
    @State private var showDeleteConfirmation = false
    @State private var showModelChangeWarning = false
    @State private var isDeleting = false

    init(searchEngine: SearchEngine) {
        self.searchEngine = searchEngine
        // Initialize with current model or default to LFM2
        _selectedModel = State(initialValue: searchEngine.currentModel ?? .lfm2)
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 24) {
                    // Current Index Section
                    if let indexState = searchEngine.indexState,
                        let currentModel = searchEngine.currentModel
                    {
                        VStack(alignment: .leading, spacing: 16) {
                            sectionHeader(title: "Current Index", icon: "folder.fill")

                            VStack(spacing: 0) {
                                indexInfoRow(
                                    label: "Model",
                                    value: currentModel.displayName,
                                    icon: currentModel.iconName,
                                    iconColor: currentModel.color
                                )

                                Divider().padding(.leading, 44)

                                indexInfoRow(
                                    label: "Documents",
                                    value: "\(indexState.totalDocuments)",
                                    icon: "doc.text",
                                    iconColor: .blue
                                )

                                Divider().padding(.leading, 44)

                                indexInfoRow(
                                    label: "Embeddings",
                                    value: "\(indexState.totalEmbeddings)",
                                    icon: "brain",
                                    iconColor: .purple
                                )

                                Divider().padding(.leading, 44)

                                indexInfoRow(
                                    label: "Created",
                                    value: formatDate(indexState.createdAt),
                                    icon: "calendar",
                                    iconColor: .green
                                )
                            }
                            .background(Color(.secondarySystemGroupedBackground))
                            .cornerRadius(12)
                        }
                    }

                    // Model Selection Section
                    VStack(alignment: .leading, spacing: 16) {
                        sectionHeader(title: "Model Selection", icon: "cpu")

                        VStack(spacing: 12) {
                            ForEach(ModelType.allCases) { model in
                                modelSelectionRow(model: model)
                            }
                        }
                        .background(Color(.secondarySystemGroupedBackground))
                        .cornerRadius(12)

                        // Warning banner if model changed
                        if hasModelChanged {
                            HStack(spacing: 12) {
                                Image(systemName: "exclamationmark.triangle.fill")
                                    .foregroundColor(.orange)

                                VStack(alignment: .leading, spacing: 4) {
                                    Text("Re-indexing Required")
                                        .font(.subheadline)
                                        .fontWeight(.semibold)

                                    Text("Changing models requires rebuilding the index")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }

                                Spacer()
                            }
                            .padding()
                            .background(Color.orange.opacity(0.1))
                            .cornerRadius(12)
                        }
                    }

                    // Danger Zone Section
                    VStack(alignment: .leading, spacing: 16) {
                        sectionHeader(title: "Danger Zone", icon: "exclamationmark.triangle.fill")

                        Button(action: { showDeleteConfirmation = true }) {
                            HStack {
                                Image(systemName: "trash.fill")
                                    .foregroundColor(.red)

                                VStack(alignment: .leading, spacing: 4) {
                                    Text("Delete Index")
                                        .font(.subheadline)
                                        .fontWeight(.medium)
                                        .foregroundColor(.primary)

                                    Text("This will remove all indexed data")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }

                                Spacer()

                                Image(systemName: "chevron.right")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            .padding()
                            .background(Color(.secondarySystemGroupedBackground))
                            .cornerRadius(12)
                        }
                        .disabled(isDeleting)
                    }

                    Spacer(minLength: 20)
                }
                .padding()
            }
            .background(Color(.systemGroupedBackground))
            .navigationTitle("Settings")
            #if os(iOS)
                .navigationBarTitleDisplayMode(.inline)
                .toolbar {
                    ToolbarItem(placement: .navigationBarTrailing) {
                        Button("Done") {
                            dismiss()
                        }
                        .fontWeight(.semibold)
                    }
                }
            #else
                .toolbar {
                    ToolbarItem(placement: .automatic) {
                        Button("Done") {
                            dismiss()
                        }
                    }
                }
            #endif
            .alert("Delete Index?", isPresented: $showDeleteConfirmation) {
                Button("Cancel", role: .cancel) {}
                Button("Delete", role: .destructive) {
                    deleteIndex()
                }
            } message: {
                Text(
                    "This will permanently delete all indexed documents and embeddings. You'll need to create a new index to search again."
                )
            }
        }
    }

    // MARK: - View Components

    private func sectionHeader(title: String, icon: String) -> some View {
        HStack(spacing: 8) {
            Image(systemName: icon)
                .font(.subheadline)
                .foregroundColor(.secondary)

            Text(title)
                .font(.headline)
                .foregroundColor(.secondary)
        }
    }

    private func indexInfoRow(label: String, value: String, icon: String, iconColor: Color)
        -> some View
    {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .font(.body)
                .foregroundColor(iconColor)
                .frame(width: 28)

            Text(label)
                .font(.subheadline)
                .foregroundColor(.primary)

            Spacer()

            Text(value)
                .font(.subheadline)
                .fontWeight(.medium)
                .foregroundColor(.secondary)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
    }

    private func modelSelectionRow(model: ModelType) -> some View {
        Button(action: {
            withAnimation {
                selectedModel = model
            }
        }) {
            HStack(spacing: 12) {
                Image(systemName: model.iconName)
                    .font(.title3)
                    .foregroundColor(model.color)
                    .frame(width: 32)

                VStack(alignment: .leading, spacing: 2) {
                    Text(model.displayName)
                        .font(.subheadline)
                        .fontWeight(.medium)
                        .foregroundColor(.primary)

                    Text(model.description)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                Spacer()

                if selectedModel == model {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundColor(model.color)
                        .transition(.scale.combined(with: .opacity))
                } else {
                    Image(systemName: "circle")
                        .foregroundColor(.secondary.opacity(0.3))
                }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
        }
        .buttonStyle(PlainButtonStyle())
    }

    private var hasModelChanged: Bool {
        guard let currentModel = searchEngine.currentModel else { return false }
        return selectedModel != currentModel
    }

    // MARK: - Helper Methods

    private func formatDate(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .none
        return formatter.string(from: date)
    }

    private func deleteIndex() {
        isDeleting = true

        Task {
            do {
                try await searchEngine.deleteIndex()
                await MainActor.run {
                    isDeleting = false
                    dismiss()
                }
            } catch {
                print("❌ Error deleting index: \(error)")
                await MainActor.run {
                    searchEngine.errorMessage = error.localizedDescription
                    isDeleting = false
                }
            }
        }
    }
}

#Preview {
    // Create a search engine with mock data
    let engine = SearchEngine()
    engine.hasIndex = true
    engine.currentModel = .lfm2
    engine.indexState = IndexState(
        documents: [:],
        createdAt: Date(),
        lastModified: Date(),
        totalDocuments: 12,
        totalEmbeddings: 12453
    )

    return SettingsView(searchEngine: engine)
}
