import Plaid
import SwiftUI

/// Browse the documents currently indexed in the store. A "document" is the
/// group of chunks that share a source filename, so this lists one row per
/// document with its chunk and embedding totals.
///
/// A whole document can be deleted here; `onChange` fires after any deletion so
/// the presenting view can refresh state the delete may have invalidated (e.g.
/// stale search-result ids, since deletion re-sequences every surviving id).
struct IndexedDocumentsView: View {
    @ObservedObject var searchEngine: SearchEngine
    var onChange: () -> Void = {}

    @Environment(\.dismiss) private var dismiss
    @State private var documents: [IndexedDocument] = []
    @State private var isLoading = true
    @State private var loadError: String?
    @State private var pendingDeletion: IndexedDocument?

    var body: some View {
        NavigationStack {
            content
                .navigationTitle("Indexed Documents")
                #if os(iOS)
                    .navigationBarTitleDisplayMode(.inline)
                #endif
                .toolbar {
                    ToolbarItem(placement: .cancellationAction) {
                        Button("Done") { dismiss() }
                    }
                }
                .task { await load() }
                .alert(
                    "Delete Document?",
                    isPresented: Binding(
                        get: { pendingDeletion != nil },
                        set: { if !$0 { pendingDeletion = nil } }
                    ),
                    presenting: pendingDeletion
                ) { doc in
                    Button("Cancel", role: .cancel) {}
                    Button("Delete", role: .destructive) { delete(doc) }
                } message: { doc in
                    Text(
                        "This removes all \(doc.chunkCount) chunk\(doc.chunkCount == 1 ? "" : "s") of “\(doc.documentName)” from the index."
                    )
                }
        }
    }

    @ViewBuilder
    private var content: some View {
        if isLoading {
            ProgressView("Loading documents…")
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else if let loadError {
            emptyState(
                icon: "exclamationmark.triangle",
                title: "Couldn’t load documents",
                subtitle: loadError
            )
        } else if documents.isEmpty {
            emptyState(
                icon: "folder.badge.questionmark",
                title: "No documents indexed",
                subtitle: "Add a folder or load the sample documents to get started."
            )
        } else {
            List {
                Section {
                    ForEach(documents) { doc in
                        documentRow(doc)
                            .swipeActions(edge: .trailing) {
                                Button(role: .destructive) {
                                    pendingDeletion = doc
                                } label: {
                                    Label("Delete", systemImage: "trash")
                                }
                            }
                            .contextMenu {
                                Button(role: .destructive) {
                                    pendingDeletion = doc
                                } label: {
                                    Label("Delete Document", systemImage: "trash")
                                }
                            }
                    }
                } header: {
                    Text("\(documents.count) document\(documents.count == 1 ? "" : "s")")
                }
            }
        }
    }

    private func documentRow(_ doc: IndexedDocument) -> some View {
        HStack(spacing: 12) {
            Image(systemName: "doc.text.fill")
                .font(.title3)
                .foregroundColor(.blue)
                .frame(width: 28)

            VStack(alignment: .leading, spacing: 4) {
                Text(doc.documentName)
                    .font(.subheadline)
                    .fontWeight(.medium)
                    .lineLimit(1)
                    .truncationMode(.middle)

                Text(
                    "\(doc.chunkCount) chunk\(doc.chunkCount == 1 ? "" : "s") · \(doc.embeddingCount) embeddings"
                )
                .font(.caption)
                .foregroundColor(.secondary)
            }

            Spacer()
        }
        .padding(.vertical, 4)
    }

    private func emptyState(icon: String, title: String, subtitle: String) -> some View {
        VStack(spacing: 12) {
            Image(systemName: icon)
                .font(.system(size: 44))
                .foregroundColor(.secondary)
            Text(title)
                .font(.headline)
                .foregroundColor(.secondary)
            Text(subtitle)
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding()
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    @MainActor
    private func load() async {
        isLoading = true
        loadError = nil
        do {
            documents = try await searchEngine.indexedDocuments()
        } catch {
            print("❌ Error loading indexed documents: \(error)")
            loadError = error.localizedDescription
        }
        isLoading = false
    }

    private func delete(_ doc: IndexedDocument) {
        Task {
            do {
                _ = try await searchEngine.deleteDocument(named: doc.documentName)
                onChange()
                await load()
            } catch {
                print("❌ Error deleting document: \(error)")
                await MainActor.run {
                    searchEngine.errorMessage = error.localizedDescription
                }
            }
        }
    }
}

#Preview {
    IndexedDocumentsView(searchEngine: SearchEngine())
}
