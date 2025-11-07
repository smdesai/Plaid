import SwiftUI

#if canImport(UIKit)
    import UIKit
#endif

struct SearchView: View {
    @ObservedObject var searchEngine: SearchEngine
    @State private var searchText = ""
    @State private var searchResults: [SearchResult] = []
    @State private var isSearching = false
    @State private var selectedResult: SearchResult?
    @State private var showSettings = false

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                // Index stats
                if let indexState = searchEngine.indexState {
                    HStack(spacing: 16) {
                        if let currentModel = searchEngine.currentModel {
                            Label(
                                currentModel.displayName,
                                systemImage: currentModel.iconName
                            )
                            .font(.caption)
                            .foregroundColor(currentModel.color)
                        }

                        Label(
                            "\(indexState.totalDocuments) docs",
                            systemImage: "doc.text"
                        )
                        .font(.caption)
                        .foregroundColor(.secondary)

                        Label(
                            "\(indexState.totalEmbeddings) embeddings",
                            systemImage: "brain"
                        )
                        .font(.caption)
                        .foregroundColor(.secondary)

                        Spacer()
                    }
                    .padding(.horizontal)
                    .padding(.vertical, 8)
                    #if os(iOS)
                        .background(Color(UIColor.systemGroupedBackground))
                    #else
                        .background(Color(.gray).opacity(0.1))
                    #endif
                }

                // Results list
                if isSearching {
                    VStack {
                        Spacer()
                        ProgressView()
                            .scaleEffect(1.2)
                        Text("Searching...")
                            .font(.caption)
                            .foregroundColor(.secondary)
                            .padding(.top, 8)
                        Spacer()
                    }
                } else if searchResults.isEmpty && !searchText.isEmpty {
                    VStack(spacing: 16) {
                        Spacer()
                        Image(systemName: "magnifyingglass")
                            .font(.system(size: 50))
                            .foregroundColor(.secondary)
                        Text("No results found")
                            .font(.headline)
                            .foregroundColor(.secondary)
                        Text("Try a different search query")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Spacer()
                    }
                } else if searchResults.isEmpty {
                    VStack(spacing: 16) {
                        Spacer()
                        Image(systemName: "doc.text.magnifyingglass")
                            .font(.system(size: 50))
                            .foregroundColor(.secondary)
                        Text("Enter a query to search")
                            .font(.headline)
                            .foregroundColor(.secondary)
                        Text("Try: \"How to handle errors in Swift?\"")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Spacer()
                    }
                } else {
                    ScrollView {
                        LazyVStack(spacing: 12) {
                            ForEach(searchResults) { result in
                                ResultCard(result: result)
                                    .onTapGesture {
                                        selectedResult = result
                                    }
                            }
                        }
                        .padding()
                    }
                }
            }
            .navigationTitle("Search")
            #if os(iOS)
                .navigationBarTitleDisplayMode(.large)
                .searchable(
                    text: $searchText,
                    placement: .navigationBarDrawer(displayMode: .always),
                    prompt: "Search documents..."
                )
                .toolbar {
                    ToolbarItem(placement: .navigationBarTrailing) {
                        Button(action: { showSettings = true }) {
                            Image(systemName: "gear")
                            .font(.body)
                            .foregroundColor(.blue)
                        }
                    }
                }
            #else
                .searchable(
                    text: $searchText,
                    prompt: "Search documents..."
                )
                .toolbar {
                    ToolbarItem(placement: .automatic) {
                        Button(action: { showSettings = true }) {
                            Image(systemName: "gear")
                        }
                    }
                }
            #endif
            .onSubmit(of: .search) {
                performSearch()
            }
            .onChange(of: searchText) { oldValue, newValue in
                // Auto-search when text changes (with debounce would be better)
                if !newValue.isEmpty {
                    performSearch()
                } else {
                    searchResults = []
                }
            }
            .sheet(item: $selectedResult) { result in
                DocumentDetailView(result: result)
            }
            .sheet(isPresented: $showSettings) {
                SettingsView(searchEngine: searchEngine)
            }
        }
    }

    private func performSearch() {
        guard !searchText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            searchResults = []
            return
        }

        isSearching = true

        Task {
            do {
                let results = try await searchEngine.search(query: searchText, topK: 3)
                await MainActor.run {
                    searchResults = results
                    isSearching = false
                }
            } catch {
                print("❌ Search error: \(error)")
                await MainActor.run {
                    searchResults = []
                    isSearching = false
                }
            }
        }
    }
}

/// Detail view showing full document text
struct DocumentDetailView: View {
    let result: SearchResult
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    // Header
                    HStack {
                        Image(systemName: "doc.text.fill")
                            .font(.title2)
                            .foregroundColor(.blue)

                        VStack(alignment: .leading) {
                            Text(result.filename)
                                .font(.headline)

                            Text("Relevance: \(result.scorePercentage)%")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }

                        Spacer()
                    }
                    .padding()
                    #if os(iOS)
                        .background(Color(UIColor.systemGroupedBackground))
                    #else
                        .background(Color(.gray).opacity(0.1))
                    #endif
                    .cornerRadius(12)

                    // Full text
                    Text(result.text)
                        .font(.body)
                        .textSelection(.enabled)
                }
                .padding()
            }
            .navigationTitle("Document")
            #if os(iOS)
                .navigationBarTitleDisplayMode(.inline)
                .toolbar {
                    ToolbarItem(placement: .navigationBarTrailing) {
                        Button("Done") {
                            dismiss()
                        }
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
        }
    }
}

#Preview {
    SearchView(searchEngine: SearchEngine())
}
