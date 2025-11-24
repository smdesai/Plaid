import SwiftUI
import UniformTypeIdentifiers

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
    @State private var showDirectoryPicker = false

    /// Whether search is enabled (has indexed data)
    private var isSearchEnabled: Bool {
        searchEngine.indexState != nil && (searchEngine.indexState?.totalDocuments ?? 0) > 0
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                // Index stats header
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
                } else if let currentModel = searchEngine.currentModel {
                    // Just show model when no index yet
                    HStack {
                        Label(
                            currentModel.displayName,
                            systemImage: currentModel.iconName
                        )
                        .font(.caption)
                        .foregroundColor(currentModel.color)
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
                if searchEngine.isIndexing {
                    // Indexing progress
                    indexingProgressView
                } else if isSearching {
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
                } else if !isSearchEnabled {
                    // No indexed data - prompt to add a folder
                    VStack(spacing: 16) {
                        Spacer()
                        Image(systemName: "folder.badge.questionmark")
                            .font(.system(size: 50))
                            .foregroundColor(.secondary)
                        Text("No documents indexed")
                            .font(.headline)
                            .foregroundColor(.secondary)
                        Text("Tap \"Add Folder\" above to index documents")
                            .font(.caption)
                            .foregroundColor(.secondary)
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
                        Text("Enter a query and tap search")
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
                .toolbar {
                    ToolbarItem(placement: .navigationBarLeading) {
                        Button(action: { showDirectoryPicker = true }) {
                            Image(systemName: "folder.badge.plus")
                            .font(.body)
                            .foregroundColor(searchEngine.isIndexing ? .gray : .blue)
                        }
                        .disabled(searchEngine.isIndexing)
                    }
                    ToolbarItem(placement: .navigationBarTrailing) {
                        Button(action: { showSettings = true }) {
                            Image(systemName: "gear")
                            .font(.body)
                            .foregroundColor(.blue)
                        }
                    }
                }
                .safeAreaInset(edge: .top) {
                    searchBar
                }
            #else
                .toolbar {
                    ToolbarItem(placement: .automatic) {
                        Button(action: { showDirectoryPicker = true }) {
                            Image(systemName: "folder.badge.plus")
                            .foregroundColor(searchEngine.isIndexing ? .gray : .blue)
                        }
                        .disabled(searchEngine.isIndexing)
                    }
                    ToolbarItem(placement: .automatic) {
                        Button(action: { showSettings = true }) {
                            Image(systemName: "gear")
                        }
                    }
                }
                .safeAreaInset(edge: .top) {
                    searchBar
                }
            #endif
            .sheet(item: $selectedResult) { result in
                DocumentDetailView(result: result)
            }
            .sheet(isPresented: $showSettings) {
                SettingsView(searchEngine: searchEngine)
            }
            .fileImporter(
                isPresented: $showDirectoryPicker,
                allowedContentTypes: [.folder],
                allowsMultipleSelection: false
            ) { result in
                handleDirectorySelection(result)
            }
        }
    }

    /// Indexing progress view
    private var indexingProgressView: some View {
        VStack(spacing: 24) {
            Spacer()

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

            Spacer()
        }
    }

    /// Handle directory selection from file importer
    private func handleDirectorySelection(_ result: Result<[URL], Error>) {
        switch result {
        case .success(let urls):
            guard let directoryURL = urls.first else { return }

            // Start security-scoped access
            guard directoryURL.startAccessingSecurityScopedResource() else {
                print("❌ Failed to access directory")
                return
            }

            Task {
                defer {
                    directoryURL.stopAccessingSecurityScopedResource()
                }

                do {
                    try await searchEngine.indexDirectory(at: directoryURL)
                } catch {
                    print("❌ Error indexing directory: \(error)")
                    await MainActor.run {
                        searchEngine.errorMessage = error.localizedDescription
                    }
                }
            }

        case .failure(let error):
            print("❌ Directory picker error: \(error)")
        }
    }

    /// Custom search bar with submit button
    private var searchBar: some View {
        HStack(spacing: 12) {
            HStack {
                Image(systemName: "magnifyingglass")
                    .foregroundColor(isSearchEnabled ? .secondary : .secondary.opacity(0.5))

                TextField("Search documents...", text: $searchText)
                    .textFieldStyle(.plain)
                    .disabled(!isSearchEnabled)
                    .onSubmit {
                        performSearch()
                    }

                if !searchText.isEmpty {
                    Button(action: {
                        searchText = ""
                        searchResults = []
                    }) {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundColor(.secondary)
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(10)
            #if os(iOS)
                .background(Color(UIColor.systemGray6).opacity(isSearchEnabled ? 1 : 0.5))
            #else
                .background(Color(.gray).opacity(isSearchEnabled ? 0.15 : 0.08))
            #endif
            .cornerRadius(10)

            Button(action: performSearch) {
                Image(systemName: "arrow.up.circle.fill")
                    .font(.title2)
                    .foregroundColor(!isSearchEnabled || searchText.isEmpty ? .gray : .blue)
            }
            .disabled(
                !isSearchEnabled
                    || searchText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            )
            .buttonStyle(.plain)
        }
        .padding(.horizontal)
        .padding(.vertical, 8)
        #if os(iOS)
            .background(Color(UIColor.systemBackground))
        #else
            .background(Color.white)
        #endif
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
