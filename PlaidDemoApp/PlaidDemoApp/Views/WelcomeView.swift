import SwiftUI

struct WelcomeView: View {
    @ObservedObject var searchEngine: SearchEngine
    @State private var isLoading = false
    @State private var showError = false

    var body: some View {
        VStack(spacing: 30) {
            Spacer()

            // Icon
            Image(systemName: "magnifyingglass.circle.fill")
                .font(.system(size: 80))
                .foregroundColor(.blue)

            // Title
            Text("ColBERT Search")
                .font(.largeTitle)
                .fontWeight(.bold)

            // Description
            Text("Semantic search powered by LiquidAI's LFM2-ColBERT model")
                .font(.body)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 40)

            Spacer()

            // Progress indicator or button
            if searchEngine.isIndexing {
                VStack(spacing: 16) {
                    ProgressView(value: searchEngine.indexingProgress, total: 1.0)
                        .progressViewStyle(LinearProgressViewStyle())
                        .frame(maxWidth: 300)

                    Text("Indexing: \(searchEngine.currentDocument)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)

                    Text("\(Int(searchEngine.indexingProgress * 100))%")
                        .font(.caption)
                        .fontWeight(.semibold)
                }
                .padding()
            } else {
                Button(action: startIndexing) {
                    HStack {
                        Image(systemName: "sparkles")
                        Text("Get Started")
                    }
                    .font(.headline)
                    .foregroundColor(.white)
                    .frame(maxWidth: 200)
                    .padding()
                    .background(Color.blue)
                    .cornerRadius(12)
                }
                .disabled(isLoading)
            }

            // Error message
            if let errorMessage = searchEngine.errorMessage {
                Text(errorMessage)
                    .font(.caption)
                    .foregroundColor(.red)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)
            }

            Spacer()
        }
        .padding()
    }

    private func startIndexing() {
        isLoading = true

        Task {
            do {
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

#Preview {
    WelcomeView(searchEngine: SearchEngine())
}
