import SwiftUI

#if canImport(UIKit)
    import UIKit
#endif

struct ContentView: View {
    @StateObject private var searchEngine = SearchEngine()
    @State private var isInitialized = false
    @State private var initializationError: Error?

    var body: some View {
        Group {
            if let error = initializationError {
                ErrorView(error: error) {
                    Task {
                        await initializeEngine()
                    }
                }
            } else if !isInitialized {
                LoadingView()
            } else if searchEngine.modelReady {
                // Once model is selected and loaded, go to SearchView
                SearchView(searchEngine: searchEngine)
            } else {
                // Show WelcomeView for model selection
                WelcomeView(searchEngine: searchEngine)
            }
        }
        .task {
            await initializeEngine()
        }
    }

    private func initializeEngine() async {
        isInitialized = false
        initializationError = nil

        do {
            // Try to load with saved model preference
            try await searchEngine.initialize()

            // If model loaded successfully and has an index, mark as ready
            if searchEngine.currentModel != nil && searchEngine.hasIndex {
                searchEngine.modelReady = true
            }

            isInitialized = true
        } catch {
            print("❌ Initialization error: \(error)")
            // If initialization fails, still show WelcomeView for fresh start
            isInitialized = true
        }
    }
}

/// Loading view shown during model initialization
struct LoadingView: View {
    var body: some View {
        VStack(spacing: 20) {
            ProgressView()
                .scaleEffect(1.5)

            Text("Loading ColBERT Model...")
                .font(.headline)
                .foregroundColor(.secondary)

            Text("This may take a few seconds on first launch")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding()
    }
}

/// Error view with retry button
struct ErrorView: View {
    let error: Error
    let onRetry: () -> Void

    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 60))
                .foregroundColor(.orange)

            Text("Initialization Failed")
                .font(.title2)
                .fontWeight(.bold)

            Text(error.localizedDescription)
                .font(.body)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal)

            Button(action: onRetry) {
                HStack {
                    Image(systemName: "arrow.clockwise")
                    Text("Retry")
                }
                .font(.headline)
                .foregroundColor(.white)
                .padding()
                .background(Color.blue)
                .cornerRadius(12)
            }
        }
        .padding()
    }
}

#Preview {
    ContentView()
}
