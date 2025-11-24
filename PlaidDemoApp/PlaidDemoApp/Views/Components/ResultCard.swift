import SwiftUI

#if canImport(UIKit)
    import UIKit
#endif

struct ResultCard: View {
    let result: SearchResult

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header with filename and chunk info
            HStack {
                Image(systemName: "doc.text.fill")
                    .foregroundColor(.blue)

                Text(result.displayName)
                    .font(.headline)
                    .lineLimit(1)

                Spacer()

                // Score badge
                Text(String(format: "%.0f%%", result.score * 100))
                    .font(.caption)
                    .fontWeight(.semibold)
                    .foregroundColor(.white)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(scoreColor)
                    .cornerRadius(8)
            }

            // Chunk text (the relevant passage)
            Text(result.snippet)
                .font(.subheadline)
                .foregroundColor(.secondary)
                .lineLimit(5)

            // Visual separator
            Rectangle()
                .fill(Color.blue.opacity(0.3))
                .frame(height: 2)
        }
        .padding()
        #if os(iOS)
            .background(Color(UIColor.systemBackground))
        #else
            .background(Color.white)
        #endif
        .cornerRadius(12)
        .shadow(color: Color.black.opacity(0.1), radius: 4, x: 0, y: 2)
    }

    private var scoreColor: Color {
        let score = result.score
        if score >= 0.7 {
            return .green
        } else if score >= 0.4 {
            return .orange
        } else {
            return .gray
        }
    }
}

#Preview {
    VStack(spacing: 16) {
        ResultCard(
            result: SearchResult(
                documentId: 0,
                filename: "swift_programming.txt",
                chunkIndex: 0,
                score: 0.92,
                text:
                    "Swift is a powerful and intuitive programming language for iOS, macOS, watchOS, and tvOS. Writing Swift code is interactive and fun, the syntax is concise yet expressive."
            ))

        ResultCard(
            result: SearchResult(
                documentId: 1,
                filename: "health_tips.txt",
                chunkIndex: 2,
                score: 0.45,
                text:
                    "Regular exercise is important for maintaining good health. Try to get at least 30 minutes of moderate activity most days of the week."
            ))

        ResultCard(
            result: SearchResult(
                documentId: 2,
                filename: "travel_guide.txt",
                chunkIndex: 5,
                score: 0.23,
                text:
                    "When traveling abroad, always keep important documents secure. Make copies of your passport and store them separately."
            ))
    }
    .padding()
    #if os(iOS)
        .background(Color(UIColor.systemGroupedBackground))
    #else
        .background(Color(.gray).opacity(0.1))
    #endif
}
