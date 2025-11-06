import SwiftUI

#if canImport(UIKit)
    import UIKit
#endif

struct ResultCard: View {
    let result: SearchResult

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header with filename and score
            HStack {
                Image(systemName: "doc.text.fill")
                    .foregroundColor(.blue)

                Text(result.filename)
                    .font(.headline)
                    .lineLimit(1)

                Spacer()

                // Score badge (commented out for now)
                //                Text(String(format: "%.2f", result.score))
                //                    .font(.caption)
                //                    .fontWeight(.semibold)
                //                    .foregroundColor(.white)
                //                    .padding(.horizontal, 8)
                //                    .padding(.vertical, 4)
                //                    .background(scoreColor)
                //                    .cornerRadius(8)
            }

            // Text snippet
            Text(result.snippet)
                .font(.subheadline)
                .foregroundColor(.secondary)
                .lineLimit(3)

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

    // Color coding removed for now
    //    private var scoreColor: Color {
    //        let score = result.score
    //        // ColBERT scores typically range from 0-50+
    //        // High relevance: > 30, Medium: 15-30, Low: < 15
    //        if score >= 30 {
    //            return .green
    //        } else if score >= 15 {
    //            return .orange
    //        } else {
    //            return .gray
    //        }
    //    }
}

#Preview {
    VStack(spacing: 16) {
        ResultCard(
            result: SearchResult(
                documentId: 0,
                filename: "swift_programming.txt",
                score: 0.92,
                text:
                    "Swift is a powerful and intuitive programming language for iOS, macOS, watchOS, and tvOS. Writing Swift code is interactive and fun, the syntax is concise yet expressive."
            ))

        ResultCard(
            result: SearchResult(
                documentId: 1,
                filename: "health_tips.txt",
                score: 0.45,
                text:
                    "Regular exercise is important for maintaining good health. Try to get at least 30 minutes of moderate activity most days of the week."
            ))

        ResultCard(
            result: SearchResult(
                documentId: 2,
                filename: "travel_guide.txt",
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
