import SwiftUI

/// One tool, shown either as a ranked search result (with `rank`/`matchPercent`)
/// or as a plain catalog entry when browsing. Tapping the card expands the
/// arguments and the raw tool definition ("what the LLM receives").
struct ToolCard: View {
    let tool: IndexedTool
    var rank: Int? = nil
    var matchPercent: Int? = nil
    /// Hidden when the card is already inside a domain-grouped section.
    var showDomain: Bool = true

    @State private var expanded = false

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            header

            Text(tool.description)
                .font(.subheadline)
                .foregroundStyle(.primary)
                .fixedSize(horizontal: false, vertical: true)

            if !tool.properties.isEmpty {
                chips
            }

            if expanded {
                Divider()
                argumentsSection
                definitionSection
            }
        }
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .fill(Color(.systemBackground))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .stroke(Color.indigo.opacity(0.35), lineWidth: 1)
        )
        .contentShape(Rectangle())
        .onTapGesture {
            withAnimation(.easeInOut(duration: 0.2)) { expanded.toggle() }
        }
    }

    private var header: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            if let rank {
                Text("\(rank)")
                    .font(.headline)
                    .foregroundStyle(.indigo)
            }

            Text(tool.name)
                .font(.system(.subheadline, design: .monospaced))
                .fontWeight(.bold)
                .foregroundStyle(.indigo)
                .lineLimit(1)

            Spacer(minLength: 8)

            if showDomain {
                pill(tool.domain.uppercased())
            }
            if let matchPercent {
                pill("\(matchPercent)%")
            }

            Image(systemName: expanded ? "chevron.up" : "chevron.down")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private var chips: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 6) {
                ForEach(tool.properties) { argument in
                    Text(argument.name)
                        .font(.system(.caption, design: .monospaced))
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(Capsule().stroke(Color.secondary.opacity(0.4)))
                }
            }
        }
    }

    private var argumentsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("ARGUMENTS")
                .font(.caption)
                .fontWeight(.semibold)
                .foregroundStyle(.secondary)

            ForEach(tool.properties) { argument in
                HStack(spacing: 8) {
                    Text(argument.name)
                        .font(.system(.subheadline, design: .monospaced))
                        .fontWeight(.semibold)
                    pill(argument.type)
                    if argument.required {
                        pill("REQUIRED", tint: .secondary)
                    }
                    Spacer()
                }
            }
        }
    }

    private var definitionSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("TOOL DEFINITION (WHAT THE LLM RECEIVES)")
                .font(.caption)
                .fontWeight(.semibold)
                .foregroundStyle(.secondary)

            ScrollView(.horizontal, showsIndicators: false) {
                Text(tool.definitionJSON)
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.primary)
                    .padding(12)
            }
            .background(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .fill(Color(.secondarySystemBackground))
            )
        }
    }

    private func pill(_ text: String, tint: Color = .indigo) -> some View {
        Text(text)
            .font(.caption2)
            .fontWeight(.semibold)
            .foregroundStyle(tint)
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(Capsule().stroke(tint.opacity(0.5)))
    }
}
