import SwiftUI

/// Root screen: a search bar on top, ranked tool-match cards below.
struct ToolSelectionView: View {
    @StateObject private var model = ToolSearchModel()

    var body: some View {
        NavigationStack {
            VStack(spacing: 16) {
                SearchField(text: $model.query, onClear: model.clear)
                    .padding(.horizontal)
                    .padding(.top, 8)

                content
            }
            .navigationTitle("Tool Selection")
            .navigationBarTitleDisplayMode(.inline)
        }
        .task {
            await model.bootstrap()
            // Optional UI-test / deep-link hook: prefill a query once the index is ready.
            if let preset = ProcessInfo.processInfo.environment["TOOLSELECTION_AUTOQUERY"],
                !preset.isEmpty
            {
                model.query = preset
            }
        }
        .onChange(of: model.query) { _, _ in model.queryChanged() }
    }

    @ViewBuilder
    private var content: some View {
        switch model.phase {
        case .loading:
            status(title: "Loading model…", systemImage: "arrow.down.circle")

        case .indexing(let done, let total):
            VStack(spacing: 12) {
                Spacer()
                ProgressView(value: Double(done), total: Double(max(total, 1)))
                    .tint(.indigo)
                    .padding(.horizontal, 40)
                Text("Indexing tools \(done)/\(total)")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                Spacer()
            }

        case .ready:
            if isQueryEmpty {
                browseList
            } else {
                searchResults
            }

        case .failed(let message):
            status(title: message, systemImage: "exclamationmark.triangle")
        }
    }

    private var isQueryEmpty: Bool {
        model.query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    /// No active query → browse the whole catalog grouped by domain (sticky headers).
    private var browseList: some View {
        ScrollView {
            LazyVStack(spacing: 12, pinnedViews: [.sectionHeaders]) {
                ForEach(model.catalog) { group in
                    Section {
                        ForEach(group.tools) { tool in
                            ToolCard(tool: tool, showDomain: false)
                                .padding(.horizontal)
                        }
                    } header: {
                        domainHeader(group)
                    }
                }
            }
            .padding(.bottom, 24)
        }
    }

    private func domainHeader(_ group: ToolDomainGroup) -> some View {
        HStack(spacing: 8) {
            Text(group.domain.uppercased())
                .font(.caption)
                .fontWeight(.bold)
                .foregroundStyle(.indigo)
            Text("\(group.tools.count)")
                .font(.caption2)
                .fontWeight(.semibold)
                .foregroundStyle(.secondary)
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(Capsule().fill(Color.indigo.opacity(0.12)))
            Spacer()
        }
        .padding(.horizontal)
        .padding(.vertical, 8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.regularMaterial)
    }

    @ViewBuilder
    private var searchResults: some View {
        if model.results.isEmpty {
            VStack(spacing: 10) {
                Spacer()
                Image(systemName: "magnifyingglass")
                    .font(.largeTitle)
                    .foregroundStyle(.indigo.opacity(0.6))
                Text("No matching tools")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                Spacer()
            }
        } else {
            ScrollView {
                LazyVStack(spacing: 12) {
                    ForEach(Array(model.results.enumerated()), id: \.element.id) { index, match in
                        ToolCard(
                            tool: match.tool, rank: index + 1, matchPercent: match.matchPercent)
                    }
                }
                .padding(.horizontal)
                .padding(.bottom, 24)
            }
        }
    }

    private func status(title: String, systemImage: String) -> some View {
        VStack(spacing: 12) {
            Spacer()
            Image(systemName: systemImage)
                .font(.largeTitle)
                .foregroundStyle(.indigo)
            Text(title)
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal)
            Spacer()
        }
    }
}
