import Foundation

/// A single tool argument, derived from a tool's JSON-Schema `properties`.
struct ToolArgument: Identifiable {
    let name: String
    let type: String
    let required: Bool

    var id: String { name }
}

/// A tool flattened out of its domain, ready to index and display.
struct IndexedTool: Identifiable {
    let id = UUID()
    let name: String
    let domain: String
    let description: String
    /// Arguments sorted required-first, then alphabetically.
    let properties: [ToolArgument]
    /// Pretty-printed JSON of the raw tool object ("what the LLM receives").
    let definitionJSON: String
}

/// All tools that share a domain, for browsing the catalog grouped by domain.
struct ToolDomainGroup: Identifiable {
    let domain: String
    let tools: [IndexedTool]

    var id: String { domain }
}

/// Loads `colbert_tool_definitions.json` from the app bundle and flattens the
/// per-domain tool lists into a single `[IndexedTool]`.
///
/// The corpus is an array of `{ domain, tools: [{ name, description, parameters }] }`.
/// The tool `description` is what gets embedded/indexed; everything else is surfaced
/// in the result card.
enum ToolCatalog {
    static func load(resource: String = "colbert_tool_definitions") -> [IndexedTool] {
        guard
            let url = Bundle.main.url(forResource: resource, withExtension: "json"),
            let data = try? Data(contentsOf: url),
            let root = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]]
        else {
            return []
        }

        var tools: [IndexedTool] = []
        for domainObject in root {
            let domain = domainObject["domain"] as? String ?? "general"
            let toolList = domainObject["tools"] as? [[String: Any]] ?? []

            for toolDict in toolList {
                guard
                    let name = toolDict["name"] as? String,
                    let description = toolDict["description"] as? String
                else { continue }

                let parameters = toolDict["parameters"] as? [String: Any]
                let propertyMap = parameters?["properties"] as? [String: Any] ?? [:]
                let requiredSet = Set(parameters?["required"] as? [String] ?? [])

                var arguments: [ToolArgument] = propertyMap.map { key, value in
                    let type = (value as? [String: Any])?["type"] as? String ?? "any"
                    return ToolArgument(name: key, type: type, required: requiredSet.contains(key))
                }
                // Required arguments first, then alphabetical — deterministic display order.
                arguments.sort {
                    ($0.required ? 0 : 1, $0.name) < ($1.required ? 0 : 1, $1.name)
                }

                tools.append(
                    IndexedTool(
                        name: name,
                        domain: domain,
                        description: description,
                        properties: arguments,
                        definitionJSON: prettyJSON(toolDict)
                    )
                )
            }
        }
        return tools
    }

    /// Groups tools by domain, preserving the domains' first-seen order in the corpus.
    static func grouped(_ tools: [IndexedTool]) -> [ToolDomainGroup] {
        var order: [String] = []
        var byDomain: [String: [IndexedTool]] = [:]
        for tool in tools {
            if byDomain[tool.domain] == nil { order.append(tool.domain) }
            byDomain[tool.domain, default: []].append(tool)
        }
        return order.map { ToolDomainGroup(domain: $0, tools: byDomain[$0] ?? []) }
    }

    /// Serializes the raw tool object with stable (alphabetical) key ordering so the
    /// displayed definition includes every field the model receives (enums, nested
    /// schemas, etc.) rather than a lossy re-encode of a typed model.
    private static func prettyJSON(_ object: Any) -> String {
        guard
            let data = try? JSONSerialization.data(
                withJSONObject: object, options: [.prettyPrinted, .sortedKeys]),
            let string = String(data: data, encoding: .utf8)
        else {
            return "{}"
        }
        return string
    }
}
