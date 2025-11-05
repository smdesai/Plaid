public protocol TokenizerProtocol {
    func tokenize(text: String) -> [String]
    func detokenize(tokens: [String]) -> String
}
