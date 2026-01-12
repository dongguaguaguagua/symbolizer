class MappingItem {
  final String symbol;   // LaTeX
  final String unicode;  // "\u{03B1}"
  final String? svg;     // Base64 SVG

  const MappingItem({
    required this.symbol,
    required this.unicode,
    this.svg,
  });

  factory MappingItem.fromJson(Map<String, dynamic> json) {
    return MappingItem(
      symbol: (json['symbol'] ?? '').toString(),
      unicode: (json['unicode'] ?? '').toString(),
      svg: json['svg']?.toString(),
    );
  }
}


class TopCandidate {
  final int index;
  final double prob;

  const TopCandidate({required this.index, required this.prob});
}
