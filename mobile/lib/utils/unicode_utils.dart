String unicodeLiteralToChar(String unicodeLiteral) {
  // "\\u{03B1}" -> "α"
  final reg = RegExp(r'\{([0-9A-Fa-f]+)\}');
  final m = reg.firstMatch(unicodeLiteral);
  if (m == null) return '?';
  final hex = m.group(1);
  if (hex == null) return '?';
  try {
    final cp = int.parse(hex, radix: 16);
    return String.fromCharCode(cp);
  } catch (_) {
    return '?';
  }
}
