import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:mobile/models/symbol_models.dart';
import 'package:mobile/utils/unicode_utils.dart';
import 'package:mobile/utils/svg_bytes_cache.dart';
import 'package:flutter_svg/flutter_svg.dart' as fsvg;

class SymbolView extends StatelessWidget {
  final MappingItem item;
  final double size;

  const SymbolView({
    super.key,
    required this.item,
    this.size = 28,
  });

  @override
  Widget build(BuildContext context) {
    final svgBase64 = item.svg;

    // 兜底：如果某些符号没有 SVG
    if (svgBase64 == null || svgBase64.isEmpty) {
      final ch = unicodeLiteralToChar(item.unicode);
      return Text(
        ch,
        style: TextStyle(fontSize: size),
      );
    }

    final bytes = SvgBytesCache.decode(svgBase64);

    return SizedBox(
      width: size,
      height: size,
      child: SvgPicture.memory(
        bytes,
        fit: BoxFit.contain,
      ),
    );
  }
}
