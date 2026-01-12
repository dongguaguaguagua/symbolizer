import 'dart:convert';
import 'package:flutter/services.dart';
import 'package:mobile/models/symbol_models.dart';

class MappingsService {
  static Map<String, MappingItem>? _cache;

  static Future<Map<String, MappingItem>> loadMappings() async {
    if (_cache != null) return _cache!;
    final s = await rootBundle.loadString('assets/mappings.json');
    final Map<String, dynamic> raw = jsonDecode(s) as Map<String, dynamic>;

    final out = <String, MappingItem>{};
    for (final e in raw.entries) {
      out[e.key] = MappingItem.fromJson(e.value as Map<String, dynamic>);
    }
    _cache = out;
    return out;
  }
}
