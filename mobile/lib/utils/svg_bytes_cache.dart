import 'dart:convert';
import 'dart:typed_data';

class SvgBytesCache {
  static final Map<String, Uint8List> _cache = <String, Uint8List>{};

  static Uint8List decode(String b64) {
    return _cache.putIfAbsent(b64, () => base64Decode(b64));
  }
}
