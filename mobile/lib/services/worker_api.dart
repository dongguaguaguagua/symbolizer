import 'dart:convert';
import 'package:http/http.dart' as http;

class WorkerApi {
  final String base;

  const WorkerApi(this.base);

  Future<Map<String, dynamic>> fetchRandomSymbol() async {
    final uri = Uri.parse('$base/random-symbol');
    final res = await http.get(uri);
    if (res.statusCode < 200 || res.statusCode >= 300) {
      throw Exception('random-symbol failed: ${res.statusCode}');
    }
    return jsonDecode(res.body) as Map<String, dynamic>;
  }

  Future<void> submitSample({
    required String label,
    required List<int> gray32,
  }) async {
    final uri = Uri.parse('$base/submit');
    final body = jsonEncode({
      'label': label,
      'image': base64Encode(gray32),
    });

    final res = await http.post(
      uri,
      headers: {'Content-Type': 'application/json'},
      body: body,
    );

    if (res.statusCode < 200 || res.statusCode >= 300) {
      throw Exception('submit failed: ${res.statusCode} ${res.body}');
    }
  }
}
