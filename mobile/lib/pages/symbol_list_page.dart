import 'package:flutter/material.dart';
import 'package:mobile/models/candidate_row.dart';
import 'package:mobile/models/symbol_models.dart';
import 'package:mobile/services/mappings_service.dart';
import 'package:mobile/utils/unicode_utils.dart';
import 'package:mobile/widgets/candidate_row_tile.dart';

class SymbolListPage extends StatefulWidget {
  const SymbolListPage({super.key});

  @override
  State<SymbolListPage> createState() => _SymbolListPageState();
}

class _SymbolListPageState extends State<SymbolListPage> {
  Map<String, MappingItem> _mappings = {};
  bool _loading = true;
  String _q = '';

  @override
  void initState() {
    super.initState();
    _boot();
  }

  Future<void> _boot() async {
    setState(() => _loading = true);
    try {
      _mappings = await MappingsService.loadMappings();
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  bool _match(MappingItem item, String label, String q) {
    if (q.isEmpty) return true;
    final qq = q.toLowerCase();

    final symbol = item.symbol.toLowerCase();
    final unicodeLit = item.unicode.toLowerCase();
    final ch = unicodeLiteralToChar(item.unicode).toLowerCase();
    final lbl = label.toLowerCase();

    // 搜索：symbol / unicode literal / 实际字符 / label
    return symbol.contains(qq) ||
        unicodeLit.contains(qq) ||
        ch.contains(qq) ||
        lbl.contains(qq);
  }

  List<CandidateRow> _filteredRows() {
    final entries = _mappings.entries.toList();

    // label 若为数字字符串，按数值排序；否则按字符串排序
    entries.sort((a, b) {
      final ai = int.tryParse(a.key);
      final bi = int.tryParse(b.key);
      if (ai != null && bi != null) return ai.compareTo(bi);
      return a.key.compareTo(b.key);
    });

    final out = <CandidateRow>[];
    for (final e in entries) {
      if (_match(e.value, e.key, _q)) {
        out.add(CandidateRow(label: e.key, item: e.value, prob: null));
      }
    }
    return out;
  }

  @override
  Widget build(BuildContext context) {
    final rows = _filteredRows();

    return Padding(
      padding: const EdgeInsets.all(12),
      child: Column(
        children: [
          Row(
            children: [
              Text('符号库', style: Theme.of(context).textTheme.headlineSmall),
              const Spacer(),
              Text(_loading ? '加载中…' : '共 ${rows.length} 项'),
            ],
          ),
          const SizedBox(height: 12),

          // 搜索框（始终置顶）
          TextField(
            onChanged: (v) => setState(() => _q = v.trim()),
            decoration: InputDecoration(
              hintText: r'搜索 LaTeX、Unicode',
              prefixIcon: const Icon(Icons.search),
              border: const OutlineInputBorder(),
              isDense: true,
            ),
          ),
          const SizedBox(height: 12),

          Expanded(
            child: Card(
              elevation: 0,
              color: Theme.of(context).colorScheme.surfaceContainerHighest,
              child: _loading
                  ? const Center(child: CircularProgressIndicator())
                  : rows.isEmpty
                      ? const Center(child: Text('无匹配结果'))
                      : ListView.separated(
                          padding: const EdgeInsets.all(12),
                          itemCount: rows.length,
                          separatorBuilder: (_, __) => const Divider(height: 16),
                          itemBuilder: (context, i) {
                            return CandidateRowTile(row: rows[i]);
                          },
                        ),
            ),
          ),
        ],
      ),
    );
  }
}
