import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:mobile/models/candidate_row.dart';
import 'package:mobile/utils/unicode_utils.dart';
import 'package:mobile/widgets/symbol_view.dart';

class CandidateRowTile extends StatelessWidget {
  final CandidateRow row;

  const CandidateRowTile({
    super.key,
    required this.row,
  });

  void _copy(BuildContext context, String text) {
    Clipboard.setData(ClipboardData(text: text));
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('已复制到剪贴板')),
    );
  }

  @override
  Widget build(BuildContext context) {
    final latex = row.item.symbol;
    final unicodeLiteral = row.item.unicode;
    final ch = unicodeLiteralToChar(unicodeLiteral);

    return InkWell(
      borderRadius: BorderRadius.circular(8),
      onTapDown: (details) async {
        final pos = details.globalPosition;

        final selected = await showMenu<String>(
          context: context,
          position: RelativeRect.fromLTRB(
            pos.dx,
            pos.dy,
            pos.dx + 1,
            pos.dy + 1,
          ),
          items: [
            PopupMenuItem(
              value: latex,
              child: const Text('复制 LaTeX'),
            ),
            PopupMenuItem(
              value: unicodeLiteral,
              child: const Text('复制 Unicode'),
            ),
            PopupMenuItem(
              value: ch,
              child: const Text('复制 Symbol'),
            ),
          ],
        );

        if (selected != null) {
          _copy(context, selected);
        }
      },
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // 符号显示（SVG）
          Container(
            width: 72,
            height: 72,
            alignment: Alignment.center,
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(10),
              border: Border.all(
                color: Theme.of(context).colorScheme.outlineVariant,
              ),
            ),
            child: SymbolView(
              item: row.item,
              size: 60, // 这里调大/调小（受父容器 72×48 限制）
            ),
          ),
          const SizedBox(width: 12),

          Expanded(
            child: DefaultTextStyle(
              style: Theme.of(context).textTheme.bodyMedium!,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Label: ${row.label}'),
                  const SizedBox(height: 2),
                  SelectableText(
                    'LaTeX: $latex',
                    style: const TextStyle(fontFamily: 'monospace'),
                  ),
                  SelectableText(
                    'Unicode: $unicodeLiteral',
                    style: const TextStyle(fontFamily: 'monospace'),
                  ),
                  SelectableText(
                    'Symbol: $ch',
                    style: const TextStyle(fontFamily: 'monospace'),
                  ),
                ],
              ),
            ),
          ),

          if (row.prob != null)
            Padding(
              padding: const EdgeInsets.only(left: 12),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Text(
                    '${(row.prob! * 100).toStringAsFixed(2)}%',
                    style: Theme.of(context).textTheme.titleSmall,
                  ),
                  const Text('概率'),
                ],
              ),
            ),
        ],
      ),
    );
  }
}
