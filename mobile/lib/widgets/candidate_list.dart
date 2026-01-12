import 'package:flutter/material.dart';
import 'package:mobile/models/candidate_row.dart';
import 'package:mobile/widgets/candidate_row_tile.dart';

class CandidateList extends StatelessWidget {
  final List<CandidateRow> rows;
  final String title;

  const CandidateList({
    super.key,
    required this.rows,
    required this.title,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 0,
      color: Theme.of(context).colorScheme.surfaceContainerHighest,
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(title, style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 8),
            for (final r in rows) ...[
              CandidateRowTile(row: r),
              const Divider(height: 16),
            ],
          ],
        ),
      ),
    );
  }
}
