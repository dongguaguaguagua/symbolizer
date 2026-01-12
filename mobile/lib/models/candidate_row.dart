import 'package:mobile/models/symbol_models.dart';

class CandidateRow {
  final String label;
  final MappingItem item;
  final double? prob;

  const CandidateRow({
    required this.label,
    required this.item,
    this.prob,
  });
}
