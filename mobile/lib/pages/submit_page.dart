import 'package:flutter/material.dart';
import 'package:mobile/models/symbol_models.dart';
import 'package:mobile/services/mappings_service.dart';
import 'package:mobile/services/worker_api.dart';
import 'package:mobile/widgets/candidate_list.dart';
import 'package:mobile/widgets/draw_pad.dart';
import 'package:mobile/models/candidate_row.dart';
import 'package:mobile/widgets/preview32.dart';
import 'package:mobile/widgets/symbol_view.dart';

class SubmitPage extends StatefulWidget {
  const SubmitPage({super.key});

  @override
  State<SubmitPage> createState() => _SubmitPageState();
}

class _SubmitPageState extends State<SubmitPage> {
  static const String WORKER_BASE = 'https://symbol-collector.melonhu.cn';

  final WorkerApi _api = const WorkerApi(WORKER_BASE);

  Map<String, MappingItem> _mappings = {};
  bool _bootLoading = true;

  int _clearSignal = 0;
  List<int>? _gray32;

  Map<String, dynamic>? _target; // {label, symbol, unicode}
  String _status = '就绪';

  @override
  void initState() {
    super.initState();
    _boot();
  }

  Future<void> _boot() async {
    setState(() => _bootLoading = true);
    try {
      _mappings = await MappingsService.loadMappings();
      await _fetchRandom();
    } catch (e) {
      // ignore
    } finally {
      if (mounted) setState(() => _bootLoading = false);
    }
  }

  Future<void> _fetchRandom() async {
    final t = await _api.fetchRandomSymbol();
    if (!mounted) return;
    setState(() => _target = t);
  }

  void _clear() {
    setState(() {
      _gray32 = null;
      _clearSignal++;
      _status = '就绪';
    });
  }

  Future<void> _submit() async {
    final t = _target;
    final g = _gray32;
    if (t == null || g == null) return;

    setState(() => _status = '上传中…');
    try {
      await _api.submitSample(label: t['label'].toString(), gray32: g);
      if (!mounted) return;
      setState(() => _status = '提交成功！');
      _clear();
      await _fetchRandom();
    } catch (e) {
      if (!mounted) return;
      setState(() => _status = '提交失败：$e');
    }
  }

  @override
  Widget build(BuildContext context) {
    final t = _target;

    // 如果 worker 只返回 label，则回退到 mappings 查。
    MappingItem targetItem;
    String targetLabel = '?';

    if (t != null) {
      targetLabel = t['label']?.toString() ?? '?';
      final sym = t['symbol']?.toString();
      final uni = t['unicode']?.toString();
      final mapped = _mappings[targetLabel];

      if (sym != null && uni != null) {
        targetItem = MappingItem(
          symbol: sym,
          unicode: uni,
          svg: mapped?.svg,
        );
      } else {
        targetItem = mapped ??
            const MappingItem(
              symbol: '?',
              unicode: r'\u{003F}',
              svg: null,
            );
      }
    } else {
      targetItem = const MappingItem(symbol: '?', unicode: r'\u{003F}');
    }

    final rows = [
      CandidateRow(label: targetLabel, item: targetItem, prob: null),
    ];

    return Padding(
      padding: const EdgeInsets.all(12),
      child: Column(
        children: [
          Row(
            children: [
              Text('提交', style: Theme.of(context).textTheme.headlineSmall),
              const Spacer(),
              Text(_bootLoading ? '加载中…' : _status),
            ],
          ),
          const SizedBox(height: 12),
          DrawPad(
            onChanged: (g) => setState(() => _gray32 = g),
            clearSignal: _clearSignal,
          ),
          const SizedBox(height: 10),
          Row(
            children: [
              Expanded(
                child: FilledButton.tonal(
                  onPressed: _clear,
                  child: const Text('清空'),
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: FilledButton(
                  onPressed: (_gray32 == null || _target == null || _bootLoading)
                      ? null
                      : _submit,
                  child: const Text('提交'),
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          // 预览裁剪后的图像
          Preview32(gray32: _gray32),

          const SizedBox(height: 12),
          Expanded(
            child: AnimatedSwitcher(
              duration: const Duration(milliseconds: 200),
              transitionBuilder: (child, anim) {
                return FadeTransition(
                  opacity: anim,
                  child: SizeTransition(
                    sizeFactor: anim,
                    axisAlignment: -1,
                    child: child,
                  ),
                );
              },
              child: _target == null
                  ? const SizedBox.shrink()
                  : SingleChildScrollView(
                      key: ValueKey(_target!['label']),
                      child: CandidateList(
                        title: '目标符号',
                        rows: rows,
                      ),
                    ),
            ),
          ),
        ],
      ),
    );
  }
}
