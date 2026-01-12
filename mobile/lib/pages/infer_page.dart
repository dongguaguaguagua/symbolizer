import 'package:flutter/material.dart';
import 'package:mobile/models/symbol_models.dart';
import 'package:mobile/services/mappings_service.dart';
import 'package:mobile/services/onnx_service.dart';
import 'package:mobile/widgets/candidate_list.dart';
import 'package:mobile/widgets/draw_pad.dart';
import 'package:mobile/models/candidate_row.dart';

class InferPage extends StatefulWidget {
  const InferPage({super.key});

  @override
  State<InferPage> createState() => _InferPageState();
}

class _InferPageState extends State<InferPage> {
  Map<String, MappingItem> _mappings = {};
  bool _bootLoading = true;
  bool _inferLoading = false;

  List<TopCandidate> _top5 = const [];
  int _clearSignal = 0;

  @override
  void initState() {
    super.initState();
    _boot();
  }

  Future<void> _boot() async {
    setState(() => _bootLoading = true);
    try {
      _mappings = await MappingsService.loadMappings();
    } catch (e) {
      // ignore; UI 会展示空
    } finally {
      if (mounted) setState(() => _bootLoading = false);
    }
  }

  Future<void> _onGrayChanged(List<int> gray32) async {
  	if (_inferLoading) return; // 防并发
    setState(() => _inferLoading = true);
    try {
      final res = await OnnxService.inferTop5(gray32);
      if (!mounted) return;
      setState(() => _top5 = res);
    } catch (e, st) {
      debugPrint("Infer error: $e");
      debugPrintStack(stackTrace: st);
    } finally {
      if (mounted) setState(() => _inferLoading = false);
    }
  }

  void _clear() {
    setState(() {
      _top5 = const [];
      _clearSignal++;
    });
  }

  @override
  Widget build(BuildContext context) {
    final status = _bootLoading
        ? '加载中...'
        : (_inferLoading ? '正在推理…' : '就绪');

    final rows = _top5.map((c) {

      final label = (c.index).toString();

      final item = _mappings[label] ??
          const MappingItem(symbol: '?', unicode: r'\u{003F}');

      return CandidateRow(
        label: label,
        item: item,
        prob: c.prob,
      );
    }).toList(growable: false);

    return Padding(
      padding: const EdgeInsets.all(12),
      child: Column(
        children: [
	        Row(
	          children: [
	            Text('识别', style: Theme.of(context).textTheme.headlineSmall),
	            const Spacer(),
	            Text(status, overflow: TextOverflow.ellipsis),
	          ],
	        ),
          const SizedBox(height: 12),
          DrawPad(
            onChanged: _onGrayChanged,
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
            ],
          ),
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
              child: _top5.isEmpty
                  ? const SizedBox.shrink()
                  : SingleChildScrollView(
                      key: const ValueKey('results'),
                      child: CandidateList(
                        title: '候选字符',
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
