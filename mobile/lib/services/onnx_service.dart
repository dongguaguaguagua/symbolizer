import 'dart:math';
import 'dart:typed_data';

import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:mobile/models/symbol_models.dart';

class OnnxService {
	static List<double> _softmax(List<double> xs) {
    if (xs.isEmpty) return const [];
    var maxV = xs[0];
    for (final v in xs) {
      if (v > maxV) maxV = v;
    }

    final exps = List<double>.filled(xs.length, 0);
    var sum = 0.0;
    for (var i = 0; i < xs.length; i++) {
      final e = exp(xs[i] - maxV);
      exps[i] = e;
      sum += e;
    }

    if (!sum.isFinite || sum <= 0) {
      final u = 1.0 / xs.length;
      return List<double>.filled(xs.length, u);
    }

    for (var i = 0; i < exps.length; i++) {
      exps[i] /= sum;
    }
    return exps;
	}
  static final OnnxRuntime _ort = OnnxRuntime();

  static Future<OrtSession>? _sessionFuture;

  static Future<OrtSession> _getSession() {
    final existing = _sessionFuture;
    if (existing != null) return existing;

    final fut = _ort.createSessionFromAsset(
    	// 'assets/model/model_int8.onnx'
     	// 'assets/model/residualcnn_int8.onnx'
      	'assets/model/residualcnn_augment_int8.onnx'
    );
    _sessionFuture = fut;
    return fut;
  }

  static bool _running = false;

  static Future<List<TopCandidate>> inferTop5(List<int> gray32) async {
    if (_running) {
      return const [];
    }
    _running = true;
    try {
      final sess = await _getSession();

      final input = Float32List(3 * 32 * 32);
      for (var i = 0; i < 1024; i++) {
        final v = gray32[i] / 255.0;
        input[i] = v;
        input[i + 1024] = v;
        input[i + 2048] = v;
      }

      final inputName = sess.inputNames.first;
      final outputName = sess.outputNames.first;

      final inputOrt = await OrtValue.fromList(input, const [1, 3, 32, 32]);

      final outputs = await sess.run({inputName: inputOrt});
      await inputOrt.dispose();

      OrtValue? outOrt = outputs[outputName] ?? (outputs.isNotEmpty ? outputs.values.first : null);
      if (outOrt == null) {
        // 没输出就返回空，避免后续 native 访问异常
        for (final v in outputs.values) {
          await v.dispose();
        }
        return const [];
      }

      final raw = await outOrt.asFlattenedList();

      for (final v in outputs.values) {
        await v.dispose();
      }

      final logits = raw.map((e) => (e as num).toDouble()).toList(growable: false);

      final probs = _softmax(logits);
      final pairs = List.generate(
        probs.length,
        (i) => TopCandidate(index: i, prob: probs[i]),
      )..sort((a, b) => b.prob.compareTo(a.prob));

      return pairs.take(5).toList(growable: false);
    } finally {
      _running = false;
    }
  }
}
