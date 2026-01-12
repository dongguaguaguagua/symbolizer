import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:image/image.dart' as img;

typedef GrayChanged = void Function(List<int> gray32);

class DrawPad extends StatefulWidget {
  final GrayChanged onChanged;
  final int clearSignal;

  const DrawPad({
    super.key,
    required this.onChanged,
    required this.clearSignal,
  });

  @override
  State<DrawPad> createState() => _DrawPadState();
}

class _DrawPadState extends State<DrawPad> {
  static const double _padSize = 256;

  final GlobalKey _repaintKey = GlobalKey();
  List<Offset?> _points = <Offset?>[];
  bool _pending = false;

  @override
  void didUpdateWidget(covariant DrawPad oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.clearSignal != widget.clearSignal) {
      setState(() => _points = <Offset?>[]);
    }
  }

  Future<void> _emitGray() async {
    if (_pending) return;
    _pending = true;
    try {
      final ctx = _repaintKey.currentContext;
      if (ctx == null) return;

      final ro = ctx.findRenderObject();
      if (ro is! RenderRepaintBoundary) return;

      // 关键点 1：只 capture CustomPaint（不含外层 Container 的 border/radius）
      // 关键点 2：不要假设输出尺寸一定是 256×256；用 ui.Image 的真实 width/height
      final ui.Image image = await ro.toImage(pixelRatio: 1.0);

      final ByteData? byteData =
          await image.toByteData(format: ui.ImageByteFormat.rawRgba);
      if (byteData == null) return;

      final Uint8List rgba = byteData.buffer.asUint8List();

      final gray32 = _downsampleAutoCropRGBA(
        rgba: rgba,
        width: image.width,
        height: image.height,
      );

      widget.onChanged(gray32);
    } finally {
      _pending = false;
    }
  }

  void _addPoint(Offset p) {
    setState(() {
      _points = List<Offset?>.of(_points)..add(p); // 强制新 List
    });
  }

  void _endStroke() {
    setState(() {
      _points = List<Offset?>.of(_points)..add(null);
    });
  }

  @override
  Widget build(BuildContext context) {
    final outline = Theme.of(context).colorScheme.outlineVariant;

    return SizedBox(
      width: _padSize,
      height: _padSize,
      child: DecoratedBox(
        decoration: BoxDecoration(
          color: Colors.white,
          border: Border.all(color: outline),
          borderRadius: BorderRadius.circular(12),
        ),
        child: ClipRRect(
          borderRadius: BorderRadius.circular(12),
          child: Listener(
            behavior: HitTestBehavior.opaque,
            onPointerDown: (e) => _addPoint(e.localPosition),
            onPointerMove: (e) => _addPoint(e.localPosition),
            onPointerUp: (_) async {
              _endStroke();
              await _emitGray();
            },
            child: RepaintBoundary(
              // 注意：RepaintBoundary 在 CustomPaint 上，而不是在外层 Container 上
              key: _repaintKey,
              child: CustomPaint(
                size: const Size(_padSize, _padSize),
                painter: _StrokePainter(points: _points),
              ),
            ),
          ),
        ),
      ),
    );
  }
}

class _StrokePainter extends CustomPainter {
  final List<Offset?> points;

  _StrokePainter({required this.points});

  @override
  void paint(Canvas canvas, Size size) {
    // 纯白底（确保 bbox 扫描时背景一致）
    canvas.drawRect(
      Offset.zero & size,
      Paint()..color = Colors.white,
    );

    final p = Paint()
      ..color = Colors.black
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round
      ..strokeWidth = 12;

    for (var i = 0; i < points.length - 1; i++) {
      final a = points[i];
      final b = points[i + 1];
      if (a != null && b != null) {
        canvas.drawLine(a, b, p);
      }
    }
  }

  @override
  bool shouldRepaint(covariant _StrokePainter oldDelegate) {
    return !identical(oldDelegate.points, points);
  }
}

/// - 找“笔迹” bbox（同时忽略透明像素）
/// - bbox -> 正方形居中裁剪
/// - resize 到 32×32
/// - 输出 1024 灰度： (r+g+b)/3
List<int> _downsampleAutoCropRGBA({
  required Uint8List rgba,
  required int width,
  required int height,
}) {
  final src = img.Image.fromBytes(
    width: width,
    height: height,
    bytes: rgba.buffer,
    order: img.ChannelOrder.rgba,
  );

  int minX = width;
  int minY = height;
  int maxX = -1;
  int maxY = -1;
  bool hasInk = false;

  // 关键：透明像素不要当 ink（圆角、抗锯齿区域可能带 alpha）
  const int alphaThreshold = 10;

  // 非白判定：任一通道 < 250
  const int whiteThreshold = 250;

  for (var y = 0; y < height; y++) {
    for (var x = 0; x < width; x++) {
      final px = src.getPixel(x, y);
      final a = px.a;
      if (a < alphaThreshold) continue;

      final r = px.r;
      final g = px.g;
      final b = px.b;

      if (r < whiteThreshold || g < whiteThreshold || b < whiteThreshold) {
        hasInk = true;
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
      }
    }
  }

  if (!hasInk) {
    return List<int>.filled(1024, 0);
  }

  final boxW = maxX - minX + 1;
  final boxH = maxY - minY + 1;
  final size = boxW > boxH ? boxW : boxH;

  final cx = ((minX + maxX) / 2).floor();
  final cy = ((minY + maxY) / 2).floor();

  int sx = cx - (size / 2).floor();
  int sy = cy - (size / 2).floor();

  if (sx < 0) sx = 0;
  if (sy < 0) sy = 0;
  if (sx + size > width) sx = width - size;
  if (sy + size > height) sy = height - size;

  final cropped = img.copyCrop(
    src,
    x: sx,
    y: sy,
    width: size,
    height: size,
  );

  final resized = img.copyResize(
    cropped,
    width: 32,
    height: 32,
    interpolation: img.Interpolation.average,
  );

  final out = List<int>.filled(1024, 0);
  for (var y = 0; y < 32; y++) {
    for (var x = 0; x < 32; x++) {
      final px = resized.getPixel(x, y);
      out[y * 32 + x] = ((px.r + px.g + px.b) / 3).round();
    }
  }
  return out;
}
