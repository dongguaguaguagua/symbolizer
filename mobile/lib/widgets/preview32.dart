import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';

class Preview32 extends StatelessWidget {
  final List<int>? gray32;

  const Preview32({super.key, required this.gray32});

  @override
  Widget build(BuildContext context) {
    if (gray32 == null || gray32!.length != 1024) {
      return _emptyBox(context);
    }

    // 灰度 -> RGBA
    final rgba = Uint8List(32 * 32 * 4);
    for (var i = 0; i < 1024; i++) {
      final v = gray32![i];
      final j = i * 4;
      rgba[j] = v;       // R
      rgba[j + 1] = v;   // G
      rgba[j + 2] = v;   // B
      rgba[j + 3] = 255; // A
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Text(
        //   '即将提交的 32×32 预览',
        //   style: Theme.of(context).textTheme.labelLarge,
        // ),
        const SizedBox(height: 6),
        Container(
          width: 64,
          height: 64,
          decoration: BoxDecoration(
            border: Border.all(
              color: Theme.of(context).colorScheme.outlineVariant,
            ),
            borderRadius: BorderRadius.circular(8),
            color: Colors.white,
          ),
          child: _RawPreviewImage(rgba: rgba),
        ),
      ],
    );
  }

  Widget _emptyBox(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const SizedBox(height: 6),
        Container(
          width: 64,
          height: 64,
          decoration: BoxDecoration(
            border: Border.all(color: Colors.grey.shade400),
            borderRadius: BorderRadius.circular(8),
            color: Colors.white,
          ),
          child: const Center(
            child: Text(
              '—',
              style: TextStyle(color: Colors.grey),
            ),
          ),
        ),
      ],
    );
  }
}

class _RawPreviewImage extends StatefulWidget {
  final Uint8List rgba;

  const _RawPreviewImage({required this.rgba});

  @override
  State<_RawPreviewImage> createState() => _RawPreviewImageState();
}

class _RawPreviewImageState extends State<_RawPreviewImage> {
  ui.Image? _image;

  @override
  void initState() {
    super.initState();
    _decode();
  }

  @override
  void didUpdateWidget(covariant _RawPreviewImage oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.rgba != widget.rgba) {
      _decode();
    }
  }

  Future<void> _decode() async {
    ui.decodeImageFromPixels(
      widget.rgba, 32, 32,
      ui.PixelFormat.rgba8888,
      (img) {
        if (mounted) {
          setState(() => _image = img);
        }
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    if (_image == null) {
      return const Center(child: CircularProgressIndicator(strokeWidth: 2));
    }

    return RawImage(
      image: _image,
      filterQuality: FilterQuality.none,
      fit: BoxFit.contain,
    );
  }
}
