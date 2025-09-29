import 'dart:io';
import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';

// Alternative camera UI inspired by Rare Candy style.
// Features:
//  - Framed card scan area with neon corner guides
//  - Single central capture button (no tap-anywhere)
//  - Draggable bottom sheet showing best match + expandable list of candidates
//  - Reuses existing /match endpoint
//  - Non-destructive: can coexist with existing CameraTab
//
// To use: push CameraAltScreen() from navigation or conditionally swap tabs.

String get matcherBaseUrl => dotenv.env['MATCHER_BASE_URL'] ?? '';

class CameraAltScreen extends StatefulWidget {
  final CameraDescription camera;
  const CameraAltScreen({super.key, required this.camera});

  @override
  State<CameraAltScreen> createState() => _CameraAltScreenState();
}

class _CameraAltScreenState extends State<CameraAltScreen> with TickerProviderStateMixin {
  CameraController? _controller;
  bool _capturing = false;
  Map<String, dynamic>? _response;
  String? _error;

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  Future<void> _initCamera() async {
    try {
      _controller = CameraController(
        widget.camera,
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.jpeg,
      );
      await _controller!.initialize();
      if (mounted) setState(() {});
    } catch (e) {
      setState(() => _error = 'Camera init failed: $e');
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final ready = _controller?.value.isInitialized ?? false;
    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        children: [
          if (ready)
            Positioned.fill(
              child: LayoutBuilder(
                builder: (context, constraints) {
                  final previewAspect = _controller!.value.aspectRatio; // width/height
                  final screenAspect = constraints.maxWidth / constraints.maxHeight;
                  // Scale so that shorter dimension covers; prevents letterboxing.
                  double scale;
                  if (screenAspect > previewAspect) {
                    // screen is wider than camera => scale height
                    scale = screenAspect / previewAspect;
                  } else {
                    // screen is taller than camera => scale width
                    scale = previewAspect / screenAspect;
                  }
                  return FittedBox(
                    fit: BoxFit.cover,
                    child: Transform.scale(
                      scale: scale,
                      child: SizedBox(
                        width: constraints.maxWidth,
                        height: constraints.maxHeight,
                        child: CameraPreview(_controller!),
                      ),
                    ),
                  );
                },
              ),
            )
          else
            const Center(child: CircularProgressIndicator()),
          // Dark overlay + cutout frame
          if (ready)
            Positioned.fill(
              child: CustomPaint(
                painter: _FrameOverlayPainter(),
              ),
            ),
          // Top bar (close / title placeholder)
          SafeArea(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
              child: Row(
                children: [
                  IconButton(
                    icon: const Icon(Icons.close, color: Colors.white),
                    onPressed: () => Navigator.of(context).maybePop(),
                  ),
                  const SizedBox(width: 8),
                  const Text('Scan', style: TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.w600)),
                  const Spacer(),
                  if (_capturing) const SizedBox(width: 24, height: 24, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white)),
                ],
              ),
            ),
          ),
          // Capture button centered horizontally near bottom (above sheet)
          Positioned(
            left: 0,
            right: 0,
            bottom: 180, // leave room for collapsed sheet
            child: Center(
              child: _CaptureButton(
                capturing: _capturing,
                onTap: ready && !_capturing ? _doCapture : null,
              ),
            ),
          ),
          // Bottom sheet for results
          _ResultSheet(
            response: _response,
            onClear: () => setState(() => _response = null),
          ),
        ],
      ),
    );
  }

  Future<void> _doCapture() async {
    if (!(_controller?.value.isInitialized ?? false)) return;
    setState(() => _capturing = true);
    try {
      final file = await _controller!.takePicture();
      final bytes = await File(file.path).readAsBytes();
      final resp = await _send(bytes);
      setState(() => _response = resp);
    } catch (e) {
      setState(() => _error = 'Capture failed: $e');
    } finally {
      if (mounted) setState(() => _capturing = false);
    }
  }

  Future<Map<String, dynamic>> _send(Uint8List fileBytes) async {
    final uri = Uri.parse('$matcherBaseUrl/match');
    final req = http.MultipartRequest('POST', uri)
      ..fields['strategy'] = 'auto'
      ..fields['top_k'] = '5'
      ..fields['cutoff'] = '18'
      ..files.add(http.MultipartFile.fromBytes(
        'file',
        fileBytes,
        filename: 'scan.jpg',
        contentType: MediaType('image', 'jpeg'),
      ));
    final streamed = await req.send();
    final resp = await http.Response.fromStream(streamed);
    if (resp.statusCode != 200) {
      throw Exception('Server ${resp.statusCode}: ${resp.body}');
    }
    return jsonDecode(resp.body) as Map<String, dynamic>;
  }
}

class _CaptureButton extends StatelessWidget {
  final bool capturing;
  final VoidCallback? onTap;
  const _CaptureButton({required this.capturing, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 180),
        width: 82,
        height: 82,
        decoration: BoxDecoration(
          shape: BoxShape.circle,
            gradient: capturing
                ? const SweepGradient(colors: [Colors.orange, Colors.deepOrange])
                : const LinearGradient(colors: [Color(0xffFFCB66), Color(0xffFF9100)]),
          boxShadow: [
            BoxShadow(color: Colors.black.withOpacity(0.4), blurRadius: 10, spreadRadius: 2),
          ],
        ),
        child: Center(
          child: capturing
              ? const Icon(Icons.hourglass_top, color: Colors.white)
              : const Icon(Icons.camera, color: Colors.white, size: 34),
        ),
      ),
    );
  }
}

class _ResultSheet extends StatefulWidget {
  final Map<String, dynamic>? response;
  final VoidCallback onClear;
  const _ResultSheet({required this.response, required this.onClear});

  @override
  State<_ResultSheet> createState() => _ResultSheetState();
}

class _ResultSheetState extends State<_ResultSheet> {
  final DraggableScrollableController _dragController = DraggableScrollableController();

  @override
  Widget build(BuildContext context) {
    return DraggableScrollableSheet(
      controller: _dragController,
      initialChildSize: 0.22,
      minChildSize: 0.18,
      maxChildSize: 0.78,
      snap: true,
      builder: (context, scroll) {
    final resp = widget.response;
    final rawTop = resp?['top'];
    final List<Map<String, dynamic>> top = (rawTop is List)
      ? rawTop
        .whereType<dynamic>()
        .where((e) => e is Map)
        .map((e) => (e as Map).cast<String, dynamic>())
        .toList()
      : <Map<String, dynamic>>[];
    final best = resp != null ? (resp['match'] ?? resp['best']) : null;
        return LayoutBuilder(
          builder: (context, constraints) {
            return Container(
              decoration: const BoxDecoration(
                color: Color(0xFF1E1E1F),
                borderRadius: BorderRadius.vertical(top: Radius.circular(28)),
              ),
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Center(
                    child: Container(
                      width: 42,
                      height: 5,
                      decoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.25),
                        borderRadius: BorderRadius.circular(3),
                      ),
                    ),
                  ),
                  const SizedBox(height: 10),
                  Row(
                    children: [
                      const Text('Results', style: TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.w600)),
                      const Spacer(),
                      if (resp != null)
                        IconButton(
                          icon: const Icon(Icons.clear, color: Colors.white70),
                          onPressed: widget.onClear,
                          tooltip: 'Clear results',
                        ),
                    ],
                  ),
                  if (best != null) _BestMatchCard(best: best),
                  const SizedBox(height: 4),
                  Expanded(
                    child: NotificationListener<OverscrollIndicatorNotification>(
                      onNotification: (o) { o.disallowIndicator(); return true; },
                      child: ListView.separated(
                        controller: scroll,
                        physics: const ClampingScrollPhysics(),
                        itemCount: top.length,
                        separatorBuilder: (_, __) => const Divider(color: Colors.white12, height: 1),
                        itemBuilder: (context, i) {
                          final item = top[i];
                          final sim = (item['similarity'] ?? 0.0) as num;
                          return ListTile(
                            dense: true,
                            contentPadding: EdgeInsets.zero,
                            title: Text(
                              item['name']?.toString() ?? 'Unknown',
                              style: const TextStyle(color: Colors.white),
                            ),
                            subtitle: Text(
                              'Similarity ${(sim * 100).toStringAsFixed(1)}%  •  ${item['set_name'] ?? ''}',
                              style: const TextStyle(color: Colors.white70, fontSize: 12),
                            ),
                            trailing: i == 0
                                ? const Icon(Icons.check_circle, color: Colors.lightGreenAccent)
                                : null,
                            onTap: () {},
                          );
                        },
                      ),
                    ),
                  ),
                ],
              ),
            );
          },
        );
      },
    );
  }
}

class _BestMatchCard extends StatelessWidget {
  final Map<String, dynamic> best;
  const _BestMatchCard({required this.best});

  @override
  Widget build(BuildContext context) {
    final sim = (best['similarity'] ?? 0.0) as num;
    return Container(
      width: double.infinity,
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(18),
        gradient: const LinearGradient(
          colors: [Color(0xFF2A2A2C), Color(0xFF303033)],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        border: Border.all(color: Colors.white12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(
                child: Text(
                  best['name']?.toString() ?? 'Best Match',
                  style: const TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.w600),
                ),
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                decoration: BoxDecoration(
                  color: Colors.green.withOpacity(0.15),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Text(
                  '${(sim * 100).toStringAsFixed(1)}%',
                  style: const TextStyle(color: Colors.lightGreenAccent, fontWeight: FontWeight.w600),
                ),
              ),
            ],
          ),
          const SizedBox(height: 6),
          Text(
            '${best['set_name'] ?? ''}  •  ${best['ext_number'] ?? ''}',
            style: const TextStyle(color: Colors.white60, fontSize: 12),
          ),
          const SizedBox(height: 8),
          LinearProgressIndicator(
            value: sim.clamp(0, 1).toDouble(),
            minHeight: 6,
            backgroundColor: Colors.white10,
            color: Colors.lightGreenAccent,
            borderRadius: BorderRadius.circular(4),
          ),
        ],
      ),
    );
  }
}

class _FrameOverlayPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint();

    // Define target card rect (centered, fixed aspect 3.5x2.5 ~ 1.4 ratio).
    final frameWidth = size.width * 0.74; // a bit narrower than screen
    final aspect = 3.5/2.5; // height / width
    final frameHeight = frameWidth * aspect;
    final top = size.height * 0.12; // push down from top
    final left = (size.width - frameWidth)/2;
    final cardRect = Rect.fromLTWH(left, top, frameWidth, frameHeight);

    // Dimmed backdrop
    paint.color = Colors.black.withOpacity(0.55);
    final backdrop = Path()..addRect(Rect.fromLTWH(0,0,size.width,size.height));
    final cutout = Path()..addRRect(RRect.fromRectAndRadius(cardRect, const Radius.circular(26)));
    final diff = Path.combine(PathOperation.difference, backdrop, cutout);
    canvas.drawPath(diff, paint);

    // Corner guides
    final cornerPaint = Paint()
      ..color = const Color(0xFFCBFB42)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 5
      ..strokeCap = StrokeCap.round;

    const cornerLen = 42.0;
    void drawCorner(Offset c, {required bool tl, required bool tr, required bool bl, required bool br}) {
      final path = Path();
      if (tl) {
        path.moveTo(c.dx, c.dy + cornerLen);
        path.lineTo(c.dx, c.dy);
        path.lineTo(c.dx + cornerLen, c.dy);
      }
      if (tr) {
        path.moveTo(c.dx - cornerLen, c.dy);
        path.lineTo(c.dx, c.dy);
        path.lineTo(c.dx, c.dy + cornerLen);
      }
      if (bl) {
        path.moveTo(c.dx, c.dy - cornerLen);
        path.lineTo(c.dx, c.dy);
        path.lineTo(c.dx + cornerLen, c.dy);
      }
      if (br) {
        path.moveTo(c.dx - cornerLen, c.dy);
        path.lineTo(c.dx, c.dy);
        path.lineTo(c.dx, c.dy - cornerLen);
      }
      canvas.drawPath(path, cornerPaint);
    }

    drawCorner(cardRect.topLeft, tl: true, tr: false, bl: false, br: false);
    drawCorner(cardRect.topRight, tl: false, tr: true, bl: false, br: false);
    drawCorner(cardRect.bottomLeft, tl: false, tr: false, bl: true, br: false);
    drawCorner(cardRect.bottomRight, tl: false, tr: false, bl: false, br: true);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}
