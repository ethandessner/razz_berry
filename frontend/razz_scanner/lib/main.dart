import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import 'package:cached_network_image/cached_network_image.dart';
import 'package:url_launcher/url_launcher.dart';
import 'dart:ui' show FontFeature;

/// CHANGE THIS to your laptop's LAN IP/port where FastAPI runs
const String matcherBaseUrl = 'http://192.168.1.69:8000';

// Ensures GCS links are fetchable by the phone (viewer → public CDN)
String _gcsToDownloadUrl(String url) {
  const viewer = 'https://storage.cloud.google.com/';
  if (url.startsWith(viewer)) {
    // viewer format: storage.cloud.google.com/<bucket>/<object>
    final rest = url.substring(viewer.length); // "<bucket>/<object>"
    return 'https://storage.googleapis.com/$rest';
  }
  return url;
}

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final cameras = await availableCameras();
  final back = cameras.firstWhere(
    (c) => c.lensDirection == CameraLensDirection.back,
    orElse: () => cameras.first,
  );
  runApp(App(camera: back));
}

class App extends StatelessWidget {
  final CameraDescription camera;
  const App({super.key, required this.camera});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Razz Berry',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFFF08828)),
        useMaterial3: true,
      ),
      home: Home(camera: camera),
    );
  }
}

class Home extends StatefulWidget {
  final CameraDescription camera;
  const Home({super.key, required this.camera});
  @override
  State<Home> createState() => _HomeState();
}

class _HomeState extends State<Home> {
  CameraController? controller;
  bool busy = false;
  Map<String, dynamic>? lastResponse;

  @override
  void initState() {
    super.initState();
    controller = CameraController(
      widget.camera,
      ResolutionPreset.high,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.jpeg,
    );
    controller!.initialize().then((_) => setState(() {}));
  }

  @override
  void dispose() {
    controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final initialized = controller?.value.isInitialized ?? false;
    return Scaffold(
      appBar: AppBar(title: const Text('Razz Scanner')),
      body: !initialized ? const Center(child: CircularProgressIndicator()) : _buildCamera(),
    );
  }

  Widget _buildCamera() {
    return Stack(
      fit: StackFit.expand,
      children: [
        CameraPreview(controller!),
        Positioned(
          left: 0,
          right: 0,
          bottom: 24,
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              ElevatedButton.icon(
                icon: const Icon(Icons.camera_alt),
                label: Text(busy ? 'Working...' : 'Capture'),
                onPressed: busy ? null : _captureAndSend,
              ),
              const SizedBox(width: 12),
              OutlinedButton.icon(
                icon: const Icon(Icons.monitor),
                label: const Text('Monitor'),
                onPressed: () => _openMonitor(context),
              ),
            ],
          ),
        ),
        if (lastResponse != null)
          Positioned(
            left: 12,
            right: 12,
            top: 12,
            child: _ResultBanner(resp: lastResponse!),
          ),
      ],
    );
  }

  Future<void> _captureAndSend() async {
    setState(() => busy = true);
    try {
      final file = await controller!.takePicture();
      final bytes = await File(file.path).readAsBytes();
      final resp = await _sendToMatcher(
        fileBytes: bytes,
        filename: 'full.jpg',
        contentType: MediaType('image', 'jpeg'),
      );
      setState(() => lastResponse = resp);
    } catch (e) {
      _toast('Match error: $e');
    } finally {
      if (mounted) setState(() => busy = false);
    }
  }

  Future<void> _openMonitor(BuildContext context) async {
    final uri = Uri.parse('$matcherBaseUrl/monitor');
    if (!await launchUrl(uri, mode: LaunchMode.externalApplication)) {
      _toast('Could not open monitor page.');
    }
  }

  Future<Map<String, dynamic>> _sendToMatcher({
    required Uint8List fileBytes,
    required String filename,
    required MediaType contentType,
  }) async {
    final uri = Uri.parse('$matcherBaseUrl/match');
    final req = http.MultipartRequest('POST', uri)
      ..fields['strategy'] = 'auto' // backend will auto-segment; extra fields are ignored if unused
      ..fields['top_k'] = '5'
      ..fields['cutoff'] = '18'
      ..files.add(http.MultipartFile.fromBytes(
        'file',
        fileBytes,
        filename: filename,
        contentType: contentType,
      ));

    final streamed = await req.send();
    final resp = await http.Response.fromStream(streamed);
    if (resp.statusCode != 200) {
      throw Exception('Server ${resp.statusCode}: ${resp.body}');
    }
    return jsonDecode(resp.body) as Map<String, dynamic>;
  }

  void _toast(String msg) {
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));
  }
}

class _ResultBanner extends StatelessWidget {
  final Map<String, dynamic> resp;
  const _ResultBanner({required this.resp});

  @override
  Widget build(BuildContext context) {
    final bool isConfident = resp['is_confident'] == true; // from backend
    final double conf = (resp['confidence'] as num?)?.toDouble() ?? 0.0;
    final best = resp['best'] as Map<String, dynamic>?;
    final top = (resp['top'] as List?)?.cast<Map<String, dynamic>>() ?? const [];

    final String bestUrl = _gcsToDownloadUrl(
      (best?['signed_image_url'] ?? best?['image_path'] ?? '') as String,
    );

    return Card(
      elevation: 6,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
      color: Theme.of(context).colorScheme.surface,
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              isConfident ? 'Match ✅  ${(conf * 100).toStringAsFixed(0)}%' : 'Best guess',
              style: Theme.of(context).textTheme.titleMedium,
            ),
            if (best != null) ...[
              const SizedBox(height: 10),
              Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  if (bestUrl.isNotEmpty) _Thumb(url: bestUrl, size: 72),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Text(
                      '${best['name']} • ${best['set_name']} • ${best['ext_number']} • ${best['subtype_name']}',
                      maxLines: 3,
                      overflow: TextOverflow.ellipsis,
                      style: Theme.of(context).textTheme.titleMedium,
                    ),
                  ),
                ],
              ),
            ],
            if (top.isNotEmpty) ...[
              const SizedBox(height: 12),
              Text('Top candidates', style: Theme.of(context).textTheme.labelLarge),
              const SizedBox(height: 6),
              SizedBox(
                height: 120,
                child: ListView.separated(
                  scrollDirection: Axis.horizontal,
                  itemCount: top.length,
                  separatorBuilder: (_, __) => const SizedBox(width: 10),
                  itemBuilder: (_, i) {
                    final t = top[i];
                    final url = _gcsToDownloadUrl(
                      (t['signed_image_url'] ?? t['image_path'] ?? '') as String,
                    );
                    return Column(
                      children: [
                        _Thumb(url: url, size: 84),
                        const SizedBox(height: 4),
                        Text(
                          'score ${t['score']}',
                          style: const TextStyle(
                            fontFeatures: [FontFeature.tabularFigures()],
                          ),
                        ),
                      ],
                    );
                  },
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

class _Thumb extends StatelessWidget {
  final String url;
  final double size;
  const _Thumb({required this.url, this.size = 72});

  @override
  Widget build(BuildContext context) {
    if (url.isEmpty) {
      return SizedBox(
        width: size,
        height: size,
        child: const DecoratedBox(
          decoration: BoxDecoration(color: Colors.black12),
        ),
      );
    }
    return SizedBox(
      width: size,
      height: size,
      child: ClipRRect(
        borderRadius: BorderRadius.circular(8),
        child: CachedNetworkImage(
          imageUrl: url,
          fit: BoxFit.cover,
          placeholder: (_, __) => const ColoredBox(color: Colors.black12),
          errorWidget: (_, __, ___) => const ColoredBox(color: Colors.black12),
        ),
      ),
    );
  }
}
