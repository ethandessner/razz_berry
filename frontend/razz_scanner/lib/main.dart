import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:crop_your_image/crop_your_image.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import 'package:cached_network_image/cached_network_image.dart';
import 'dart:ui' show FontFeature;


/// CHANGE THIS to your laptop's LAN IP/port where FastAPI runs
const String matcherBaseUrl = 'http://192.168.1.69:8000';

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
  bool autoDetect = false; // toggle: server segments vs manual ROI
  Uint8List? capturedBytes;
  final cropController = CropController();

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
      appBar: AppBar(
        title: const Text('Razz Scanner'),
        actions: [
          Row(
            children: [
              const Text('Auto detect'),
              Switch(
                value: autoDetect,
                onChanged: (v) => setState(() => autoDetect = v),
              ),
              const SizedBox(width: 8),
            ],
          ),
        ],
      ),
      body: !initialized
          ? const Center(child: CircularProgressIndicator())
          : capturedBytes == null
              ? _buildCamera()
              : (autoDetect ? _buildAutoSend() : _buildCropper()),
    );
  }

  Widget _buildCamera() {
    return Stack(
      fit: StackFit.expand,
      children: [
        CameraPreview(controller!),
        Positioned(
          left: 0, right: 0, bottom: 24,
          child: Center(
            child: ElevatedButton.icon(
              icon: const Icon(Icons.camera_alt),
              label: Text(busy ? 'Working...' : 'Capture'),
              onPressed: busy ? null : _capture,
            ),
          ),
        ),
        if (lastResponse != null)
          Positioned(
            left: 12, right: 12, top: 12,
            child: _ResultBanner(resp: lastResponse!),
          ),
      ],
    );
  }

  Future<void> _capture() async {
    setState(() => busy = true);
    try {
      final file = await controller!.takePicture();
      final bytes = await File(file.path).readAsBytes();
      setState(() => capturedBytes = bytes);
    } catch (e) {
      _toast('Capture failed: $e');
    } finally {
      setState(() => busy = false);
    }
  }

  // AUTO: send full capture to server for segmentation
  Widget _buildAutoSend() {
    scheduleMicrotask(() async {
      if (!busy && capturedBytes != null) {
        setState(() => busy = true);
        try {
          final resp = await _sendToMatcher(
            fileBytes: capturedBytes!,
            filename: 'full.jpg',
            contentType: MediaType('image', 'jpeg'),
            strategy: 'auto',
          );
          setState(() => lastResponse = resp);
        } catch (e) {
          _toast('Match error: $e');
        } finally {
          if (mounted) {
            setState(() {
              capturedBytes = null; // back to camera
              busy = false;
            });
          }
        }
      }
    });
    return const Center(child: CircularProgressIndicator());
  }

  // ROI: user crops, then we send the crop to server
  Widget _buildCropper() {
    return Column(
      children: [
        Expanded(
          child: Crop(
            image: capturedBytes!,
            controller: cropController,
            baseColor: Colors.black,
            maskColor: Colors.black.withOpacity(0.6),
            onCropped: (cropped) async {
              setState(() => busy = true);
              try {
                final resp = await _sendToMatcher(
                  fileBytes: cropped,
                  filename: 'roi.png',
                  contentType: MediaType('image', 'png'),
                  strategy: 'roi',
                );
                setState(() => lastResponse = resp);
              } catch (e) {
                _toast('Match error: $e');
              } finally {
                if (mounted) {
                  setState(() {
                    capturedBytes = null; // back to camera
                    busy = false;
                  });
                }
              }
            },
            interactive: true,
            initialSize: 0.85,
            withCircleUi: false,
            cornerDotBuilder: (size, edge) => Container(
              width: size, height: size,
              decoration: const BoxDecoration(
                color: Colors.redAccent, shape: BoxShape.circle),
            ),
          ),
        ),
        SafeArea(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            child: Row(
              children: [
                TextButton(
                  onPressed: busy ? null : () => setState(() => capturedBytes = null),
                  child: const Text('Retake'),
                ),
                const Spacer(),
                ElevatedButton.icon(
                  onPressed: busy ? null : () => cropController.crop(),
                  icon: const Icon(Icons.search),
                  label: const Text('Scan'),
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }

  Future<Map<String, dynamic>> _sendToMatcher({
    required Uint8List fileBytes,
    required String filename,
    required MediaType contentType,
    required String strategy, // 'roi' | 'auto'
  }) async {
    final uri = Uri.parse('$matcherBaseUrl/match');
    final req = http.MultipartRequest('POST', uri)
      ..fields['strategy'] = strategy
      ..fields['max_side'] = '900'
      ..fields['top_k'] = '5'
      ..fields['cutoff'] = '18'
      ..files.add(http.MultipartFile.fromBytes(
        'file', fileBytes,
        filename: filename, contentType: contentType,
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
    final best = resp['best'] as Map<String, dynamic>?;
    final match = resp['match'] as Map<String, dynamic>?;
    final top = (resp['top'] as List?)?.cast<Map<String, dynamic>>() ?? const [];

    return Card(
      elevation: 3,
      child: Padding(
        padding: const EdgeInsets.all(10),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(match != null ? 'Match ✅ (score ${match['score']})' : 'No confident match',
                style: Theme.of(context).textTheme.titleMedium),
            if (best != null) ...[
              const SizedBox(height: 6),
              Row(
                children: [
                  _Thumb(url: best['image_path'] ?? ''),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Text(
                      '${best['name']} • ${best['set_name']} • ${best['ext_number']} • ${best['subtype_name']}',
                      maxLines: 2, overflow: TextOverflow.ellipsis,
                    ),
                  ),
                ],
              ),
            ],
            if (top.isNotEmpty) ...[
              const SizedBox(height: 10),
              Text('Top candidates', style: Theme.of(context).textTheme.labelLarge),
              const SizedBox(height: 6),
              SizedBox(
                height: 110,
                child: ListView.separated(
                  scrollDirection: Axis.horizontal,
                  itemCount: top.length,
                  separatorBuilder: (_, __) => const SizedBox(width: 8),
                  itemBuilder: (ctx, i) {
                    final t = top[i];
                    return Column(
                      children: [
                        _Thumb(url: t['image_path'] ?? '', size: 80),
                        const SizedBox(height: 4),
                        Text(
                          '${t['score']}',
                          style: TextStyle(
                            fontFeatures: [FontFeature.tabularFigures()],
                          ),
                        ),
                      ],
                    );
                  },
                ),
              ),
            ]
          ],
        ),
      ),
    );
  }
}

class _Thumb extends StatelessWidget {
  final String url;
  final double size;
  const _Thumb({required this.url, this.size = 64});
  @override
  Widget build(BuildContext context) {
    if (url.isEmpty) {
      return Container(width: size, height: size, color: Colors.black12);
    }
    return ClipRRect(
      borderRadius: BorderRadius.circular(8),
      child: CachedNetworkImage(
        imageUrl: url, width: size, height: size, fit: BoxFit.cover,
        placeholder: (_, __) => Container(width: size, height: size, color: Colors.black12),
        errorWidget: (_, __, ___) => Container(width: size, height: size, color: Colors.black12),
      ),
    );
  }
}
