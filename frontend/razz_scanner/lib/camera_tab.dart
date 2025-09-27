import 'dart:io';
import 'dart:typed_data';
import 'dart:convert';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import 'package:url_launcher/url_launcher.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'cached_network_image.dart'; // If you use cached_network_image, import here
import 'main.dart'; // For _ResultBanner
import 'result_banner.dart';

String get matcherBaseUrl => dotenv.env['MATCHER_BASE_URL'] ?? '';

class CameraTab extends StatefulWidget {
  final CameraDescription camera;
  const CameraTab({super.key, required this.camera});
  @override
  State<CameraTab> createState() => _CameraTabState();
}

class _CameraTabState extends State<CameraTab> {
  CameraController? controller;
  bool busy = false;
  Map<String, dynamic>? lastResponse;
  String? errorMsg;

  @override
  void initState() {
    super.initState();
    try {
      controller = CameraController(
        widget.camera,
        ResolutionPreset.high,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.jpeg,
      );
      controller!.initialize().then((_) => setState(() {})).catchError((e) {
        setState(() => errorMsg = 'Camera error: $e');
      });
    } catch (e) {
      setState(() => errorMsg = 'Camera error: $e');
    }
  }

  @override
  void dispose() {
    controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (errorMsg != null) {
      return Center(child: Text(errorMsg!, style: const TextStyle(color: Colors.red)));
    }
    final initialized = controller?.value.isInitialized ?? false;
    return !initialized
        ? const Center(child: CircularProgressIndicator())
        : _buildCamera();
  }

  Widget _buildCamera() {
    final isReady = controller != null && controller!.value.isInitialized;
    return Column(
      children: [
        Expanded(
          child: isReady
              ? CameraPreview(controller!)
              : const Center(child: CircularProgressIndicator()),
        ),
        Padding(
          padding: const EdgeInsets.only(bottom: 24, top: 8),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              ElevatedButton.icon(
                icon: const Icon(Icons.camera_alt),
                label: Text(busy ? 'Working...' : 'Capture'),
                onPressed: (!isReady || busy) ? null : _captureAndSend,
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
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            child: ResultBanner(resp: lastResponse!),
          ),
      ],
    );
  }

  Future<void> _captureAndSend() async {
    setState(() => busy = true);
    try {
      debugPrint('matcherBaseUrl: $matcherBaseUrl');
      final file = await controller!.takePicture();
      final bytes = await File(file.path).readAsBytes();
      final resp = await _sendToMatcher(
        fileBytes: bytes,
        filename: 'full.jpg',
        contentType: MediaType('image', 'jpeg'),
      );
      setState(() => lastResponse = resp);
    } catch (e, stack) {
      debugPrint('Match error: $e');
      debugPrint('Stack trace: $stack');
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
      ..fields['strategy'] = 'auto'
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
