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
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'dart:ui' show FontFeature;
import 'home_tab.dart';
import 'camera_tab.dart';
import 'camera_alt.dart';
import 'collection_tab.dart';
import 'user_tab.dart';

final String matcherBaseUrl = dotenv.env['MATCHER_BASE_URL'] ?? '';

// Ensures GCS links are fetchable by the phone
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
  try {
    await dotenv.load();
  } catch (e) {
    // If .env file is missing, continue with defaults
    debugPrint('Warning: .env file not found or could not be loaded. Using default environment values.');
  }
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
      home: MainHome(camera: camera),
    );
  }
}

class MainHome extends StatefulWidget {
  final CameraDescription camera;
  const MainHome({super.key, required this.camera});
  @override
  State<MainHome> createState() => _MainHomeState();
}

class _MainHomeState extends State<MainHome> {
  int _selectedIndex = 0; // Start on Home tab by default

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: IndexedStack(
        index: _selectedIndex,
        children: [
          const HomeTab(),
          CameraTab(camera: widget.camera),
          const CollectionTab(),
          const UserTab(),
        ],
      ),
      bottomNavigationBar: BottomNavigationBar(
        type: BottomNavigationBarType.fixed,
        currentIndex: _selectedIndex,
        onTap: (index) => setState(() => _selectedIndex = index),
        selectedItemColor: const Color(0xFFF08828),
        unselectedItemColor: Colors.grey,
        items: const [
          BottomNavigationBarItem(
            icon: Icon(Icons.home),
            label: 'Home',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.camera_alt),
            label: 'Scan',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.collections),
            label: 'Collection',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.person),
            label: 'User',
          ),
        ],
      ),
    );
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
