import 'package:flutter/material.dart';
import 'cached_network_image.dart';
import 'dart:ui' show FontFeature;

String _gcsToDownloadUrl(String url) {
  const viewer = 'https://storage.cloud.google.com/';
  if (url.startsWith(viewer)) {
    final rest = url.substring(viewer.length);
    return 'https://storage.googleapis.com/$rest';
  }
  return url;
}

class ResultBanner extends StatelessWidget {
  final Map<String, dynamic> resp;
  const ResultBanner({required this.resp});

  @override
  Widget build(BuildContext context) {
    final bool isConfident = resp['is_confident'] == true;
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
                  if (bestUrl.isNotEmpty) Thumb(url: bestUrl, size: 72),
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
                        Thumb(url: url, size: 84),
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

class Thumb extends StatelessWidget {
  final String url;
  final double size;
  const Thumb({required this.url, this.size = 72});

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
