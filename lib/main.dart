import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:theme/theme.dart';

import 'platform/camera_channel.dart';
import 'screens/camera_screen.dart';
import 'settings/settings_store.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);
  final settings = await SettingsStore.load();
  runApp(KlickApp(settings: settings, camera: CameraChannel()));
}

class KlickApp extends StatelessWidget {
  const KlickApp({super.key, required this.settings, required this.camera});

  final SettingsStore settings;
  final CameraApi camera;

  @override
  Widget build(BuildContext context) {
    return UiApp(
      title: 'Klick',
      home: CameraScreen(camera: camera, settings: settings),
    );
  }
}
