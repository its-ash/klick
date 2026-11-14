import 'package:flutter/material.dart';
import 'package:theme/theme.dart';

import '../platform/camera_channel.dart';
import '../settings/settings_store.dart';

class SettingsScreen extends StatelessWidget {
  const SettingsScreen({super.key, required this.settings, required this.camera});

  final SettingsStore settings;
  final CameraApi camera;

  @override
  Widget build(BuildContext context) {
    return UiScaffold(
      appBar: const UiAppBar(title: 'Settings', back: true),
      scrollable: true,
      body: ListenableBuilder(
        listenable: settings,
        builder: (context, _) => Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            UiListGroup(
              header: 'Capture',
              children: [
                UiSwitchTile(
                  'Keep RAW (DNG)',
                  subtitle: 'Saves the reference frame next to each photo. Uses more storage.',
                  icon: Icons.raw_on_rounded,
                  value: settings.keepDng,
                  onChanged: settings.setKeepDng,
                ),
                UiSwitchTile(
                  'Shutter sound',
                  icon: Icons.volume_up_rounded,
                  value: settings.shutterSound,
                  onChanged: settings.setShutterSound,
                ),
                UiSwitchTile(
                  'Grid',
                  subtitle: 'Rule-of-thirds lines on the viewfinder',
                  icon: Icons.grid_3x3_rounded,
                  value: settings.grid,
                  onChanged: settings.setGrid,
                ),
                UiSwitchTile(
                  'Auto macro',
                  subtitle: 'Switches to the close-focusing lens when the subject is too close',
                  icon: Icons.local_florist_outlined,
                  value: settings.autoMacro,
                  onChanged: settings.setAutoMacro,
                ),
              ],
            ),
            const UiGap(UiSpace.lg),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: UiSpace.xs),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('JPEG quality', style: context.text.titleSmall),
                  const UiGap(UiSpace.sm),
                  UiSegmentedControl(
                    [for (final q in SettingsStore.qualities) UiSegment<int>(q, label: '$q')],
                    value: settings.jpegQuality,
                    expand: true,
                    onChanged: settings.setJpegQuality,
                  ),
                  const UiGap(UiSpace.xs),
                  Text('Higher is larger. 95 is the best balance.', style: context.text.bodySmall),
                ],
              ),
            ),
            const UiGap(UiSpace.lg),
            UiListGroup(
              header: 'Privacy',
              children: [
                UiSwitchTile(
                  'Location tagging',
                  subtitle: 'Adds GPS coordinates to photos. Off by default.',
                  icon: Icons.location_on_rounded,
                  value: settings.locationTag,
                  onChanged: (v) async {
                    settings.setLocationTag(v);
                    if (v) await camera.permission(location: true);
                  },
                ),
                const UiListTile(
                  'Everything stays on this phone',
                  subtitle: 'Photos are processed on the device. The internet is used only to check for app updates.',
                  icon: Icons.lock_rounded,
                  iconStatus: UiStatus.success,
                ),
              ],
            ),
            const UiGap(UiSpace.lg),
            const UiAppearanceSection(),
            const UiGap(UiSpace.lg),
            const UiUpdateSection(storeId: 'klick'),
            const UiGap(UiSpace.lg),
            const UiDeveloperSection(),
          ],
        ),
      ),
    );
  }
}
