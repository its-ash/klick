import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// Camera options, saved with `shared_preferences`.
class SettingsStore extends ChangeNotifier {
  SettingsStore._(this._prefs)
    : _keepDng = _prefs?.getBool(_kKeepDng) ?? true,
      _quality = _prefs?.getInt(_kQuality) ?? 95,
      _shutterSound = _prefs?.getBool(_kSound) ?? true,
      _grid = _prefs?.getBool(_kGrid) ?? false,
      _locationTag = _prefs?.getBool(_kLocation) ?? false,
      _autoMacro = _prefs?.getBool(_kAutoMacro) ?? false;

  /// For tests: keeps everything in memory.
  SettingsStore.memory()
    : _prefs = null,
      _keepDng = true,
      _quality = 95,
      _shutterSound = true,
      _grid = false,
      _locationTag = false,
      _autoMacro = false;

  static const _kKeepDng = 'keep_dng';
  static const _kQuality = 'jpeg_quality';
  static const _kSound = 'shutter_sound';
  static const _kGrid = 'grid';
  static const _kLocation = 'location_tag';
  static const _kAutoMacro = 'auto_macro';

  /// JPEG qualities offered in Settings.
  static const qualities = [85, 90, 95, 100];

  static Future<SettingsStore> load() async => SettingsStore._(await SharedPreferences.getInstance());

  final SharedPreferences? _prefs;
  bool _keepDng;
  int _quality;
  bool _shutterSound;
  bool _grid;
  bool _locationTag;
  bool _autoMacro;

  bool get keepDng => _keepDng;
  int get jpegQuality => _quality;
  bool get shutterSound => _shutterSound;
  bool get grid => _grid;
  bool get locationTag => _locationTag;

  /// Switch to the close-focusing lens automatically when the subject is too close for the main one.
  /// Off by default: the viewfinder's Normal | Macro switch is the main control.
  bool get autoMacro => _autoMacro;

  void setKeepDng(bool v) {
    _keepDng = v;
    _prefs?.setBool(_kKeepDng, v);
    notifyListeners();
  }

  void setJpegQuality(int v) {
    _quality = qualities.contains(v) ? v : 95;
    _prefs?.setInt(_kQuality, _quality);
    notifyListeners();
  }

  void setShutterSound(bool v) {
    _shutterSound = v;
    _prefs?.setBool(_kSound, v);
    notifyListeners();
  }

  void setGrid(bool v) {
    _grid = v;
    _prefs?.setBool(_kGrid, v);
    notifyListeners();
  }

  void setAutoMacro(bool v) {
    _autoMacro = v;
    _prefs?.setBool(_kAutoMacro, v);
    notifyListeners();
  }

  void setLocationTag(bool v) {
    _locationTag = v;
    _prefs?.setBool(_kLocation, v);
    notifyListeners();
  }
}
