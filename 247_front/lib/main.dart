import 'package:flutter/material.dart';
import 'screens/learning_screen.dart';
import 'package:flutter_sound/flutter_sound.dart';
import 'package:permission_handler/permission_handler.dart';
import 'dart:io';
import 'package:path_provider/path_provider.dart';
import 'api_server.dart';
import 'models/learning_material.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Learning App',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: const HomeScreen(),
    );
  }
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  _HomeScreenState createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  FlutterSoundRecorder? _recorder;
  bool _isRecorderInitialized = false;
  bool _isRecording = false;
  String? _filePath;

  @override
  void initState() {
    super.initState();
    _recorder = FlutterSoundRecorder();
    _initializeRecorder();
  }

  Future<void> _initializeRecorder() async {
    var status = await Permission.microphone.request();
    if (status != PermissionStatus.granted) {
      throw RecordingPermissionException('Microphone permission not granted');
    }

    await _recorder!.openRecorder();
    _isRecorderInitialized = true;

    Directory tempDir = await getApplicationDocumentsDirectory();
    _filePath = '${tempDir.path}/audio_example.wav';

    setState(() {});
  }

  Future<void> _startRecording() async {
    if (!_isRecorderInitialized) return;
    await _recorder!.startRecorder(
      toFile: _filePath,
      codec: Codec.pcm16WAV,
    );
    setState(() {
      _isRecording = true;
    });
    print('녹음 시작: $_filePath');
  }

  Future<void> _stopRecording() async {
    if (!_isRecorderInitialized) return;
    await _recorder!.stopRecorder();
    setState(() {
      _isRecording = false;
    });
    print('녹음 중지 및 파일 저장: $_filePath');

    if (_filePath != null) {
      try {
        LearningMaterial? learningMaterial = await uploadAudioFile(File(_filePath!));
        if (learningMaterial != null) {
          Navigator.push(
            context,
            MaterialPageRoute(
              builder: (context) => LearningScreen(learningMaterial: learningMaterial),
            ),
          );
        } else {
          throw Exception('학습 자료를 가져오는데 실패했습니다.');
        }
      } catch (e) {
        print('Error: $e');
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('오류 발생: $e')),
        );
      }
    }
  }

  @override
  void dispose() {
    _recorder!.closeRecorder();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Home Screen'),
        backgroundColor: Colors.blueAccent,
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(
              _isRecording ? '녹음 중...' : '대기 중...',
              style: TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.bold,
                color: _isRecording ? Colors.red : Colors.green,
              ),
            ),
            const SizedBox(height: 30),
            ElevatedButton(
              onPressed: _isRecording ? null : _startRecording,
              child: const Text('Start Recording'),
            ),
            ElevatedButton(
              onPressed: !_isRecording ? null : _stopRecording,
              child: const Text('Stop Recording'),
            ),
          ],
        ),
      ),
    );
  }
}