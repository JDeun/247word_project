import 'package:http/http.dart' as http;
import 'dart:io';
import 'dart:convert';
import 'models/learning_material.dart';

Future<LearningMaterial?> uploadAudioFile(File audioFile) async {
  try {
    print('uploadAudioFile 시작');
    var uri = Uri.parse('http://172.30.1.45:8000/generate_material');
    print('서버 URI: $uri');
    var request = http.MultipartRequest('POST', uri)
      ..files.add(await http.MultipartFile.fromPath('file', audioFile.path));

    print('파일 경로: ${audioFile.path}');
    print('요청 전송 중...');
    var streamedResponse = await request.send();
    var response = await http.Response.fromStream(streamedResponse);
    print('응답 받음. 상태 코드: ${response.statusCode}');

    if (response.statusCode == 200) {
      // UTF-8 디코딩 적용
      String decodedBody = utf8.decode(response.bodyBytes);
      print('디코딩된 응답 본문: $decodedBody');
      var jsonData = jsonDecode(decodedBody);
      return LearningMaterial.fromJson(jsonData);
    } else {
      print("오디오 업로드 실패: ${response.statusCode}");
      print("응답 본문: ${utf8.decode(response.bodyBytes)}");
      return null;
    }
  } catch (e) {
    print('파일 업로드 중 오류 발생: $e');
    return null;
  }
}