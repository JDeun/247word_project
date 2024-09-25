class Dialogue {
  String speaker;
  String english;
  String korean;

  Dialogue({required this.speaker, required this.english, required this.korean});

  factory Dialogue.fromJson(Map<String, dynamic> json) {
    return Dialogue(
      speaker: json['speaker'] ?? '',
      english: json['english'] ?? '',
      korean: json['korean'] ?? '',
    );
  }
}

class Vocabulary {
  String word;
  String meaning;

  Vocabulary({required this.word, required this.meaning});

  factory Vocabulary.fromJson(Map<String, dynamic> json) {
    return Vocabulary(
      word: json['word'] ?? '',
      meaning: json['meaning'] ?? '',
    );
  }
}

class LearningMaterial {
  List<Dialogue> dialogue;
  List<Vocabulary> vocabulary;
  String? error;
  String? partialContent;

  LearningMaterial({required this.dialogue, required this.vocabulary, this.error, this.partialContent});

  factory LearningMaterial.fromJson(Map<String, dynamic> json) {
    return LearningMaterial(
      dialogue: (json['dialogue'] as List?)?.map((d) => Dialogue.fromJson(d)).toList() ?? [],
      vocabulary: (json['vocabulary'] as List?)?.map((v) => Vocabulary.fromJson(v)).toList() ?? [],
      error: json['error'],
      partialContent: json['partial_content'],
    );
  }

  bool get isEmpty => dialogue.isEmpty && vocabulary.isEmpty && error == null && partialContent == null;
}