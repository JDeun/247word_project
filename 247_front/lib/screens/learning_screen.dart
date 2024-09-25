import 'package:flutter/material.dart';
import '../models/learning_material.dart';

class LearningScreen extends StatelessWidget {
  final LearningMaterial learningMaterial;

  const LearningScreen({Key? key, required this.learningMaterial}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Learning Screen'),
      ),
      body: learningMaterial.isEmpty
          ? Center(child: Text('No learning material available. Please try again.'))
          : ListView(
        children: [
          if (learningMaterial.error != null)
            ListTile(
              title: Text('Error'),
              subtitle: Text(learningMaterial.error!),
              tileColor: Colors.red.shade100,
            ),
          if (learningMaterial.partialContent != null)
            ListTile(
              title: Text('Partial Content'),
              subtitle: Text(learningMaterial.partialContent!),
              tileColor: Colors.yellow.shade100,
            ),
          // Display dialogues
          ...learningMaterial.dialogue.map((dialogue) => ListTile(
            title: Text(dialogue.english),
            subtitle: Text(dialogue.korean),
            leading: Text(dialogue.speaker),
          )),
          if (learningMaterial.dialogue.isNotEmpty && learningMaterial.vocabulary.isNotEmpty)
            Divider(),
          // Display vocabulary
          ...learningMaterial.vocabulary.map((vocab) => ListTile(
            title: Text(vocab.word),
            subtitle: Text(vocab.meaning),
          )),
        ],
      ),
    );
  }
}