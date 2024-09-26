import os
from speech_recognizer import SpeechRecognizer
from audio_handler import AudioHandler
from tutor import EnglishTutor
import config
import speech_recognition as sr

def main():
    print("영어 튜터 프로그램을 시작합니다.")

    # 각 모듈 초기화
    speech_recognizer = SpeechRecognizer()
    audio_handler = AudioHandler()
    tutor = EnglishTutor()

    # 마이크 선택
    mic_index = speech_recognizer.select_microphone()

    with sr.Microphone(device_index=mic_index) as source:
        print("주변 소음을 조정중입니다...")
        speech_recognizer.recognizer.adjust_for_ambient_noise(source, duration=1)
        print("영어 튜터와 대화를 시작하세요. '종료'라고 말하면 프로그램이 종료됩니다.")

        # 초기 인사 출력 및 음성 재생
        print(f"Tutor: {config.INITIAL_GREETING}")
        audio_handler.text_to_speech(config.INITIAL_GREETING)

        while True:
            # 음성 입력 받기
            audio_file = speech_recognizer.record_audio(source)
            
            if audio_file is None:
                print("음성 입력이 없습니다. 다시 말씀해 주세요.")
                continue

            try:
                # 음성을 텍스트로 변환
                user_input = speech_recognizer.transcribe_audio(audio_file)
                print(f"You: {user_input}")

                if '종료' in user_input.lower():
                    print("대화를 종료합니다.")
                    break
                
                # 튜터의 응답 생성
                tutor_response = tutor.get_response(user_input)
                print(f"Tutor: {tutor_response}")

                # 튜터의 응답을 음성으로 출력
                audio_handler.text_to_speech(tutor_response)

            except Exception as e:
                print(f"오류 발생: {e}")
            finally:
                # 임시 파일 삭제
                if audio_file and os.path.exists(audio_file):
                    os.remove(audio_file)

if __name__ == "__main__":
    main()