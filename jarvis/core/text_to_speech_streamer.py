from collections import deque
import time

from jarvis.core.speech_to_text.stream_content_extractor import StreamContentExtractor
from jarvis.core.voice_generator import VoiceGenerator

class TextToSpeechStreamer:
    def __init__(self, voice_generator: VoiceGenerator, extractor: StreamContentExtractor):
        self.tts = voice_generator
        self.extractor = extractor  
        self.paragraphs = deque()  

    def process_stream(self, openai_stream):
        buffer = ""

        for chunk in openai_stream:
            part = self.extractor.extract(chunk)
            buffer += part

            while "\n\n" in buffer:
                paragraph, buffer = buffer.split("\n\n", 1)
                self.paragraphs.append(paragraph.strip())

                print(f"\n📝 Neuer Absatz erkannt:\n{paragraph.strip()}")
                self.speak_text(paragraph.strip())

        if buffer.strip():
            self.paragraphs.append(buffer.strip())
            print(f"\n📝 Letzter Absatz:\n{buffer.strip()}")
            self.speak_text(buffer.strip())

        return list(self.paragraphs)

    def speak_text(self, text):
        if self.tts:
            print(f"🔊 TTS: Spreche Absatz ...")
            self.tts.speak(text)
            time.sleep(0.5)  
        else:
            print(f"🔇 Kein TTS-System vorhanden, Absatz: {text}")
