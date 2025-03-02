from collections import deque
import time
from jarvis.core.voice_generator import VoiceGenerator  

class TextToSpeechStreamer:
    def __init__(self, voice_generator: VoiceGenerator):
        self.tts = voice_generator
        self.paragraphs = deque()  # Queue für erkannte Absätze

    def process_stream(self, openai_stream):
        def extract_content(chunk):
            if hasattr(chunk, "choices") and hasattr(chunk.choices[0], "delta"):
                return chunk.choices[0].delta.content or ""
            if hasattr(chunk, "content"):
                return chunk.content or ""
            return ""

        buffer = ""  # Zwischenspeicher für den aktuellen Absatz

        for chunk in openai_stream:
            part = extract_content(chunk)
            buffer += part  # Chunk in den Puffer anhängen

            while "\n\n" in buffer:  
                paragraph, buffer = buffer.split("\n\n", 1)  
                self.paragraphs.append(paragraph.strip())  

                print(f"\n📝 Neuer Absatz erkannt:\n{paragraph.strip()}")  
                
                # 🔊 Sprachausgabe für den erkannten Absatz
                self.speak_text(paragraph.strip())

        if buffer.strip():
            self.paragraphs.append(buffer.strip())
            print(f"\n📝 Letzter Absatz:\n{buffer.strip()}")
            
            # 🔊 Sprachausgabe für den letzten Absatz
            self.speak_text(buffer.strip())

        return list(self.paragraphs)

    def speak_text(self, text):
        """Gibt den Text über das TTS-System aus"""
        if self.tts:
            print(f"🔊 TTS: Spreche Absatz ...")
            self.tts.speak(text)
            time.sleep(0.5)  
        else:
            print(f"🔇 Kein TTS-System vorhanden, Absatz: {text}")
