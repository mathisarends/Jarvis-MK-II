from jarvis.core.text_to_speech_streamer import TextToSpeechStreamer
from langchain.schema import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from jarvis.core.voice_generator import VoiceGenerator
from jarvis.core.speech_to_text.gemini_content_extractor import GeminiContentExtractor
import time

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-thinking-exp-01-21")
voice_generator = VoiceGenerator()
tts_streamer = TextToSpeechStreamer(voice_generator, GeminiContentExtractor())

chat_history = []

system_prompt = SystemMessage(
    content="Du bist Jarvis, ein hochentwickelter KI-Assistent, der präzise, höfliche und intelligente Antworten gibt. "
            "Dein Ziel ist es, dem Benutzer mit logischen und detaillierten Erklärungen zu helfen. Formuliere deine Antworten bitte so, dass sie in gesprochener Sprache Sinn machen. "
            "Also kein Markdown verwenden und immer auf Englisch antworten."
)

chat_history.append(system_prompt)

chat_history.append(HumanMessage(content="Gebe mir drei interessante Fakten und mache dabei eine Auflistung 1. 2. 3."))

response_stream = model.stream(chat_history)
full_response = tts_streamer.process_stream(response_stream)

print("\n🔵 Warten auf Abbruch... Drücke STRG + C, um das Programm zu beenden.")

try:
    while True:
        time.sleep(1)  
except KeyboardInterrupt:
    print("\n🔴 Programm wurde manuell beendet.")
