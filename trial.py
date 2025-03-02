from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from jarvis.core.text_to_speech_streamer import TextToSpeechStreamer
from jarvis.core.voice_generator import VoiceGenerator

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-thinking-exp-01-21")

tts = VoiceGenerator()
tts_streamer = TextToSpeechStreamer(tts)

chat_history = []

system_prompt = SystemMessage(
    content="Du bist Jarvis, ein hochentwickelter KI-Assistent, der präzise, höfliche und intelligente Antworten gibt. "
            "Dein Ziel ist es, dem Benutzer mit logischen und detaillierten Erklärungen zu helfen. Formuliere deine Antworten bitte so, dass sie in gesprochener Sprache Sinn machen."
            "Also kein Markdown verwenden und immer auf englsich antwortne."
)

chat_history.append(system_prompt)

print("🔵 Jarvis AI gestartet. Gib 'exit' ein, um das Gespräch zu beenden.")

while True:
    user_input = input("\n🟢 Du: ")
    chat_history.append(HumanMessage(content=user_input))

    if user_input.lower() in ["exit", "quit", "stop"]:
        print("🔴 Jarvis: Gespräch beendet.")
        break
    
    response_stream = model.stream(chat_history)
    full_response = tts_streamer.process_openai_stream(response_stream)
    chat_history.append(AIMessage(content=full_response))