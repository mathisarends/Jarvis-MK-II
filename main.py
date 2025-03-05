from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

chat_history = [
    SystemMessage(content="You are an advanced AI assistant that responds precisely, politely, and intelligently.")
]

print("🔵 Jarvis AI gestartet. Gib 'exit' ein, um das Gespräch zu beenden.")

while True:
    user_input = input("\n🟢 Du: ")

    if user_input.lower() in ["exit", "quit", "stop"]:
        print("🔴 Jarvis: Gespräch beendet.")
        break

    chat_history.append(HumanMessage(content=user_input))

    response = model.stream(chat_history)

    print("\n🔵 Jarvis:", response.content)
    chat_history.append(AIMessage(content=response.content))