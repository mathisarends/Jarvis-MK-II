# from rag.smart_rag_manager import SmartRAGManager

# rag_manager = SmartRAGManager()

# query = "Was ist Gegenstand meines Jarvis Projekts?"
# response = rag_manager.query(query)
# print("Antwort:", response)

from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent

from tools.notion.clipboard.notion_clipboard_tool import NotionClipboardTool

notion_tool = NotionClipboardTool()

llm = ChatOpenAI(model="gpt-4o-mini")

agent = initialize_agent(
    tools=[notion_tool],
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

# Testaufruf
response = agent.invoke("Speichere den Text 'Wichtige Meeting-Notiz' in die Notion-Zwischenablage.")
print(response)