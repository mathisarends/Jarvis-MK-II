# from rag.smart_rag_manager import SmartRAGManager
# from tools.notion.todo.notion_todo_tool import NotionTodoTool

# rag_manager = SmartRAGManager()

# query = "Was ist Gegenstand meines Jarvis Projekts?"
# response = rag_manager.query(query)
# print("Antwort:", response)

from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent

from tools.notion.clipboard.notion_clipboard_tool import NotionClipboardTool
from tools.notion.todo.notion_todo_tool import NotionTodoTool

notion_tool = NotionClipboardTool()
notion_todo_tool = NotionTodoTool()

llm = ChatOpenAI(model="gpt-4o-mini")

agent = initialize_agent(
    tools=[notion_tool, notion_todo_tool],  # `notion_todo_tool` ist jetzt ein `StructuredTool`
    llm=llm,
    agent="structured-chat-zero-shot-react-description",  # 🎯 Unterstützt Multi-Input Tools!
    verbose=True
)

# Testaufruf
response = agent.invoke("Schreibe bitte 'Milch kaufen' auf die ToDo-List")
print(response)