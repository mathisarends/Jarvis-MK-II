from rag.smart_rag_manager import SmartRAGManager

rag_manager = SmartRAGManager()

query = "Was ist Gegenstand meines Jarvis Projekts?"
response = rag_manager.query(query)
print("Antwort:", response)