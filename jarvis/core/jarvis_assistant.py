import os
import asyncio

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

from experiments.rag.notion.notion_vector_db_updater import NotionVectorDBUpdater

class JarvisAssistant:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        """
        Initialisiert den Jarvis-Assistenten mit Vektor-DB und Chat-Modell.
        
        Args:
            model_name (str): Name des zu verwendenden Chat-Modells.
        """
        # Lade Umgebungsvariablen
        load_dotenv()

        # Initialisiere Chat-Modell
        self.chat_model = ChatOpenAI(model=model_name)

        # Initialisiere Embeddings und Vektor-Datenbank
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Pfad zur Chroma-Datenbank
        current_dir = os.path.dirname(os.path.abspath(__file__))
        persistent_directory = os.path.join(current_dir, "..", "..", "experimental", "rag", "notion", "db", "chroma_db")
        print(persistent_directory)
        
        if os.path.exists(persistent_directory):
            print("the path exists")
        
        # Initialisiere Chroma-Datenbank
        self.vectordb = Chroma(
            persist_directory=persistent_directory, 
            embedding_function=self.embeddings
        )

        # Konfiguriere Retriever
        self.retriever = self.vectordb.as_retriever()

        # Initialisiere Konversations-Retrieval-Kette
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.chat_model,
            retriever=self.retriever,
            return_source_documents=True
        )

        # Chat-Verlauf
        self.chat_history = [
            SystemMessage(
                content="Du bist ein hilfreicher KI-Assistent namens Jarvis. "
                "Nutze verfügbare Kontextinformationen, um präzise und intelligent zu antworten."
            )
        ]

        # Initialisiere Notion Vector DB Updater
        self.notion_updater = NotionVectorDBUpdater()

    async def start_background_updates(self):
        """Startet Hintergrund-Updates der Notion-Datenbank."""
        update_task = asyncio.create_task(self.notion_updater.start_scheduled_updates())
        return update_task

    def retrieve_context(self, query: str) -> str:
        """
        Hole relevante Kontextinformationen für eine Anfrage.
        
        Args:
            query (str): Benutzeranfrage
        
        Returns:
            str: Kontextinformationen
        """
        # Hole relevante Dokumente
        docs = self.retriever.get_relevant_documents(query)
        
        # Formatiere Kontext
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])
        return context

    async def process_query(self, query: str) -> str:
        """
        Verarbeitet eine Benutzeranfrage mit Kontext-Retrieval.
        
        Args:
            query (str): Benutzeranfrage
        
        Returns:
            str: KI-Antwort
        """
        # Hole Kontext
        context = self.retrieve_context(query)
        
        # Erweitere Systemprompt mit Kontext
        system_message = SystemMessage(
            content=f"Nutze folgende Kontextinformationen, um die Anfrage zu beantworten:\n\n{context}\n\n"
                    "Falls die Informationen relevant sind, beziehe sie in deine Antwort ein. "
                    "Wenn nicht, antworte basierend auf deinem allgemeinen Wissen."
        )
        
        # Aktualisiere Chat-Verlauf
        self.chat_history.append(system_message)
        self.chat_history.append(HumanMessage(content=query))
        
        # Generiere Antwort
        response = await asyncio.to_thread(
            self.conversation_chain.invoke, 
            {"question": query, "chat_history": self.chat_history}
        )
        
        answer = response['answer']
        
        # Aktualisiere Chat-Verlauf
        self.chat_history.append(AIMessage(content=answer))
        
        return answer