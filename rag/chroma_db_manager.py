import os
import logging
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import MarkdownTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI

load_dotenv()

class ChromaDBManager:
    """
    Erweiterter ChromaDBManager für Notion Vector Database mit semantischer Suche.
    """

    def __init__(self):
        """
        Initialisiert den ChromaDBManager für Notion-Seiten.
        """
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(__name__)

        # Verzeichnis für die Datenbank festlegen
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.persistent_directory = os.path.join(current_dir, "chroma_db")
        os.makedirs(self.persistent_directory, exist_ok=True)

        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self._initialize_retriever()
        
        
    def _initialize_retriever(self):
        self.db = Chroma(persist_directory=self.persistent_directory, embedding_function=self.embeddings)

        self.retriever = self.db.as_retriever(verbose=True)

        self.llm = ChatOpenAI(model="gpt-4o-mini")

        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Reformuliere die Frage so, dass sie ohne vorherigen Chatverlauf verständlich ist."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        self.history_aware_retriever = create_history_aware_retriever(self.llm, self.retriever, contextualize_q_prompt)

        # Antwortgenerierung mit Dokumenten
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "Beantworte die folgende Frage basierend auf dem gegebenen Kontext:\n\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        self.question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        
        # Gesamtkette für RAG (Retrieval-Augmented Generation)
        self.rag_chain = create_retrieval_chain(self.history_aware_retriever, self.question_answer_chain)
        

    def get_existing_page_docs(self, page_id: str) -> dict:
        """Liefert alle existierenden Dokumente einer Seite."""
        return self.db.get(where={"source": page_id})

    def delete_page_docs(self, page_id: str):
        """Löscht Dokumente einer bestimmten Notion-Seite."""
        self.db.delete(where={"source": page_id})
        self.logger.info(f"Alte Einträge für Seite {page_id} gelöscht")

    def add_page_documents(self, page_id: str, markdown_text: str, last_edited_time: str):
        """
        Fügt Dokumente für eine Notion-Seite zur Datenbank hinzu.
        """
        text_splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=100)
        docs = text_splitter.create_documents([markdown_text])

        # Metadaten für jeden Chunk hinzufügen
        for doc in docs:
            doc.metadata = {"source": page_id, "last_edited": last_edited_time}

        self.db.add_documents(docs)
        self.logger.info(f"{len(docs)} neue Chunks für Seite {page_id} gespeichert")

    def query_semantic(self, query: str, chat_history=None) -> str:
        """
        Führt eine semantische Abfrage durch und liefert eine Antwort basierend auf gespeicherten Dokumenten.
        
        Args:
            query (str): Die Nutzeranfrage.
            chat_history (list): Bisheriger Chatverlauf.

        Returns:
            str: Generierte Antwort.
        """
        if chat_history is None:
            chat_history = []

        result = self.rag_chain.invoke({"input": query, "chat_history": chat_history}, verbose=True)
        
        context_docs = result.get("context", "Kein Kontext gefunden.")
        
        print("\n===== RETRIEVED CONTEXT =====")
        if isinstance(context_docs, list):
            for idx, doc in enumerate(context_docs):
                print(f"\n--- Dokument {idx + 1} ---")
                print(doc)
        else:
            print(context_docs)
        print("====================================\n")
        
        return result["answer"]
