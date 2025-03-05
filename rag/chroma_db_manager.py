import os
import logging

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import MarkdownTextSplitter

class ChromaDBManager:
    """
    Minimaler ChromaDBManager für Notion Vector Database.
    """

    def __init__(
        self
    ):
        """
        Initialisiert den ChromaDBManager für Notion-Seiten.
        
        Args:
            current_dir (str): Aktuelles Verzeichnis für Datenbankablage
            log_level (int): Logging-Level
        """
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

        # Verzeichnis für die Datenbank festlegen
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.persistent_directory = os.path.join(current_dir, "chroma_db")
        os.makedirs(self.persistent_directory, exist_ok=True)

        # Embeddings initialisieren
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        # Chroma-Datenbank erstellen
        self.db = Chroma(
            persist_directory=self.persistent_directory, 
            embedding_function=self.embeddings
        )

    def get_existing_page_docs(self, page_id: str) -> dict:
        return self.db.get(where={"source": page_id})

    def delete_page_docs(self, page_id: str):
        self.db.delete(where={"source": page_id})
        self.logger.info(f"Alte Einträge für Seite {page_id} gelöscht")

    def add_page_documents(
        self, 
        page_id: str, 
        markdown_text: str, 
        last_edited_time: str
    ):
        """
        Fügt Dokumente für eine Seite zur Datenbank hinzu.
        
        Args:
            page_id (str): ID der Notion-Seite
            markdown_text (str): Markdown-Inhalt der Seite
            last_edited_time (str): Zeitpunkt der letzten Bearbeitung
        """
        text_splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=100)
        docs = text_splitter.create_documents([markdown_text])

        # Metadaten für jeden Chunk hinzufügen
        for doc in docs:
            doc.metadata = {"source": page_id, "last_edited": last_edited_time}

        # Dokumente zur Datenbank hinzufügen
        self.db.add_documents(docs)
        self.logger.info(f"{len(docs)} neue Chunks für Seite {page_id} gespeichert")