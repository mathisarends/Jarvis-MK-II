import asyncio
import logging
from datetime import datetime, timedelta

from rag.chroma_db_manager import ChromaDBManager
from tools.notion.core.notion_pages import NotionPages
from tools.notion.notion_page_client import NotionPageClient

class NotionVectorDBUpdater:
    def __init__(
        self, 
        update_time: str = "00:00",
    ):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

        self.client = NotionPageClient()
        self.chroma_db = ChromaDBManager()
        
        self.update_time = update_time

    async def update_page(self, page_id: str):
        """Aktualisiert eine einzelne Notion-Seite in der Vektordatenbank."""
        try:
            last_edited_time = await self.client.get_page_metadata(page_id)
            existing_docs = self.chroma_db.get_existing_page_docs(page_id)

            # Prüfen, ob bereits ein Eintrag existiert und sich die Seite geändert hat
            if existing_docs and existing_docs["documents"]:
                stored_last_edited_time = existing_docs["metadatas"][0].get("last_edited", None)
                if stored_last_edited_time == last_edited_time:
                    self.logger.info("✅ Keine Änderungen für Seite %s. Kein Update erforderlich.", page_id)
                    return  
            else:
                self.logger.info("ℹ️ Kein bestehender Eintrag für %s gefunden. Speichere die Seite neu.", page_id)

            # Notion-Inhalt abrufen
            markdown_text = await self.client.get_page_markdown_content(page_id)

            # Alte Einträge löschen
            self.chroma_db.delete_page_docs(page_id)

            # Neue Dokumente hinzufügen
            self.chroma_db.add_page_documents(page_id, markdown_text, last_edited_time)

        except Exception as e:
            self.logger.error("❌ Fehler beim Aktualisieren von Seite %s: %s", page_id, e)

    async def update_all_pages(self):
        """Aktualisiert alle Notion-Seiten."""
        page_ids = NotionPages.list_all_project_pages()
        self.logger.info("🔄 Starte Update für %d Seiten...", len(page_ids))
        
        for page_id in page_ids:
            await self.update_page(page_id)
        
        self.logger.info("✅ Alle Seiten wurden aktualisiert.")

    async def start_scheduled_updates(self):
        """
        Startet tägliche Updates um Mitternacht.
        """
        while True:
            wait_seconds = self._determine_time_till_midnight()
            self.logger.info("⏳ Nächstes Update in %d Sekunden (um Mitternacht)...", wait_seconds)
            
            try:
                await asyncio.sleep(wait_seconds)
            except asyncio.CancelledError:
                self.logger.warning("⚠️ Der Scheduler wurde abgebrochen.")
                break
            except Exception as e:
                self.logger.error("❌ Fehler beim Warten auf Mitternacht: %s", e)
            
            try:
                self.logger.info("🌙 Es ist Mitternacht! Starte tägliches Update...")
                await self.update_all_pages()
            except Exception as e:
                self.logger.error("❌ Fehler beim täglichen Update: %s", e)

    def _determine_time_till_midnight(self) -> int:
        now = datetime.now()
        next_midnight = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return int((next_midnight - now).total_seconds())

async def main():
    """
    Hauptfunktion zum Starten des Updaters.
    Kann parallel mit anderen asyncio-Aufgaben laufen.
    """
    updater = NotionVectorDBUpdater()
    
    await updater.update_all_pages()
    
    await updater.start_scheduled_updates()

if __name__ == "__main__":
    asyncio.run(main())
