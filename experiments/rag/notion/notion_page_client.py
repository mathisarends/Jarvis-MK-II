import asyncio
import re
from typing import Any, Dict, Optional
from experiments.rag.notion.abstract_notion_client import AbstractNotionClient
from experiments.rag.notion.core.notion_block import NotionBlock
from experiments.rag.notion.core.notion_markdown_converter import NotionMarkdownConverter

class NotionPageClient(AbstractNotionClient):
    """Client für die Interaktion mit Notion-Seiten."""
    
    def __init__(self):
        """
        Initialisiert den Client mit einem Markdown-Konverter.
        
        Args:
            markdown_converter (NotionMarkdownConverter, optional): Benutzerdefinierter Konverter.
        """
        super().__init__()
        self._markdown_converter = NotionMarkdownConverter()
    
    async def get_page_metadata(self, page_id: str) -> Optional[str]:
        """
        Ruft Metadaten einer Notion-Seite ab.
        
        Args:
            page_id (str): Eindeutige Kennung der Seite.
        
        Returns:
            Optional[str]: Zeitpunkt der letzten Bearbeitung oder None.
        """
        endpoint = f"pages/{page_id}"
        response = await self._make_request("get", endpoint)
        response_json = response.json()

        return response_json.get("last_edited_time") if "error" not in response_json else None

    async def _parse_block(self, block: Dict[str, Any], depth: int = 0) -> NotionBlock:
        """
        Parst einen Notion-Block rekursiv.
        
        Args:
            block (Dict): Rohdaten des Blocks.
            depth (int): Verschachtelungstiefe.
        
        Returns:
            NotionBlock: Strukturierter Block mit Kindern.
        """
        block_type = block.get("type")
        rich_text = block.get(block_type, {}).get("rich_text", [])
        text = "".join([t.get("plain_text", "") for t in rich_text]).strip()

        # Überspringe leere Blöcke
        if not text and not block.get("has_children", False):
            return None

        notion_block = NotionBlock(
            type=block_type,
            text=text,
            depth=depth
        )

        # Rekursives Laden von Kinderblöcken
        if block.get("has_children", False):
            children_endpoint = f"blocks/{block['id']}/children"
            response = await self._make_request("get", children_endpoint)
            children_blocks = response.json().get("results", [])
            
            for child_block in children_blocks:
                child = await self._parse_block(child_block, depth + 1)
                if child:
                    notion_block.children.append(child)

        return notion_block

    async def get_page_markdown_content(self, page_id: str) -> str:
        """
        Ruft den Inhalt einer Notion-Seite als Markdown ab.
        
        Args:
            page_id (str): Eindeutige Kennung der Seite.
        
        Returns:
            str: Markdown-formatierter Seiteninhalt.
        """
        # Hole den Stammblock der Seite
        root_block = await self._parse_block({"id": page_id, "type": "page", "has_children": True})
        
        # Konvertiere Block in Markdown
        markdown_lines = self._markdown_converter.convert_block_to_markdown(root_block)
        
        # Bereinige und formatiere Markdown
        clean_text = "\n".join(markdown_lines)
        clean_text = re.sub(r"\n{3,}", "\n\n", clean_text)

        return clean_text.strip()

async def main():
    page_id = "1a6389d5-7bd3-80ac-a51b-ea79142d8204"
    client = NotionPageClient()
    
    content = await client.get_page_markdown_content(page_id)
    print("\n--- Notion Page Content ---\n")
    print(content)

if __name__ == "__main__":
  asyncio.run(main())