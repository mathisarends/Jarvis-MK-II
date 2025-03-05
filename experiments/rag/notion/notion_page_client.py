
import asyncio
from experiments.rag.notion.abstract_notion_client import AbstractNotionClient


class NotionPageClient(AbstractNotionClient):
    """Client for interacting with Notion pages."""
    
    async def get_page_metadata(self, page_id: str):
        """Retrieves metadata of a Notion page, including last edited time."""
        endpoint = f"pages/{page_id}"
        response = await self._make_request("get", endpoint)
        response_json = response.json()

        # Prüfe, ob ein Fehler aufgetreten ist
        if "error" in response_json:
            return None

        return response_json.get("last_edited_time")  # Zeitstempel der letzten Änderung
    

    async def get_page_text_content(self, page_id: str):
        """Retrieves the text content from a Notion page by fetching its block children."""
        endpoint = f"blocks/{page_id}/children"
        response = await self._make_request("get", endpoint)

        # HTTPX Response in JSON umwandeln
        response_json = response.json()  # Hier erfolgt die Umwandlung

        if "error" in response_json:
            return response_json

        # Extrahiere nur textuelle Inhalte
        text_content = []
        for block in response_json.get("results", []):
            block_type = block.get("type")
            if block_type in ["paragraph", "heading_1", "heading_2", "heading_3", "bulleted_list_item", "numbered_list_item"]:
                rich_text = block[block_type].get("rich_text", [])
                text = "".join([t.get("plain_text", "") for t in rich_text])
                if text:
                    text_content.append(text)

        return "\n".join(text_content)
    
    async def get_page_markdown_content(self, page_id: str):
        """Retrieves the text content from a Notion page and formats it as Markdown."""
        endpoint = f"blocks/{page_id}/children"
        response = await self._make_request("get", endpoint)
        response_json = response.json()

        if "error" in response_json:
            return response_json

        # Markdown-Text speichern
        markdown_content = []

        for block in response_json.get("results", []):
            block_type = block.get("type")
            rich_text = block.get(block_type, {}).get("rich_text", [])

            # Extrahiere den reinen Text aus Rich-Text-Objekten
            text = "".join([t.get("plain_text", "") for t in rich_text])

            # Wandle Notion-Typen in Markdown um
            if block_type == "heading_1":
                markdown_content.append(f"# {text}")
            elif block_type == "heading_2":
                markdown_content.append(f"## {text}")
            elif block_type == "heading_3":
                markdown_content.append(f"### {text}")
            elif block_type == "bulleted_list_item":
                markdown_content.append(f"- {text}")
            elif block_type == "numbered_list_item":
                markdown_content.append(f"1. {text}")  # Notion gibt keine Zahlen, nur Reihenfolge
            elif block_type == "quote":
                markdown_content.append(f"> {text}")
            elif block_type == "code":
                markdown_content.append(f"```\n{text}\n```")
            elif block_type == "paragraph":
                markdown_content.append(text)
        
        return "\n\n".join(markdown_content)  # Trenne Blöcke durch doppelte Zeilenumbrüche




async def main():
    page_id = "1a6389d5-7bd3-80ac-a51b-ea79142d8204"
    client = NotionPageClient()
    
    # last_edited_time = await client.get_page_metadata(page_id)
    # print(last_edited_time)
    
    content = await client.get_page_markdown_content(page_id)
    print("\n--- Notion Page Content ---\n")
    print(content)

if __name__ == "__main__":
    asyncio.run(main())