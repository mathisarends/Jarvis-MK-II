import os
import asyncio
from langchain.text_splitter import MarkdownTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from experiments.rag.notion.notion_page_client import NotionPageClient

async def store_notion_page_in_vector_db(page_id: str):
    client = NotionPageClient()
    markdown_text = await client.get_page_markdown_content(page_id)

    text_splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.create_documents([markdown_text])

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    persistent_directory = os.path.join(current_dir, "chroma_db")

    if not os.path.exists(persistent_directory):
        print(f"📁 Initializing new vector store at {persistent_directory}...")
        db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
    else:
        print("✅ Vector store already exists. Adding new documents...")
        db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
        db.add_documents(docs)

    print(f"✅ {len(docs)} Chunks wurden erfolgreich in der Vektordatenbank gespeichert!")

if __name__ == "__main__":
    asyncio.run(store_notion_page_in_vector_db("1a6389d5-7bd3-80ac-a51b-ea79142d8204"))
