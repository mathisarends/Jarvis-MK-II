import os
from langchain.text_splitter import MarkdownTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "jarvis.txt")
persistent_directory = os.path.join(current_dir, "db", "chroma_db")


if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check the path."
        )

    loader = TextLoader(file_path, encoding="UTF-8")
    documents = loader.load()
    
    for doc in documents:
        doc.metadata = {"source": file_path}

    text_splitter = MarkdownTextSplitter()
    docs = text_splitter.split_documents(documents)

    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk:\n{docs[0].page_content}\n")

    print("\n--- Creating embeddings ---")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )  
    print("\n--- Finished creating embeddings ---")

    print("\n--- Creating vector store ---")
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_directory)
    print("\n--- Finished creating vector store ---")

else:
    print("Vector store already exists. No need to initialize.")
