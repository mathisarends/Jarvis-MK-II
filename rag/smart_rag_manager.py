import logging
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from rag.chroma_db_manager import ChromaDBManager

class SmartRAGManager:
    """Steuert die dynamische Entscheidung, ob Retrieval genutzt wird."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.db_manager = ChromaDBManager()
        self.llm = ChatOpenAI(model="gpt-4o-mini")

        # Entscheidungs-Prompt
        self.decision_prompt = PromptTemplate.from_template(
            """Beantworte mit 'yes' oder 'no'.
    
            Soll für die folgende Frage eine semantische Suche in der Datenbank genutzt werden?
    
            Frage: {question}
    
            Antworte mit 'yes' falls es sich um eine persönliche, projektbezogene oder spezifische Frage handelt.
            Antworte mit 'no' falls es sich um allgemeines Wissen handelt."""
        )

    def should_use_retrieval(self, question: str) -> bool:
        """Lässt das LLM entscheiden, ob eine semantische Suche notwendig ist."""
        decision_chain = LLMChain(llm=self.llm, prompt=self.decision_prompt)
        result = decision_chain.run(question).strip().lower()
        self.logger.info("Entscheidung für Retrieval: %s", result)
        return result == "yes"

    def query(self, query: str, chat_history=None) -> str:
        """Dynamische Entscheidungslogik für die Abfrage."""
        if chat_history is None:
            chat_history = []
            
        use_retrieval = self.should_use_retrieval(query)
            
        if use_retrieval:
            self.logger.info("Verwende RAG für die Abfrage.")
            result = self.db_manager.rag_chain.invoke({"input": query, "chat_history": chat_history})
            return result["answer"]
            
        else:
            self.logger.info("Kein RAG notwendig, nutze direkte LLM-Antwort.")
            direct_answer = self.llm.invoke(query)
            return direct_answer.content
