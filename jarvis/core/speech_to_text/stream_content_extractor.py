from abc import ABC, abstractmethod

class StreamContentExtractor(ABC):
    """Strategie-Interface für das Extrahieren von Text aus einem Stream-Chunk."""
    
    @abstractmethod
    def extract(self, chunk) -> str:
        """Extrahiert den relevanten Text aus einem Chunk."""
        pass
