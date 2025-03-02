from jarvis.core.speech_to_text.stream_content_extractor import StreamContentExtractor

class GeminiContentExtractor(StreamContentExtractor):
    """Extrahiert den Inhalt aus Google Gemini-Streaming-Responses."""

    def extract(self, chunk) -> str:
        if hasattr(chunk, "content"):
            return chunk.content or ""
        return ""