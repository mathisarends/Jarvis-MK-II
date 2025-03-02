from jarvis.core.speech_to_text.stream_content_extractor import StreamContentExtractor

class OpenAIContentExtractor(StreamContentExtractor):
    """Extrahiert den Inhalt aus OpenAI-Streaming-Responses."""

    def extract(self, chunk) -> str:
        if hasattr(chunk, "choices") and hasattr(chunk.choices[0], "delta"):
            return chunk.choices[0].delta.content or ""
        if hasattr(chunk, "content"):
            return chunk.content or ""
        return ""