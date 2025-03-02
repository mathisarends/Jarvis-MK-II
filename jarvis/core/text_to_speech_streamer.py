import re

class TextToSpeechStreamer:
    def __init__(self, voice_generator, min_chunk_size=80, optimal_chunk_size=150, max_chunk_size=250):
        """Initialisiert den Text-zu-Sprache-Streamer mit konfigurierbaren Chunk-Größen"""
        self.tts = voice_generator
        self.MIN_CHUNK_SIZE = min_chunk_size
        self.OPTIMAL_CHUNK_SIZE = optimal_chunk_size
        self.MAX_CHUNK_SIZE = max_chunk_size
        
        self.BREAK_PATTERNS = [
            r'(?<=[.!?])\s+(?=[A-Z"„\'])',
            r'(?<=[.!?])\s+',
            r'(?<=[:;])\s+',
            r'(?<=,)\s+(?=[und|oder|aber|denn|sondern|weil|dass|wenn])',
            r'(?<=,)\s+',
            r'(?<=–|—)\s+',
            r'\n+',
        ]
    
    def stream_text(self, text_stream):
        """
        Verarbeitet einen Text-Stream und gibt ihn in optimierten Chunks für TTS aus.
        
        Args:
            text_stream: Iterator, der Text-Chunks liefert
            
        Returns:
            Der vollständige Text als String
        """
        full_response = ""
        buffer = ""
        processed_segments = set()
        
        for text_chunk in text_stream:
            buffer += text_chunk
            full_response += text_chunk
            
            if len(buffer) < self.MIN_CHUNK_SIZE:
                continue
                
            buffer = self._process_buffer(buffer, processed_segments)
        
        self._process_remaining_buffer(buffer, processed_segments)
        
        return full_response
    
    def _process_buffer(self, buffer, processed_segments):
        if len(buffer) < self.OPTIMAL_CHUNK_SIZE:
            return buffer
        
        chunk_text, new_buffer = self._find_optimal_chunk(buffer)
        
        if chunk_text and chunk_text not in processed_segments:
            processed_segments.add(chunk_text)
            self.tts.speak(chunk_text)
        
        return new_buffer

    def _find_optimal_chunk(self, text):
        for pattern in self.BREAK_PATTERNS:
            matches = list(re.finditer(pattern, text))
            optimal_matches = [m for m in matches if m.end() >= self.MIN_CHUNK_SIZE]
            
            if optimal_matches:
                best_match = self._select_best_match(optimal_matches)
                split_pos = best_match.end()
                
                return text[:split_pos].strip(), text[split_pos:]
        
        if len(text) > self.MAX_CHUNK_SIZE:
            last_space = text.rfind(' ', self.MIN_CHUNK_SIZE, self.MAX_CHUNK_SIZE)
            if last_space > self.MIN_CHUNK_SIZE:
                return text[:last_space].strip(), text[last_space:].lstrip()
        
        return None, text

    def _select_best_match(self, matches):
        best_match = None
        best_distance = float('inf')
        
        for match in matches:
            distance = abs(match.end() - self.OPTIMAL_CHUNK_SIZE)
            if distance < best_distance:
                best_distance = distance
                best_match = match
        
        return best_match

    def _process_remaining_buffer(self, buffer, processed_segments):
        remaining_text = buffer.strip()
        if remaining_text and remaining_text not in processed_segments:
            self.tts.speak(remaining_text)
        
    def process_openai_stream(self, openai_stream):
        """
        Spezialisierte Methode für OpenAI- und Gemini-Stream-Verarbeitung
        
        Args:
            openai_stream: OpenAI oder Gemini Chat Completion Stream
            
        Returns:
            Der vollständige Text als String
        """
        def extract_content(chunk):
            """
            Extrahiert den Text aus einem Stream-Chunk.
            Unterstützt OpenAI (GPT) und Gemini (Google AI).
            """
            # 🔹 OpenAI-Modelle (GPT-4o, GPT-3.5)
            if hasattr(chunk, "choices") and hasattr(chunk.choices[0], "delta"):
                return chunk.choices[0].delta.content or ""

            # 🔹 Gemini-Modelle (Google AI)
            elif hasattr(chunk, "content"):
                return chunk.content or ""

            return ""

        text_chunks = (extract_content(chunk) for chunk in openai_stream)
        return self.stream_text(text_chunks)