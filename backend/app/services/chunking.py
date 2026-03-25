from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str
    source: str
    text: str
    metadata: dict


class TextChunker:
    def __init__(self, chunk_size: int = 900, chunk_overlap: int = 150) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list[str]:
        cleaned = ' '.join(text.split())
        if not cleaned:
            return []

        chunks: list[str] = []
        start = 0
        while start < len(cleaned):
            end = start + self.chunk_size
            chunk = cleaned[start:end]
            if chunk:
                chunks.append(chunk)
            if end >= len(cleaned):
                break
            start = max(0, end - self.chunk_overlap)
        return chunks
