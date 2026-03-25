from uuid import uuid4

from backend.app.services.chunking import Chunk, TextChunker
from backend.app.services.ingestion import DocumentIngestionService
from backend.app.services.vector_store import VectorStoreService


class DocumentPipelineService:
    def __init__(self, ingestion_service: DocumentIngestionService, vector_store: VectorStoreService, chunk_size: int, chunk_overlap: int) -> None:
        self.ingestion_service = ingestion_service
        self.vector_store = vector_store
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    async def ingest_uploads(self, uploads) -> tuple[int, list[str]]:
        all_chunks: list[Chunk] = []
        sources: list[str] = []

        for upload in uploads:
            saved_path = await self.ingestion_service.save_upload(upload)
            text = self.ingestion_service.parse_file(saved_path)
            if not text.strip():
                continue

            sources.append(saved_path.name)
            pieces = self.chunker.split_text(text)
            for index, piece in enumerate(pieces):
                all_chunks.append(
                    Chunk(
                        chunk_id=f'{saved_path.stem}-{index}-{uuid4().hex[:8]}',
                        source=saved_path.name,
                        text=piece,
                        metadata={
                            'source': saved_path.name,
                            'chunk_index': index,
                            'file_type': saved_path.suffix.lower(),
                        },
                    )
                )

        count = self.vector_store.add_chunks(all_chunks)
        return count, sources
