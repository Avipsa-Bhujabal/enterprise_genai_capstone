from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from backend.app.core.config import Settings, get_settings
from backend.app.models.schemas import AskRequest, AskResponse, Citation, IngestResponse
from backend.app.services.agents import AgenticRAGService
from backend.app.services.ingestion import DocumentIngestionService
from backend.app.services.llm import LLMService
from backend.app.services.rag import DocumentPipelineService
from backend.app.services.vector_store import VectorStoreService

router = APIRouter(prefix='/api/v1', tags=['enterprise-rag'])


def get_vector_store(settings: Settings = Depends(get_settings)) -> VectorStoreService:
    return VectorStoreService(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.collection_name,
    )


def get_pipeline(
    settings: Settings = Depends(get_settings),
    vector_store: VectorStoreService = Depends(get_vector_store),
) -> DocumentPipelineService:
    ingestion = DocumentIngestionService(settings.upload_dir_path)
    return DocumentPipelineService(
        ingestion_service=ingestion,
        vector_store=vector_store,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


def get_agent_service(
    settings: Settings = Depends(get_settings),
    vector_store: VectorStoreService = Depends(get_vector_store),
) -> AgenticRAGService:
    llm = LLMService(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        base_url=settings.openai_base_url,
    )
    return AgenticRAGService(
        llm_service=llm,
        vector_store=vector_store,
        default_top_k=settings.top_k,
    )


@router.post('/ingest', response_model=IngestResponse)
async def ingest_documents(
    files: list[UploadFile] = File(...),
    pipeline: DocumentPipelineService = Depends(get_pipeline),
) -> IngestResponse:
    try:
        chunks_created, sources = await pipeline.ingest_uploads(files)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f'Ingestion failed: {exc}') from exc

    return IngestResponse(
        message='Documents ingested successfully.',
        files_processed=len(sources),
        chunks_created=chunks_created,
        sources=sources,
    )


@router.post('/ask', response_model=AskResponse)
def ask_question(
    request: AskRequest,
    agent_service: AgenticRAGService = Depends(get_agent_service),
) -> AskResponse:
    try:
        result = agent_service.ask(request.question, request.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f'Question answering failed: {exc}') from exc

    return AskResponse(
        answer=result.answer,
        citations=[Citation(**citation) for citation in result.citations],
        agent_trace=result.trace,
    )
