from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import router
from backend.app.core.logging import setup_logging

setup_logging()

app = FastAPI(
    title='Enterprise Document Q&A with Agentic RAG',
    version='1.0.0',
    description='Upload enterprise documents and ask grounded questions using an agentic RAG pipeline.',
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(router)


@app.get('/health')
def health() -> dict:
    return {'status': 'ok'}
