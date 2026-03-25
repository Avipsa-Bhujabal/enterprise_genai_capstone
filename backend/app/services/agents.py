from dataclasses import dataclass

from backend.app.services.llm import LLMService
from backend.app.services.vector_store import VectorStoreService


@dataclass
class AgentResult:
    answer: str
    citations: list[dict]
    trace: list[str]


class AgenticRAGService:
    def __init__(self, llm_service: LLMService, vector_store: VectorStoreService, default_top_k: int = 4) -> None:
        self.llm = llm_service
        self.vector_store = vector_store
        self.default_top_k = default_top_k

    def ask(self, question: str, top_k: int | None = None) -> AgentResult:
        trace: list[str] = []

        trace.append('Planner agent: analyzing the question and forming a retrieval strategy.')
        plan = self._planner(question)

        trace.append('Retriever agent: retrieving semantically similar chunks from the knowledge store.')
        hits = self.vector_store.similarity_search(plan, top_k=top_k or self.default_top_k)
        if not hits:
            return AgentResult(
                answer="I couldn't find relevant information in the uploaded documents.",
                citations=[],
                trace=trace + ['Validator agent: no evidence found, returned safe fallback response.'],
            )

        context = '\n\n'.join(
            [f"[{idx+1}] Source={hit['metadata'].get('source')} | Chunk={hit['chunk_id']}\n{hit['text']}" for idx, hit in enumerate(hits)]
        )

        trace.append('Reasoner agent: generating a grounded answer using retrieved evidence only.')
        answer = self._reason(question, context)

        trace.append('Validator agent: checking whether the answer is grounded and appropriately cautious.')
        validated_answer = self._validate(question, context, answer)

        citations = [
            {
                'source': hit['metadata'].get('source', 'unknown'),
                'chunk_id': hit['chunk_id'],
                'preview': hit['text'][:180],
            }
            for hit in hits
        ]

        return AgentResult(answer=validated_answer, citations=citations, trace=trace)

    def _planner(self, question: str) -> str:
        system_prompt = (
            'You are a planning agent for enterprise document retrieval. '
            'Rewrite the user question into a compact retrieval query with key entities and intent. '
            'Return only the optimized retrieval query.'
        )
        return self.llm.chat(system_prompt, question).strip()

    def _reason(self, question: str, context: str) -> str:
        system_prompt = (
            'You are a grounded enterprise Q&A assistant. '
            'Answer only from the retrieved context. '
            'If the answer is not fully supported, say that the documents do not contain enough evidence. '
            'Do not invent policies, numbers, or facts. '
            'When possible, cite source numbers like [1], [2].'
        )
        user_prompt = f'Question:\n{question}\n\nRetrieved Context:\n{context}'
        return self.llm.chat(system_prompt, user_prompt).strip()

    def _validate(self, question: str, context: str, draft_answer: str) -> str:
        system_prompt = (
            'You are a validation agent. '
            'Check whether the draft answer is supported by the provided context. '
            'If unsupported claims exist, rewrite the answer to be cautious and grounded. '
            'Keep the answer concise and professional.'
        )
        user_prompt = (
            f'Question:\n{question}\n\n'
            f'Context:\n{context}\n\n'
            f'Draft Answer:\n{draft_answer}'
        )
        return self.llm.chat(system_prompt, user_prompt).strip()
