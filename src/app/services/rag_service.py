import logging
from typing import Any, AsyncGenerator
from langchain_core.embeddings import Embeddings
from lib_utils.interfaces.vector_store import VectorStoreClient
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

from app.configs.config import Settings
from app.schemas.ask import AskRequest, AskResponse
from app.prompts.ask import rag_prompt

logger = logging.getLogger(__name__)


class RagService:
    """
    Service class for handling Retrieval-Augmented Generation (RAG) operations.
    This class initializes embedding and vector store services based on the provided settings.
    """

    def __init__(self, settings: Settings):
        """
        Initialize the RagService with the specified settings.
        """

        # Embedding instance
        if settings.embedding.provider == "huggingface":
            from langchain_huggingface import HuggingFaceEmbeddings

            self.embedding: Embeddings = HuggingFaceEmbeddings(
                model_name=settings.embedding.model,
            )
        elif settings.embedding.provider == "openai":
            from langchain_openai import OpenAIEmbeddings

            self.embedding = OpenAIEmbeddings(
                model=settings.embedding.model,
                dimensions=settings.embedding.size,
                api_key=settings.embedding.api_key,
            )

        # VectorStore Client instance
        if settings.vector_store.provider == "pinecone":
            from lib_utils.vector_database.pinecone import PineconeService

            if settings.vector_store.api_key:
                self.vs_client: VectorStoreClient = PineconeService(
                    self.embedding,
                    settings.vector_store.api_key.get_secret_value(),
                    settings.vector_store.index_name,
                    settings.embedding.size,
                    rerank_top_n=settings.vector_store.rerank_top_n,
                )
            else:
                raise EnvironmentError("Vector Store API KEY not found")

        if settings.llm.api_key:
            self.chat_llm = init_chat_model(
                model=settings.llm.name,
                model_provider=settings.llm.provider,
                api_key=settings.llm.api_key.get_secret_value(),
            )
        else:
            raise EnvironmentError("LLM_API_KEY not found")

        self.rag_prompt: ChatPromptTemplate = rag_prompt

    async def _retrieve_and_rerank(
        self, ask_request: AskRequest, retrieval_type: str
    ) -> list[dict[str, Any]]:
        """
        Retrieve documents using the vector store client and optionally rerank them.
        """
        try:
            docs_retrieved = await self.vs_client.retrieve(
                ask_request.query, retrieval_type, ask_request.k
            )
            docs_reranked = await self.vs_client.rerank_context(docs_retrieved, ask_request.query)
            return docs_reranked
        except Exception as e:
            logger.exception(f"Error in _retrieve_and_rerank: {e}")
            return []

    async def _stream_llm_events(
        self, chain, ask_request, retrieved_docs
    ) -> AsyncGenerator[str, None]:
        try:
            async for event in chain.astream_events(
                input={
                    "question": ask_request.query,
                    "context": retrieved_docs,
                    "language": ask_request.language,
                },
                version="v2",
                include_tags=["seq:step:2"],
                temperature=ask_request.temperature,
            ):
                if event["event"] == "on_chat_model_stream":
                    data = event["data"].get("chunk")
                    if data:
                        logger.debug(f"Yielding: {data.text()}")
                        yield AskResponse(stage="tok", data=data.text()).model_dump_json()
                elif event["event"] == "on_chat_model_end":
                    data = event["data"].get("output")
                    if data:
                        logger.info("LLM stream ended, yielding final response.")
                        logger.debug(f"Yielding data: {data.text()}\nContexts: {retrieved_docs}")
                        yield AskResponse(
                            stage="end", data=data.text(), contexts=retrieved_docs
                        ).model_dump_json()
        except Exception as e:
            logger.error(f"Error during LLM streaming: {e}", exc_info=True)

    async def run_rag_pipeline_stream(
        self, ask_request: AskRequest, retrieval_type: str
    ) -> AsyncGenerator[str, None]:
        """
        Executes a streaming RAG pipeline: yields LLM response chunks.
        """
        retrieved_docs = await self._retrieve_and_rerank(ask_request, retrieval_type)
        if not retrieved_docs:
            logger.warning("No documents retrieved; aborting pipeline.")
            return
        chain = self.rag_prompt | self.chat_llm
        async for response in self._stream_llm_events(chain, ask_request, retrieved_docs):
            yield response
