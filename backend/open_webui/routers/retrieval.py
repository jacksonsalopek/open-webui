from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import os
import re
import shutil
import time
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Iterator, NamedTuple, Optional, Sequence, Union
from urllib.parse import urlparse

import tiktoken
from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)
from open_webui.config import (
    ENV,
    RAG_EMBEDDING_CONTENT_PREFIX,
    RAG_EMBEDDING_MODEL_AUTO_UPDATE,
    RAG_EMBEDDING_MODEL_TRUST_REMOTE_CODE,
    RAG_EMBEDDING_QUERY_PREFIX,
    RAG_RERANKING_MODEL_AUTO_UPDATE,
    RAG_RERANKING_MODEL_TRUST_REMOTE_CODE,
    UPLOAD_DIR,
)
from open_webui.constants import ERROR_MESSAGES
from open_webui.env import (
    DEVICE_TYPE,
    DOCKER,
    RAG_EMBEDDING_TIMEOUT,
    SENTENCE_TRANSFORMERS_BACKEND,
    SENTENCE_TRANSFORMERS_CROSS_ENCODER_BACKEND,
    SENTENCE_TRANSFORMERS_CROSS_ENCODER_MODEL_KWARGS,
    SENTENCE_TRANSFORMERS_CROSS_ENCODER_SIGMOID_ACTIVATION_FUNCTION,
    SENTENCE_TRANSFORMERS_MODEL_KWARGS,
)
from open_webui.internal.db import get_async_db, get_async_session
from open_webui.models.files import FileModel, Files, FileUpdateForm
from open_webui.models.knowledge import Knowledges

# Document loaders
from open_webui.retrieval.loaders.youtube import YoutubeLoader
from open_webui.retrieval.concepts.integration.router_wiring import (
    build_concept_graph_extras as _build_concept_graph_extras,
)
from open_webui.retrieval.utils import (
    build_loader_from_config,
    filter_accessible_collections,
    get_content_from_url,
    get_embedding_function,
    get_model_path,
    get_reranking_function,
    query_collection,
    query_collection_with_hybrid_search,
    query_doc,
    query_doc_with_hybrid_search,
)
from open_webui.retrieval.vector.async_client import ASYNC_VECTOR_DB_CLIENT
from open_webui.retrieval.vector.factory import VECTOR_DB_CLIENT
from open_webui.retrieval.vector.utils import filter_metadata
# Web search: Kagi is the default engine in this fork; arXiv, MDN, and
# Microsoft Learn are parallel providers that subject-specific queries get
# routed to. Other providers were stripped to keep the surface area (and
# config sprawl) minimal.
from open_webui.retrieval.web.arxiv import search_arxiv
from open_webui.retrieval.web.arxiv_router import route_query as route_arxiv
from open_webui.retrieval.web.bb_router import route_query as route_bitbucket
from open_webui.retrieval.web.bitbucket import search_bitbucket
from open_webui.retrieval.web.docs_router import route_query as route_docs
from open_webui.retrieval.web.hf_router import route_query as route_hf
from open_webui.retrieval.web.huggingface import search_huggingface
from open_webui.retrieval.web.kagi import search_kagi
from open_webui.retrieval.web.kagi_lenses import route_query as route_kagi_lens
from open_webui.retrieval.web.mdn import search_mdn
from open_webui.retrieval.web.mslearn import search_mslearn
from open_webui.retrieval.web.godot import search_godot
from open_webui.retrieval.web.main import SearchResult
from open_webui.retrieval.web.nl_filter import (
    WebSearchFilter,
    extract_filter_from_query,
    has_full_native_support,
)
from open_webui.retrieval.web import page_cache
from open_webui.retrieval.web.utils import get_web_loader
from open_webui.storage.provider import Storage
from open_webui.utils.access_control import has_permission
from open_webui.utils.access_control.files import has_access_to_file
from open_webui.utils.auth import get_admin_user, get_verified_user
from open_webui.utils.misc import (
    calculate_sha256_string,
    sanitize_text_for_db,
)
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

log = logging.getLogger(__name__)

##########################################
#
# Utility functions
# Give us this day our relevant chunks, and lead us
# not into hallucination, but deliver us from noise.
#
##########################################


def get_ef(
    engine: str,
    embedding_model: str,
    auto_update: bool = RAG_EMBEDDING_MODEL_AUTO_UPDATE,
):
    ef = None
    if embedding_model and engine == '':
        from sentence_transformers import SentenceTransformer

        try:
            ef = SentenceTransformer(
                get_model_path(embedding_model, auto_update),
                device=DEVICE_TYPE,
                trust_remote_code=RAG_EMBEDDING_MODEL_TRUST_REMOTE_CODE,
                backend=SENTENCE_TRANSFORMERS_BACKEND,
                model_kwargs=SENTENCE_TRANSFORMERS_MODEL_KWARGS,
            )
        except Exception as e:
            log.error(f'Error loading SentenceTransformer: {e}')

    return ef


def get_rf(
    engine: str = '',
    reranking_model: str | None = None,
    external_reranker_url: str = '',
    external_reranker_api_key: str = '',
    external_reranker_timeout: str = '',
    auto_update: bool = RAG_RERANKING_MODEL_AUTO_UPDATE,
):
    rf = None
    # Convert timeout string to int or None (system default)
    timeout_value = int(external_reranker_timeout) if external_reranker_timeout else None
    if reranking_model:
        if any(model in reranking_model for model in ['jinaai/jina-colbert-v2']):
            try:
                from open_webui.retrieval.models.colbert import ColBERT

                rf = ColBERT(
                    get_model_path(reranking_model, auto_update),
                    env='docker' if DOCKER else None,
                )

            except Exception as e:
                log.error(f'ColBERT: {e}')
                raise Exception(ERROR_MESSAGES.DEFAULT(e))
        else:
            if engine == 'external':
                try:
                    from open_webui.retrieval.models.external import ExternalReranker

                    rf = ExternalReranker(
                        url=external_reranker_url,
                        api_key=external_reranker_api_key,
                        model=reranking_model,
                        timeout=timeout_value,
                    )
                except Exception as e:
                    log.error(f'ExternalReranking: {e}')
                    raise Exception(ERROR_MESSAGES.DEFAULT(e))
            else:
                import sentence_transformers
                import torch

                try:
                    rf = sentence_transformers.CrossEncoder(
                        get_model_path(reranking_model, auto_update),
                        device=DEVICE_TYPE,
                        trust_remote_code=RAG_RERANKING_MODEL_TRUST_REMOTE_CODE,
                        backend=SENTENCE_TRANSFORMERS_CROSS_ENCODER_BACKEND,
                        model_kwargs=SENTENCE_TRANSFORMERS_CROSS_ENCODER_MODEL_KWARGS,
                        activation_fn=(
                            torch.nn.Sigmoid()
                            if SENTENCE_TRANSFORMERS_CROSS_ENCODER_SIGMOID_ACTIVATION_FUNCTION
                            else None
                        ),
                    )
                except Exception as e:
                    log.error(f'CrossEncoder: {e}')
                    raise Exception(ERROR_MESSAGES.DEFAULT('CrossEncoder error'))

                # Safely adjust pad_token_id if missing as some models do not have this in config
                try:
                    model_cfg = getattr(rf, 'model', None)
                    if model_cfg and hasattr(model_cfg, 'config'):
                        cfg = model_cfg.config
                        if getattr(cfg, 'pad_token_id', None) is None:
                            # Fallback to eos_token_id when available
                            eos = getattr(cfg, 'eos_token_id', None)
                            if eos is not None:
                                cfg.pad_token_id = eos
                                log.debug(f'Missing pad_token_id detected; set to eos_token_id={eos}')
                            else:
                                log.warning('Neither pad_token_id nor eos_token_id present in model config')
                except Exception as e2:
                    log.warning(f'Failed to adjust pad_token_id on CrossEncoder: {e2}')

    return rf


##########################################
#
# API routes
#
##########################################


router = APIRouter()


class CollectionNameForm(BaseModel):
    collection_name: str | None = None


class ProcessUrlForm(CollectionNameForm):
    url: str


class SearchForm(BaseModel):
    queries: list[str]


@router.get('/embedding')
async def get_embedding_config(request: Request, user=Depends(get_admin_user)):
    return {
        'status': True,
        'RAG_EMBEDDING_ENGINE': request.app.state.config.RAG_EMBEDDING_ENGINE,
        'RAG_EMBEDDING_MODEL': request.app.state.config.RAG_EMBEDDING_MODEL,
        'RAG_EMBEDDING_BATCH_SIZE': request.app.state.config.RAG_EMBEDDING_BATCH_SIZE,
        'ENABLE_ASYNC_EMBEDDING': request.app.state.config.ENABLE_ASYNC_EMBEDDING,
        'RAG_EMBEDDING_CONCURRENT_REQUESTS': request.app.state.config.RAG_EMBEDDING_CONCURRENT_REQUESTS,
        'openai_config': {
            'url': request.app.state.config.RAG_OPENAI_API_BASE_URL,
            'key': request.app.state.config.RAG_OPENAI_API_KEY,
        },
        'ollama_config': {
            'url': request.app.state.config.RAG_OLLAMA_BASE_URL,
            'key': request.app.state.config.RAG_OLLAMA_API_KEY,
        },
        'azure_openai_config': {
            'url': request.app.state.config.RAG_AZURE_OPENAI_BASE_URL,
            'key': request.app.state.config.RAG_AZURE_OPENAI_API_KEY,
            'version': request.app.state.config.RAG_AZURE_OPENAI_API_VERSION,
        },
    }


class OpenAIConfigForm(BaseModel):
    url: str
    key: str


class OllamaConfigForm(BaseModel):
    url: str
    key: str


class AzureOpenAIConfigForm(BaseModel):
    url: str
    key: str
    version: str


class EmbeddingModelUpdateForm(BaseModel):
    openai_config: OpenAIConfigForm | None = None
    ollama_config: OllamaConfigForm | None = None
    azure_openai_config: AzureOpenAIConfigForm | None = None
    RAG_EMBEDDING_ENGINE: str
    RAG_EMBEDDING_MODEL: str
    RAG_EMBEDDING_BATCH_SIZE: int | None = 1
    ENABLE_ASYNC_EMBEDDING: bool | None = True
    RAG_EMBEDDING_CONCURRENT_REQUESTS: int | None = 0


def unload_embedding_model(request: Request):
    if request.app.state.config.RAG_EMBEDDING_ENGINE == '':
        # unloads current internal embedding model and clears VRAM cache
        request.app.state.ef = None
        request.app.state.EMBEDDING_FUNCTION = None
        import gc

        gc.collect()
        if DEVICE_TYPE == 'cuda':
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()


@router.post('/embedding/update')
async def update_embedding_config(request: Request, form_data: EmbeddingModelUpdateForm, user=Depends(get_admin_user)):
    log.info(
        f'Updating embedding model: {request.app.state.config.RAG_EMBEDDING_MODEL} to {form_data.RAG_EMBEDDING_MODEL}'
    )
    unload_embedding_model(request)
    try:
        request.app.state.config.RAG_EMBEDDING_ENGINE = form_data.RAG_EMBEDDING_ENGINE
        request.app.state.config.RAG_EMBEDDING_MODEL = form_data.RAG_EMBEDDING_MODEL.strip()
        request.app.state.config.RAG_EMBEDDING_BATCH_SIZE = form_data.RAG_EMBEDDING_BATCH_SIZE
        request.app.state.config.ENABLE_ASYNC_EMBEDDING = form_data.ENABLE_ASYNC_EMBEDDING
        request.app.state.config.RAG_EMBEDDING_CONCURRENT_REQUESTS = form_data.RAG_EMBEDDING_CONCURRENT_REQUESTS

        if request.app.state.config.RAG_EMBEDDING_ENGINE in [
            'ollama',
            'openai',
            'azure_openai',
        ]:
            if form_data.openai_config is not None:
                request.app.state.config.RAG_OPENAI_API_BASE_URL = form_data.openai_config.url
                request.app.state.config.RAG_OPENAI_API_KEY = form_data.openai_config.key

            if form_data.ollama_config is not None:
                request.app.state.config.RAG_OLLAMA_BASE_URL = form_data.ollama_config.url
                request.app.state.config.RAG_OLLAMA_API_KEY = form_data.ollama_config.key

            if form_data.azure_openai_config is not None:
                request.app.state.config.RAG_AZURE_OPENAI_BASE_URL = form_data.azure_openai_config.url
                request.app.state.config.RAG_AZURE_OPENAI_API_KEY = form_data.azure_openai_config.key
                request.app.state.config.RAG_AZURE_OPENAI_API_VERSION = form_data.azure_openai_config.version

        request.app.state.ef = get_ef(
            request.app.state.config.RAG_EMBEDDING_ENGINE,
            request.app.state.config.RAG_EMBEDDING_MODEL,
        )

        request.app.state.EMBEDDING_FUNCTION = get_embedding_function(
            request.app.state.config.RAG_EMBEDDING_ENGINE,
            request.app.state.config.RAG_EMBEDDING_MODEL,
            request.app.state.ef,
            (
                request.app.state.config.RAG_OPENAI_API_BASE_URL
                if request.app.state.config.RAG_EMBEDDING_ENGINE == 'openai'
                else (
                    request.app.state.config.RAG_OLLAMA_BASE_URL
                    if request.app.state.config.RAG_EMBEDDING_ENGINE == 'ollama'
                    else request.app.state.config.RAG_AZURE_OPENAI_BASE_URL
                )
            ),
            (
                request.app.state.config.RAG_OPENAI_API_KEY
                if request.app.state.config.RAG_EMBEDDING_ENGINE == 'openai'
                else (
                    request.app.state.config.RAG_OLLAMA_API_KEY
                    if request.app.state.config.RAG_EMBEDDING_ENGINE == 'ollama'
                    else request.app.state.config.RAG_AZURE_OPENAI_API_KEY
                )
            ),
            request.app.state.config.RAG_EMBEDDING_BATCH_SIZE,
            azure_api_version=(
                request.app.state.config.RAG_AZURE_OPENAI_API_VERSION
                if request.app.state.config.RAG_EMBEDDING_ENGINE == 'azure_openai'
                else None
            ),
            enable_async=request.app.state.config.ENABLE_ASYNC_EMBEDDING,
            concurrent_requests=request.app.state.config.RAG_EMBEDDING_CONCURRENT_REQUESTS,
        )

        return {
            'status': True,
            'RAG_EMBEDDING_ENGINE': request.app.state.config.RAG_EMBEDDING_ENGINE,
            'RAG_EMBEDDING_MODEL': request.app.state.config.RAG_EMBEDDING_MODEL,
            'RAG_EMBEDDING_BATCH_SIZE': request.app.state.config.RAG_EMBEDDING_BATCH_SIZE,
            'ENABLE_ASYNC_EMBEDDING': request.app.state.config.ENABLE_ASYNC_EMBEDDING,
            'RAG_EMBEDDING_CONCURRENT_REQUESTS': request.app.state.config.RAG_EMBEDDING_CONCURRENT_REQUESTS,
            'openai_config': {
                'url': request.app.state.config.RAG_OPENAI_API_BASE_URL,
                'key': request.app.state.config.RAG_OPENAI_API_KEY,
            },
            'ollama_config': {
                'url': request.app.state.config.RAG_OLLAMA_BASE_URL,
                'key': request.app.state.config.RAG_OLLAMA_API_KEY,
            },
            'azure_openai_config': {
                'url': request.app.state.config.RAG_AZURE_OPENAI_BASE_URL,
                'key': request.app.state.config.RAG_AZURE_OPENAI_API_KEY,
                'version': request.app.state.config.RAG_AZURE_OPENAI_API_VERSION,
            },
        }
    except Exception as e:
        log.exception(f'Problem updating embedding model: {e}')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )


@router.get('/config')
async def get_rag_config(request: Request, user=Depends(get_admin_user)):
    return {
        'status': True,
        # RAG settings
        'RAG_TEMPLATE': request.app.state.config.RAG_TEMPLATE,
        'TOP_K': request.app.state.config.TOP_K,
        'BYPASS_EMBEDDING_AND_RETRIEVAL': request.app.state.config.BYPASS_EMBEDDING_AND_RETRIEVAL,
        'RAG_FULL_CONTEXT': request.app.state.config.RAG_FULL_CONTEXT,
        # Hybrid search settings
        'ENABLE_RAG_HYBRID_SEARCH': request.app.state.config.ENABLE_RAG_HYBRID_SEARCH,
        'ENABLE_RAG_HYBRID_SEARCH_ENRICHED_TEXTS': request.app.state.config.ENABLE_RAG_HYBRID_SEARCH_ENRICHED_TEXTS,
        'TOP_K_RERANKER': request.app.state.config.TOP_K_RERANKER,
        'RELEVANCE_THRESHOLD': request.app.state.config.RELEVANCE_THRESHOLD,
        'HYBRID_BM25_WEIGHT': request.app.state.config.HYBRID_BM25_WEIGHT,
        # Content extraction settings
        'CONTENT_EXTRACTION_ENGINE': request.app.state.config.CONTENT_EXTRACTION_ENGINE,
        'PDF_EXTRACT_IMAGES': request.app.state.config.PDF_EXTRACT_IMAGES,
        'PDF_LOADER_MODE': request.app.state.config.PDF_LOADER_MODE,
        'DATALAB_MARKER_API_KEY': request.app.state.config.DATALAB_MARKER_API_KEY,
        'DATALAB_MARKER_API_BASE_URL': request.app.state.config.DATALAB_MARKER_API_BASE_URL,
        'DATALAB_MARKER_ADDITIONAL_CONFIG': request.app.state.config.DATALAB_MARKER_ADDITIONAL_CONFIG,
        'DATALAB_MARKER_SKIP_CACHE': request.app.state.config.DATALAB_MARKER_SKIP_CACHE,
        'DATALAB_MARKER_FORCE_OCR': request.app.state.config.DATALAB_MARKER_FORCE_OCR,
        'DATALAB_MARKER_PAGINATE': request.app.state.config.DATALAB_MARKER_PAGINATE,
        'DATALAB_MARKER_STRIP_EXISTING_OCR': request.app.state.config.DATALAB_MARKER_STRIP_EXISTING_OCR,
        'DATALAB_MARKER_DISABLE_IMAGE_EXTRACTION': request.app.state.config.DATALAB_MARKER_DISABLE_IMAGE_EXTRACTION,
        'DATALAB_MARKER_FORMAT_LINES': request.app.state.config.DATALAB_MARKER_FORMAT_LINES,
        'DATALAB_MARKER_USE_LLM': request.app.state.config.DATALAB_MARKER_USE_LLM,
        'DATALAB_MARKER_OUTPUT_FORMAT': request.app.state.config.DATALAB_MARKER_OUTPUT_FORMAT,
        'EXTERNAL_DOCUMENT_LOADER_URL': request.app.state.config.EXTERNAL_DOCUMENT_LOADER_URL,
        'EXTERNAL_DOCUMENT_LOADER_API_KEY': request.app.state.config.EXTERNAL_DOCUMENT_LOADER_API_KEY,
        'TIKA_SERVER_URL': request.app.state.config.TIKA_SERVER_URL,
        'DOCLING_SERVER_URL': request.app.state.config.DOCLING_SERVER_URL,
        'DOCLING_API_KEY': request.app.state.config.DOCLING_API_KEY,
        'DOCLING_PARAMS': request.app.state.config.DOCLING_PARAMS,
        'DOCUMENT_INTELLIGENCE_ENDPOINT': request.app.state.config.DOCUMENT_INTELLIGENCE_ENDPOINT,
        'DOCUMENT_INTELLIGENCE_KEY': request.app.state.config.DOCUMENT_INTELLIGENCE_KEY,
        'DOCUMENT_INTELLIGENCE_MODEL': request.app.state.config.DOCUMENT_INTELLIGENCE_MODEL,
        'MISTRAL_OCR_API_BASE_URL': request.app.state.config.MISTRAL_OCR_API_BASE_URL,
        'MISTRAL_OCR_API_KEY': request.app.state.config.MISTRAL_OCR_API_KEY,
        'PADDLEOCR_VL_BASE_URL': request.app.state.config.PADDLEOCR_VL_BASE_URL,
        'PADDLEOCR_VL_TOKEN': request.app.state.config.PADDLEOCR_VL_TOKEN,
        # MinerU settings
        'MINERU_API_MODE': request.app.state.config.MINERU_API_MODE,
        'MINERU_API_URL': request.app.state.config.MINERU_API_URL,
        'MINERU_API_KEY': request.app.state.config.MINERU_API_KEY,
        'MINERU_API_TIMEOUT': request.app.state.config.MINERU_API_TIMEOUT,
        'MINERU_PARAMS': request.app.state.config.MINERU_PARAMS,
        'MINERU_FILE_EXTENSIONS': request.app.state.config.MINERU_FILE_EXTENSIONS,
        # Reranking settings
        'RAG_RERANKING_MODEL': request.app.state.config.RAG_RERANKING_MODEL,
        'RAG_RERANKING_ENGINE': request.app.state.config.RAG_RERANKING_ENGINE,
        'RAG_RERANKING_BATCH_SIZE': request.app.state.config.RAG_RERANKING_BATCH_SIZE,
        'RAG_EXTERNAL_RERANKER_URL': request.app.state.config.RAG_EXTERNAL_RERANKER_URL,
        'RAG_EXTERNAL_RERANKER_API_KEY': request.app.state.config.RAG_EXTERNAL_RERANKER_API_KEY,
        'RAG_EXTERNAL_RERANKER_TIMEOUT': request.app.state.config.RAG_EXTERNAL_RERANKER_TIMEOUT,
        # Chunking settings
        'TEXT_SPLITTER': request.app.state.config.TEXT_SPLITTER,
        'ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER': request.app.state.config.ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER,
        'CHUNK_SIZE': request.app.state.config.CHUNK_SIZE,
        'CHUNK_MIN_SIZE_TARGET': request.app.state.config.CHUNK_MIN_SIZE_TARGET,
        'CHUNK_OVERLAP': request.app.state.config.CHUNK_OVERLAP,
        # File upload settings
        'FILE_MAX_SIZE': request.app.state.config.FILE_MAX_SIZE,
        'FILE_MAX_COUNT': request.app.state.config.FILE_MAX_COUNT,
        'FILE_IMAGE_COMPRESSION_WIDTH': request.app.state.config.FILE_IMAGE_COMPRESSION_WIDTH,
        'FILE_IMAGE_COMPRESSION_HEIGHT': request.app.state.config.FILE_IMAGE_COMPRESSION_HEIGHT,
        'ALLOWED_FILE_EXTENSIONS': request.app.state.config.ALLOWED_FILE_EXTENSIONS,
        # Integration settings
        'ENABLE_GOOGLE_DRIVE_INTEGRATION': request.app.state.config.ENABLE_GOOGLE_DRIVE_INTEGRATION,
        'ENABLE_ONEDRIVE_INTEGRATION': request.app.state.config.ENABLE_ONEDRIVE_INTEGRATION,
        # Web search settings. This fork only ships Kagi as a search engine, so
        # other providers' API keys are intentionally absent. Loader-related keys
        # (firecrawl/tavily/external) are kept because they're still selectable
        # web *loader* engines, distinct from search.
        'web': {
            'ENABLE_WEB_SEARCH': request.app.state.config.ENABLE_WEB_SEARCH,
            'WEB_SEARCH_ENGINE': request.app.state.config.WEB_SEARCH_ENGINE,
            'WEB_SEARCH_TRUST_ENV': request.app.state.config.WEB_SEARCH_TRUST_ENV,
            'WEB_SEARCH_RESULT_COUNT': request.app.state.config.WEB_SEARCH_RESULT_COUNT,
            'WEB_SEARCH_CONCURRENT_REQUESTS': request.app.state.config.WEB_SEARCH_CONCURRENT_REQUESTS,
            'WEB_FETCH_MAX_CONTENT_LENGTH': request.app.state.config.WEB_FETCH_MAX_CONTENT_LENGTH,
            'WEB_LOADER_CONCURRENT_REQUESTS': request.app.state.config.WEB_LOADER_CONCURRENT_REQUESTS,
            'WEB_SEARCH_DOMAIN_FILTER_LIST': request.app.state.config.WEB_SEARCH_DOMAIN_FILTER_LIST,
            'BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL': request.app.state.config.BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL,
            'BYPASS_WEB_SEARCH_WEB_LOADER': request.app.state.config.BYPASS_WEB_SEARCH_WEB_LOADER,
            'KAGI_SEARCH_API_KEY': request.app.state.config.KAGI_SEARCH_API_KEY,
            'WEB_LOADER_ENGINE': request.app.state.config.WEB_LOADER_ENGINE,
            'WEB_LOADER_TIMEOUT': request.app.state.config.WEB_LOADER_TIMEOUT,
            'ENABLE_WEB_LOADER_SSL_VERIFICATION': request.app.state.config.ENABLE_WEB_LOADER_SSL_VERIFICATION,
            'PLAYWRIGHT_WS_URL': request.app.state.config.PLAYWRIGHT_WS_URL,
            'PLAYWRIGHT_TIMEOUT': request.app.state.config.PLAYWRIGHT_TIMEOUT,
            'FIRECRAWL_API_KEY': request.app.state.config.FIRECRAWL_API_KEY,
            'FIRECRAWL_API_BASE_URL': request.app.state.config.FIRECRAWL_API_BASE_URL,
            'FIRECRAWL_TIMEOUT': request.app.state.config.FIRECRAWL_TIMEOUT,
            'TAVILY_API_KEY': request.app.state.config.TAVILY_API_KEY,
            'TAVILY_EXTRACT_DEPTH': request.app.state.config.TAVILY_EXTRACT_DEPTH,
            'EXTERNAL_WEB_LOADER_URL': request.app.state.config.EXTERNAL_WEB_LOADER_URL,
            'EXTERNAL_WEB_LOADER_API_KEY': request.app.state.config.EXTERNAL_WEB_LOADER_API_KEY,
            'YOUTUBE_LOADER_LANGUAGE': request.app.state.config.YOUTUBE_LOADER_LANGUAGE,
            'YOUTUBE_LOADER_PROXY_URL': request.app.state.config.YOUTUBE_LOADER_PROXY_URL,
            'YOUTUBE_LOADER_TRANSLATION': request.app.state.YOUTUBE_LOADER_TRANSLATION,
        },
    }


class WebConfig(BaseModel):
    ENABLE_WEB_SEARCH: bool | None = None
    WEB_SEARCH_ENGINE: str | None = None
    WEB_SEARCH_TRUST_ENV: bool | None = None
    WEB_SEARCH_RESULT_COUNT: int | None = None
    WEB_SEARCH_CONCURRENT_REQUESTS: int | None = None
    WEB_SEARCH_DOMAIN_FILTER_LIST: list[str | None] = []
    WEB_FETCH_MAX_CONTENT_LENGTH: int | None = None
    WEB_LOADER_CONCURRENT_REQUESTS: int | None = None
    BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL: bool | None = None
    BYPASS_WEB_SEARCH_WEB_LOADER: bool | None = None
    KAGI_SEARCH_API_KEY: str | None = None
    WEB_LOADER_ENGINE: str | None = None
    WEB_LOADER_TIMEOUT: str | None = None
    ENABLE_WEB_LOADER_SSL_VERIFICATION: bool | None = None
    PLAYWRIGHT_WS_URL: str | None = None
    PLAYWRIGHT_TIMEOUT: int | None = None
    FIRECRAWL_API_KEY: str | None = None
    FIRECRAWL_API_BASE_URL: str | None = None
    FIRECRAWL_TIMEOUT: str | None = None
    TAVILY_API_KEY: str | None = None
    TAVILY_EXTRACT_DEPTH: str | None = None
    EXTERNAL_WEB_LOADER_URL: str | None = None
    EXTERNAL_WEB_LOADER_API_KEY: str | None = None
    YOUTUBE_LOADER_LANGUAGE: list[str | None] = None
    YOUTUBE_LOADER_PROXY_URL: str | None = None
    YOUTUBE_LOADER_TRANSLATION: str | None = None


class ConfigForm(BaseModel):
    # RAG settings
    RAG_TEMPLATE: str | None = None
    TOP_K: int | None = None
    BYPASS_EMBEDDING_AND_RETRIEVAL: bool | None = None
    RAG_FULL_CONTEXT: bool | None = None

    # Hybrid search settings
    ENABLE_RAG_HYBRID_SEARCH: bool | None = None
    ENABLE_RAG_HYBRID_SEARCH_ENRICHED_TEXTS: bool | None = None
    TOP_K_RERANKER: int | None = None
    RELEVANCE_THRESHOLD: float | None = None
    HYBRID_BM25_WEIGHT: float | None = None

    # Content extraction settings
    CONTENT_EXTRACTION_ENGINE: str | None = None
    PDF_EXTRACT_IMAGES: bool | None = None
    PDF_LOADER_MODE: str | None = None

    DATALAB_MARKER_API_KEY: str | None = None
    DATALAB_MARKER_API_BASE_URL: str | None = None
    DATALAB_MARKER_ADDITIONAL_CONFIG: str | None = None
    DATALAB_MARKER_SKIP_CACHE: bool | None = None
    DATALAB_MARKER_FORCE_OCR: bool | None = None
    DATALAB_MARKER_PAGINATE: bool | None = None
    DATALAB_MARKER_STRIP_EXISTING_OCR: bool | None = None
    DATALAB_MARKER_DISABLE_IMAGE_EXTRACTION: bool | None = None
    DATALAB_MARKER_FORMAT_LINES: bool | None = None
    DATALAB_MARKER_USE_LLM: bool | None = None
    DATALAB_MARKER_OUTPUT_FORMAT: str | None = None

    EXTERNAL_DOCUMENT_LOADER_URL: str | None = None
    EXTERNAL_DOCUMENT_LOADER_API_KEY: str | None = None

    TIKA_SERVER_URL: str | None = None
    DOCLING_SERVER_URL: str | None = None
    DOCLING_API_KEY: str | None = None
    DOCLING_PARAMS: dict | None = None
    DOCUMENT_INTELLIGENCE_ENDPOINT: str | None = None
    DOCUMENT_INTELLIGENCE_KEY: str | None = None
    DOCUMENT_INTELLIGENCE_MODEL: str | None = None
    MISTRAL_OCR_API_BASE_URL: str | None = None
    MISTRAL_OCR_API_KEY: str | None = None
    PADDLEOCR_VL_BASE_URL: str | None = None
    PADDLEOCR_VL_TOKEN: str | None = None

    # MinerU settings
    MINERU_API_MODE: str | None = None
    MINERU_API_URL: str | None = None
    MINERU_API_KEY: str | None = None
    MINERU_API_TIMEOUT: str | None = None
    MINERU_PARAMS: dict | None = None
    MINERU_FILE_EXTENSIONS: list[str] | None = None

    # Reranking settings
    RAG_RERANKING_MODEL: str | None = None
    RAG_RERANKING_ENGINE: str | None = None
    RAG_RERANKING_BATCH_SIZE: int | None = None
    RAG_EXTERNAL_RERANKER_URL: str | None = None
    RAG_EXTERNAL_RERANKER_API_KEY: str | None = None
    RAG_EXTERNAL_RERANKER_TIMEOUT: str | None = None

    # Chunking settings
    TEXT_SPLITTER: str | None = None
    ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER: bool | None = None
    CHUNK_SIZE: int | None = None
    CHUNK_MIN_SIZE_TARGET: int | None = None
    CHUNK_OVERLAP: int | None = None

    # File upload settings
    FILE_MAX_SIZE: Union[int, str | None] = None
    FILE_MAX_COUNT: Union[int, str | None] = None
    FILE_IMAGE_COMPRESSION_WIDTH: Union[int, str | None] = None
    FILE_IMAGE_COMPRESSION_HEIGHT: Union[int, str | None] = None
    ALLOWED_FILE_EXTENSIONS: list[str | None] = None

    # Integration settings
    ENABLE_GOOGLE_DRIVE_INTEGRATION: bool | None = None
    ENABLE_ONEDRIVE_INTEGRATION: bool | None = None

    # Web search settings
    web: WebConfig | None = None


@router.post('/config/update')
async def update_rag_config(request: Request, form_data: ConfigForm, user=Depends(get_admin_user)):
    # RAG settings
    request.app.state.config.RAG_TEMPLATE = (
        form_data.RAG_TEMPLATE if form_data.RAG_TEMPLATE is not None else request.app.state.config.RAG_TEMPLATE
    )
    request.app.state.config.TOP_K = form_data.TOP_K if form_data.TOP_K is not None else request.app.state.config.TOP_K
    request.app.state.config.BYPASS_EMBEDDING_AND_RETRIEVAL = (
        form_data.BYPASS_EMBEDDING_AND_RETRIEVAL
        if form_data.BYPASS_EMBEDDING_AND_RETRIEVAL is not None
        else request.app.state.config.BYPASS_EMBEDDING_AND_RETRIEVAL
    )
    request.app.state.config.RAG_FULL_CONTEXT = (
        form_data.RAG_FULL_CONTEXT
        if form_data.RAG_FULL_CONTEXT is not None
        else request.app.state.config.RAG_FULL_CONTEXT
    )

    # Hybrid search settings
    request.app.state.config.ENABLE_RAG_HYBRID_SEARCH = (
        form_data.ENABLE_RAG_HYBRID_SEARCH
        if form_data.ENABLE_RAG_HYBRID_SEARCH is not None
        else request.app.state.config.ENABLE_RAG_HYBRID_SEARCH
    )
    request.app.state.config.ENABLE_RAG_HYBRID_SEARCH_ENRICHED_TEXTS = (
        form_data.ENABLE_RAG_HYBRID_SEARCH_ENRICHED_TEXTS
        if form_data.ENABLE_RAG_HYBRID_SEARCH_ENRICHED_TEXTS is not None
        else request.app.state.config.ENABLE_RAG_HYBRID_SEARCH_ENRICHED_TEXTS
    )

    request.app.state.config.TOP_K_RERANKER = (
        form_data.TOP_K_RERANKER if form_data.TOP_K_RERANKER is not None else request.app.state.config.TOP_K_RERANKER
    )
    request.app.state.config.RELEVANCE_THRESHOLD = (
        form_data.RELEVANCE_THRESHOLD
        if form_data.RELEVANCE_THRESHOLD is not None
        else request.app.state.config.RELEVANCE_THRESHOLD
    )
    request.app.state.config.HYBRID_BM25_WEIGHT = (
        form_data.HYBRID_BM25_WEIGHT
        if form_data.HYBRID_BM25_WEIGHT is not None
        else request.app.state.config.HYBRID_BM25_WEIGHT
    )

    # Content extraction settings
    request.app.state.config.CONTENT_EXTRACTION_ENGINE = (
        form_data.CONTENT_EXTRACTION_ENGINE
        if form_data.CONTENT_EXTRACTION_ENGINE is not None
        else request.app.state.config.CONTENT_EXTRACTION_ENGINE
    )
    request.app.state.config.PDF_EXTRACT_IMAGES = (
        form_data.PDF_EXTRACT_IMAGES
        if form_data.PDF_EXTRACT_IMAGES is not None
        else request.app.state.config.PDF_EXTRACT_IMAGES
    )
    request.app.state.config.PDF_LOADER_MODE = (
        form_data.PDF_LOADER_MODE if form_data.PDF_LOADER_MODE is not None else request.app.state.config.PDF_LOADER_MODE
    )
    request.app.state.config.DATALAB_MARKER_API_KEY = (
        form_data.DATALAB_MARKER_API_KEY
        if form_data.DATALAB_MARKER_API_KEY is not None
        else request.app.state.config.DATALAB_MARKER_API_KEY
    )
    request.app.state.config.DATALAB_MARKER_API_BASE_URL = (
        form_data.DATALAB_MARKER_API_BASE_URL
        if form_data.DATALAB_MARKER_API_BASE_URL is not None
        else request.app.state.config.DATALAB_MARKER_API_BASE_URL
    )
    request.app.state.config.DATALAB_MARKER_ADDITIONAL_CONFIG = (
        form_data.DATALAB_MARKER_ADDITIONAL_CONFIG
        if form_data.DATALAB_MARKER_ADDITIONAL_CONFIG is not None
        else request.app.state.config.DATALAB_MARKER_ADDITIONAL_CONFIG
    )
    request.app.state.config.DATALAB_MARKER_SKIP_CACHE = (
        form_data.DATALAB_MARKER_SKIP_CACHE
        if form_data.DATALAB_MARKER_SKIP_CACHE is not None
        else request.app.state.config.DATALAB_MARKER_SKIP_CACHE
    )
    request.app.state.config.DATALAB_MARKER_FORCE_OCR = (
        form_data.DATALAB_MARKER_FORCE_OCR
        if form_data.DATALAB_MARKER_FORCE_OCR is not None
        else request.app.state.config.DATALAB_MARKER_FORCE_OCR
    )
    request.app.state.config.DATALAB_MARKER_PAGINATE = (
        form_data.DATALAB_MARKER_PAGINATE
        if form_data.DATALAB_MARKER_PAGINATE is not None
        else request.app.state.config.DATALAB_MARKER_PAGINATE
    )
    request.app.state.config.DATALAB_MARKER_STRIP_EXISTING_OCR = (
        form_data.DATALAB_MARKER_STRIP_EXISTING_OCR
        if form_data.DATALAB_MARKER_STRIP_EXISTING_OCR is not None
        else request.app.state.config.DATALAB_MARKER_STRIP_EXISTING_OCR
    )
    request.app.state.config.DATALAB_MARKER_DISABLE_IMAGE_EXTRACTION = (
        form_data.DATALAB_MARKER_DISABLE_IMAGE_EXTRACTION
        if form_data.DATALAB_MARKER_DISABLE_IMAGE_EXTRACTION is not None
        else request.app.state.config.DATALAB_MARKER_DISABLE_IMAGE_EXTRACTION
    )
    request.app.state.config.DATALAB_MARKER_FORMAT_LINES = (
        form_data.DATALAB_MARKER_FORMAT_LINES
        if form_data.DATALAB_MARKER_FORMAT_LINES is not None
        else request.app.state.config.DATALAB_MARKER_FORMAT_LINES
    )
    request.app.state.config.DATALAB_MARKER_OUTPUT_FORMAT = (
        form_data.DATALAB_MARKER_OUTPUT_FORMAT
        if form_data.DATALAB_MARKER_OUTPUT_FORMAT is not None
        else request.app.state.config.DATALAB_MARKER_OUTPUT_FORMAT
    )
    request.app.state.config.DATALAB_MARKER_USE_LLM = (
        form_data.DATALAB_MARKER_USE_LLM
        if form_data.DATALAB_MARKER_USE_LLM is not None
        else request.app.state.config.DATALAB_MARKER_USE_LLM
    )
    request.app.state.config.EXTERNAL_DOCUMENT_LOADER_URL = (
        form_data.EXTERNAL_DOCUMENT_LOADER_URL
        if form_data.EXTERNAL_DOCUMENT_LOADER_URL is not None
        else request.app.state.config.EXTERNAL_DOCUMENT_LOADER_URL
    )
    request.app.state.config.EXTERNAL_DOCUMENT_LOADER_API_KEY = (
        form_data.EXTERNAL_DOCUMENT_LOADER_API_KEY
        if form_data.EXTERNAL_DOCUMENT_LOADER_API_KEY is not None
        else request.app.state.config.EXTERNAL_DOCUMENT_LOADER_API_KEY
    )
    request.app.state.config.TIKA_SERVER_URL = (
        form_data.TIKA_SERVER_URL if form_data.TIKA_SERVER_URL is not None else request.app.state.config.TIKA_SERVER_URL
    )
    request.app.state.config.DOCLING_SERVER_URL = (
        form_data.DOCLING_SERVER_URL
        if form_data.DOCLING_SERVER_URL is not None
        else request.app.state.config.DOCLING_SERVER_URL
    )
    request.app.state.config.DOCLING_API_KEY = (
        form_data.DOCLING_API_KEY if form_data.DOCLING_API_KEY is not None else request.app.state.config.DOCLING_API_KEY
    )
    request.app.state.config.DOCLING_PARAMS = (
        form_data.DOCLING_PARAMS if form_data.DOCLING_PARAMS is not None else request.app.state.config.DOCLING_PARAMS
    )
    request.app.state.config.DOCUMENT_INTELLIGENCE_ENDPOINT = (
        form_data.DOCUMENT_INTELLIGENCE_ENDPOINT
        if form_data.DOCUMENT_INTELLIGENCE_ENDPOINT is not None
        else request.app.state.config.DOCUMENT_INTELLIGENCE_ENDPOINT
    )
    request.app.state.config.DOCUMENT_INTELLIGENCE_KEY = (
        form_data.DOCUMENT_INTELLIGENCE_KEY
        if form_data.DOCUMENT_INTELLIGENCE_KEY is not None
        else request.app.state.config.DOCUMENT_INTELLIGENCE_KEY
    )
    request.app.state.config.DOCUMENT_INTELLIGENCE_MODEL = (
        form_data.DOCUMENT_INTELLIGENCE_MODEL
        if form_data.DOCUMENT_INTELLIGENCE_MODEL is not None
        else request.app.state.config.DOCUMENT_INTELLIGENCE_MODEL
    )

    request.app.state.config.MISTRAL_OCR_API_BASE_URL = (
        form_data.MISTRAL_OCR_API_BASE_URL
        if form_data.MISTRAL_OCR_API_BASE_URL is not None
        else request.app.state.config.MISTRAL_OCR_API_BASE_URL
    )
    request.app.state.config.MISTRAL_OCR_API_KEY = (
        form_data.MISTRAL_OCR_API_KEY
        if form_data.MISTRAL_OCR_API_KEY is not None
        else request.app.state.config.MISTRAL_OCR_API_KEY
    )
    request.app.state.config.PADDLEOCR_VL_BASE_URL = (
        form_data.PADDLEOCR_VL_BASE_URL
        if form_data.PADDLEOCR_VL_BASE_URL is not None
        else request.app.state.config.PADDLEOCR_VL_BASE_URL
    )
    request.app.state.config.PADDLEOCR_VL_TOKEN = (
        form_data.PADDLEOCR_VL_TOKEN
        if form_data.PADDLEOCR_VL_TOKEN is not None
        else request.app.state.config.PADDLEOCR_VL_TOKEN
    )

    # MinerU settings
    request.app.state.config.MINERU_API_MODE = (
        form_data.MINERU_API_MODE if form_data.MINERU_API_MODE is not None else request.app.state.config.MINERU_API_MODE
    )
    request.app.state.config.MINERU_API_URL = (
        form_data.MINERU_API_URL if form_data.MINERU_API_URL is not None else request.app.state.config.MINERU_API_URL
    )
    request.app.state.config.MINERU_API_KEY = (
        form_data.MINERU_API_KEY if form_data.MINERU_API_KEY is not None else request.app.state.config.MINERU_API_KEY
    )
    request.app.state.config.MINERU_API_TIMEOUT = (
        form_data.MINERU_API_TIMEOUT
        if form_data.MINERU_API_TIMEOUT is not None
        else request.app.state.config.MINERU_API_TIMEOUT
    )
    request.app.state.config.MINERU_PARAMS = (
        form_data.MINERU_PARAMS if form_data.MINERU_PARAMS is not None else request.app.state.config.MINERU_PARAMS
    )
    request.app.state.config.MINERU_FILE_EXTENSIONS = (
        form_data.MINERU_FILE_EXTENSIONS
        if form_data.MINERU_FILE_EXTENSIONS is not None
        else request.app.state.config.MINERU_FILE_EXTENSIONS
    )

    # Reranking settings
    if request.app.state.config.RAG_RERANKING_ENGINE == '':
        # Unloading the internal reranker and clear VRAM memory
        request.app.state.rf = None
        request.app.state.RERANKING_FUNCTION = None
        import gc

        gc.collect()
        if DEVICE_TYPE == 'cuda':
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    request.app.state.config.RAG_RERANKING_ENGINE = (
        form_data.RAG_RERANKING_ENGINE
        if form_data.RAG_RERANKING_ENGINE is not None
        else request.app.state.config.RAG_RERANKING_ENGINE
    )

    request.app.state.config.RAG_EXTERNAL_RERANKER_URL = (
        form_data.RAG_EXTERNAL_RERANKER_URL
        if form_data.RAG_EXTERNAL_RERANKER_URL is not None
        else request.app.state.config.RAG_EXTERNAL_RERANKER_URL
    )

    request.app.state.config.RAG_EXTERNAL_RERANKER_API_KEY = (
        form_data.RAG_EXTERNAL_RERANKER_API_KEY
        if form_data.RAG_EXTERNAL_RERANKER_API_KEY is not None
        else request.app.state.config.RAG_EXTERNAL_RERANKER_API_KEY
    )

    request.app.state.config.RAG_EXTERNAL_RERANKER_TIMEOUT = (
        form_data.RAG_EXTERNAL_RERANKER_TIMEOUT
        if form_data.RAG_EXTERNAL_RERANKER_TIMEOUT is not None
        else request.app.state.config.RAG_EXTERNAL_RERANKER_TIMEOUT
    )

    request.app.state.config.RAG_RERANKING_BATCH_SIZE = (
        form_data.RAG_RERANKING_BATCH_SIZE
        if form_data.RAG_RERANKING_BATCH_SIZE is not None
        else request.app.state.config.RAG_RERANKING_BATCH_SIZE
    )

    log.info(
        f'Updating reranking model: {request.app.state.config.RAG_RERANKING_MODEL} to {form_data.RAG_RERANKING_MODEL}'
    )
    try:
        request.app.state.config.RAG_RERANKING_MODEL = (
            form_data.RAG_RERANKING_MODEL
            if form_data.RAG_RERANKING_MODEL is not None
            else request.app.state.config.RAG_RERANKING_MODEL
        )

        try:
            if (
                request.app.state.config.ENABLE_RAG_HYBRID_SEARCH
                and not request.app.state.config.BYPASS_EMBEDDING_AND_RETRIEVAL
            ):
                request.app.state.rf = get_rf(
                    request.app.state.config.RAG_RERANKING_ENGINE,
                    request.app.state.config.RAG_RERANKING_MODEL,
                    request.app.state.config.RAG_EXTERNAL_RERANKER_URL,
                    request.app.state.config.RAG_EXTERNAL_RERANKER_API_KEY,
                    request.app.state.config.RAG_EXTERNAL_RERANKER_TIMEOUT,
                )

                request.app.state.RERANKING_FUNCTION = get_reranking_function(
                    request.app.state.config.RAG_RERANKING_ENGINE,
                    request.app.state.config.RAG_RERANKING_MODEL,
                    request.app.state.rf,
                    reranking_batch_size=request.app.state.config.RAG_RERANKING_BATCH_SIZE,
                )
        except Exception as e:
            log.error(f'Error loading reranking model: {e}')
            request.app.state.config.ENABLE_RAG_HYBRID_SEARCH = False
    except Exception as e:
        log.exception(f'Problem updating reranking model: {e}')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )

    # Chunking settings
    request.app.state.config.TEXT_SPLITTER = (
        form_data.TEXT_SPLITTER if form_data.TEXT_SPLITTER is not None else request.app.state.config.TEXT_SPLITTER
    )
    request.app.state.config.ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER = (
        form_data.ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER
        if form_data.ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER is not None
        else request.app.state.config.ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER
    )
    request.app.state.config.CHUNK_SIZE = (
        form_data.CHUNK_SIZE if form_data.CHUNK_SIZE is not None else request.app.state.config.CHUNK_SIZE
    )
    request.app.state.config.CHUNK_MIN_SIZE_TARGET = (
        form_data.CHUNK_MIN_SIZE_TARGET
        if form_data.CHUNK_MIN_SIZE_TARGET is not None
        else request.app.state.config.CHUNK_MIN_SIZE_TARGET
    )
    request.app.state.config.CHUNK_OVERLAP = (
        form_data.CHUNK_OVERLAP if form_data.CHUNK_OVERLAP is not None else request.app.state.config.CHUNK_OVERLAP
    )

    # File upload settings
    # Empty string means "clear to None" (unlimited/no compression),
    # None means "don't change", int means "set to this value"
    if form_data.FILE_MAX_SIZE is not None:
        request.app.state.config.FILE_MAX_SIZE = None if form_data.FILE_MAX_SIZE == '' else form_data.FILE_MAX_SIZE
    if form_data.FILE_MAX_COUNT is not None:
        request.app.state.config.FILE_MAX_COUNT = None if form_data.FILE_MAX_COUNT == '' else form_data.FILE_MAX_COUNT
    if form_data.FILE_IMAGE_COMPRESSION_WIDTH is not None:
        request.app.state.config.FILE_IMAGE_COMPRESSION_WIDTH = (
            None if form_data.FILE_IMAGE_COMPRESSION_WIDTH == '' else form_data.FILE_IMAGE_COMPRESSION_WIDTH
        )
    if form_data.FILE_IMAGE_COMPRESSION_HEIGHT is not None:
        request.app.state.config.FILE_IMAGE_COMPRESSION_HEIGHT = (
            None if form_data.FILE_IMAGE_COMPRESSION_HEIGHT == '' else form_data.FILE_IMAGE_COMPRESSION_HEIGHT
        )

    request.app.state.config.ALLOWED_FILE_EXTENSIONS = (
        form_data.ALLOWED_FILE_EXTENSIONS
        if form_data.ALLOWED_FILE_EXTENSIONS is not None
        else request.app.state.config.ALLOWED_FILE_EXTENSIONS
    )

    # Integration settings
    request.app.state.config.ENABLE_GOOGLE_DRIVE_INTEGRATION = (
        form_data.ENABLE_GOOGLE_DRIVE_INTEGRATION
        if form_data.ENABLE_GOOGLE_DRIVE_INTEGRATION is not None
        else request.app.state.config.ENABLE_GOOGLE_DRIVE_INTEGRATION
    )
    request.app.state.config.ENABLE_ONEDRIVE_INTEGRATION = (
        form_data.ENABLE_ONEDRIVE_INTEGRATION
        if form_data.ENABLE_ONEDRIVE_INTEGRATION is not None
        else request.app.state.config.ENABLE_ONEDRIVE_INTEGRATION
    )

    if form_data.web is not None:
        # Web search settings — Kagi-only fork.
        request.app.state.config.ENABLE_WEB_SEARCH = form_data.web.ENABLE_WEB_SEARCH
        request.app.state.config.WEB_SEARCH_ENGINE = form_data.web.WEB_SEARCH_ENGINE
        request.app.state.config.WEB_SEARCH_TRUST_ENV = form_data.web.WEB_SEARCH_TRUST_ENV
        request.app.state.config.WEB_SEARCH_RESULT_COUNT = form_data.web.WEB_SEARCH_RESULT_COUNT
        request.app.state.config.WEB_SEARCH_CONCURRENT_REQUESTS = form_data.web.WEB_SEARCH_CONCURRENT_REQUESTS
        request.app.state.config.WEB_FETCH_MAX_CONTENT_LENGTH = form_data.web.WEB_FETCH_MAX_CONTENT_LENGTH
        request.app.state.config.WEB_LOADER_CONCURRENT_REQUESTS = form_data.web.WEB_LOADER_CONCURRENT_REQUESTS
        request.app.state.config.WEB_SEARCH_DOMAIN_FILTER_LIST = form_data.web.WEB_SEARCH_DOMAIN_FILTER_LIST
        request.app.state.config.BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL = (
            form_data.web.BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL
        )
        request.app.state.config.BYPASS_WEB_SEARCH_WEB_LOADER = form_data.web.BYPASS_WEB_SEARCH_WEB_LOADER
        request.app.state.config.KAGI_SEARCH_API_KEY = form_data.web.KAGI_SEARCH_API_KEY

        # Web loader settings
        request.app.state.config.WEB_LOADER_ENGINE = form_data.web.WEB_LOADER_ENGINE
        request.app.state.config.WEB_LOADER_TIMEOUT = form_data.web.WEB_LOADER_TIMEOUT

        request.app.state.config.ENABLE_WEB_LOADER_SSL_VERIFICATION = form_data.web.ENABLE_WEB_LOADER_SSL_VERIFICATION
        request.app.state.config.PLAYWRIGHT_WS_URL = form_data.web.PLAYWRIGHT_WS_URL
        request.app.state.config.PLAYWRIGHT_TIMEOUT = form_data.web.PLAYWRIGHT_TIMEOUT
        request.app.state.config.FIRECRAWL_API_KEY = form_data.web.FIRECRAWL_API_KEY
        request.app.state.config.FIRECRAWL_API_BASE_URL = form_data.web.FIRECRAWL_API_BASE_URL
        request.app.state.config.FIRECRAWL_TIMEOUT = form_data.web.FIRECRAWL_TIMEOUT
        request.app.state.config.TAVILY_API_KEY = form_data.web.TAVILY_API_KEY
        request.app.state.config.TAVILY_EXTRACT_DEPTH = form_data.web.TAVILY_EXTRACT_DEPTH
        request.app.state.config.EXTERNAL_WEB_LOADER_URL = form_data.web.EXTERNAL_WEB_LOADER_URL
        request.app.state.config.EXTERNAL_WEB_LOADER_API_KEY = form_data.web.EXTERNAL_WEB_LOADER_API_KEY
        request.app.state.config.YOUTUBE_LOADER_LANGUAGE = form_data.web.YOUTUBE_LOADER_LANGUAGE
        request.app.state.config.YOUTUBE_LOADER_PROXY_URL = form_data.web.YOUTUBE_LOADER_PROXY_URL
        request.app.state.YOUTUBE_LOADER_TRANSLATION = form_data.web.YOUTUBE_LOADER_TRANSLATION

    return {
        'status': True,
        # RAG settings
        'RAG_TEMPLATE': request.app.state.config.RAG_TEMPLATE,
        'TOP_K': request.app.state.config.TOP_K,
        'BYPASS_EMBEDDING_AND_RETRIEVAL': request.app.state.config.BYPASS_EMBEDDING_AND_RETRIEVAL,
        'RAG_FULL_CONTEXT': request.app.state.config.RAG_FULL_CONTEXT,
        # Hybrid search settings
        'ENABLE_RAG_HYBRID_SEARCH': request.app.state.config.ENABLE_RAG_HYBRID_SEARCH,
        'TOP_K_RERANKER': request.app.state.config.TOP_K_RERANKER,
        'RELEVANCE_THRESHOLD': request.app.state.config.RELEVANCE_THRESHOLD,
        'HYBRID_BM25_WEIGHT': request.app.state.config.HYBRID_BM25_WEIGHT,
        # Content extraction settings
        'CONTENT_EXTRACTION_ENGINE': request.app.state.config.CONTENT_EXTRACTION_ENGINE,
        'PDF_EXTRACT_IMAGES': request.app.state.config.PDF_EXTRACT_IMAGES,
        'PDF_LOADER_MODE': request.app.state.config.PDF_LOADER_MODE,
        'DATALAB_MARKER_API_KEY': request.app.state.config.DATALAB_MARKER_API_KEY,
        'DATALAB_MARKER_API_BASE_URL': request.app.state.config.DATALAB_MARKER_API_BASE_URL,
        'DATALAB_MARKER_ADDITIONAL_CONFIG': request.app.state.config.DATALAB_MARKER_ADDITIONAL_CONFIG,
        'DATALAB_MARKER_SKIP_CACHE': request.app.state.config.DATALAB_MARKER_SKIP_CACHE,
        'DATALAB_MARKER_FORCE_OCR': request.app.state.config.DATALAB_MARKER_FORCE_OCR,
        'DATALAB_MARKER_PAGINATE': request.app.state.config.DATALAB_MARKER_PAGINATE,
        'DATALAB_MARKER_STRIP_EXISTING_OCR': request.app.state.config.DATALAB_MARKER_STRIP_EXISTING_OCR,
        'DATALAB_MARKER_DISABLE_IMAGE_EXTRACTION': request.app.state.config.DATALAB_MARKER_DISABLE_IMAGE_EXTRACTION,
        'DATALAB_MARKER_USE_LLM': request.app.state.config.DATALAB_MARKER_USE_LLM,
        'DATALAB_MARKER_OUTPUT_FORMAT': request.app.state.config.DATALAB_MARKER_OUTPUT_FORMAT,
        'EXTERNAL_DOCUMENT_LOADER_URL': request.app.state.config.EXTERNAL_DOCUMENT_LOADER_URL,
        'EXTERNAL_DOCUMENT_LOADER_API_KEY': request.app.state.config.EXTERNAL_DOCUMENT_LOADER_API_KEY,
        'TIKA_SERVER_URL': request.app.state.config.TIKA_SERVER_URL,
        'DOCLING_SERVER_URL': request.app.state.config.DOCLING_SERVER_URL,
        'DOCLING_API_KEY': request.app.state.config.DOCLING_API_KEY,
        'DOCLING_PARAMS': request.app.state.config.DOCLING_PARAMS,
        'DOCUMENT_INTELLIGENCE_ENDPOINT': request.app.state.config.DOCUMENT_INTELLIGENCE_ENDPOINT,
        'DOCUMENT_INTELLIGENCE_KEY': request.app.state.config.DOCUMENT_INTELLIGENCE_KEY,
        'DOCUMENT_INTELLIGENCE_MODEL': request.app.state.config.DOCUMENT_INTELLIGENCE_MODEL,
        'MISTRAL_OCR_API_BASE_URL': request.app.state.config.MISTRAL_OCR_API_BASE_URL,
        'MISTRAL_OCR_API_KEY': request.app.state.config.MISTRAL_OCR_API_KEY,
        'PADDLEOCR_VL_BASE_URL': request.app.state.config.PADDLEOCR_VL_BASE_URL,
        'PADDLEOCR_VL_TOKEN': request.app.state.config.PADDLEOCR_VL_TOKEN,
        # MinerU settings
        'MINERU_API_MODE': request.app.state.config.MINERU_API_MODE,
        'MINERU_API_URL': request.app.state.config.MINERU_API_URL,
        'MINERU_API_KEY': request.app.state.config.MINERU_API_KEY,
        'MINERU_API_TIMEOUT': request.app.state.config.MINERU_API_TIMEOUT,
        'MINERU_PARAMS': request.app.state.config.MINERU_PARAMS,
        # Reranking settings
        'RAG_RERANKING_MODEL': request.app.state.config.RAG_RERANKING_MODEL,
        'RAG_RERANKING_ENGINE': request.app.state.config.RAG_RERANKING_ENGINE,
        'RAG_EXTERNAL_RERANKER_URL': request.app.state.config.RAG_EXTERNAL_RERANKER_URL,
        'RAG_EXTERNAL_RERANKER_API_KEY': request.app.state.config.RAG_EXTERNAL_RERANKER_API_KEY,
        'RAG_EXTERNAL_RERANKER_TIMEOUT': request.app.state.config.RAG_EXTERNAL_RERANKER_TIMEOUT,
        # Chunking settings
        'TEXT_SPLITTER': request.app.state.config.TEXT_SPLITTER,
        'CHUNK_SIZE': request.app.state.config.CHUNK_SIZE,
        'CHUNK_MIN_SIZE_TARGET': request.app.state.config.CHUNK_MIN_SIZE_TARGET,
        'ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER': request.app.state.config.ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER,
        'CHUNK_OVERLAP': request.app.state.config.CHUNK_OVERLAP,
        # File upload settings
        'FILE_MAX_SIZE': request.app.state.config.FILE_MAX_SIZE,
        'FILE_MAX_COUNT': request.app.state.config.FILE_MAX_COUNT,
        'FILE_IMAGE_COMPRESSION_WIDTH': request.app.state.config.FILE_IMAGE_COMPRESSION_WIDTH,
        'FILE_IMAGE_COMPRESSION_HEIGHT': request.app.state.config.FILE_IMAGE_COMPRESSION_HEIGHT,
        'ALLOWED_FILE_EXTENSIONS': request.app.state.config.ALLOWED_FILE_EXTENSIONS,
        # Integration settings
        'ENABLE_GOOGLE_DRIVE_INTEGRATION': request.app.state.config.ENABLE_GOOGLE_DRIVE_INTEGRATION,
        'ENABLE_ONEDRIVE_INTEGRATION': request.app.state.config.ENABLE_ONEDRIVE_INTEGRATION,
        # Web search settings — Kagi-only fork. See the GET /config note above.
        'web': {
            'ENABLE_WEB_SEARCH': request.app.state.config.ENABLE_WEB_SEARCH,
            'WEB_SEARCH_ENGINE': request.app.state.config.WEB_SEARCH_ENGINE,
            'WEB_SEARCH_TRUST_ENV': request.app.state.config.WEB_SEARCH_TRUST_ENV,
            'WEB_SEARCH_RESULT_COUNT': request.app.state.config.WEB_SEARCH_RESULT_COUNT,
            'WEB_SEARCH_CONCURRENT_REQUESTS': request.app.state.config.WEB_SEARCH_CONCURRENT_REQUESTS,
            'WEB_FETCH_MAX_CONTENT_LENGTH': request.app.state.config.WEB_FETCH_MAX_CONTENT_LENGTH,
            'WEB_LOADER_CONCURRENT_REQUESTS': request.app.state.config.WEB_LOADER_CONCURRENT_REQUESTS,
            'WEB_SEARCH_DOMAIN_FILTER_LIST': request.app.state.config.WEB_SEARCH_DOMAIN_FILTER_LIST,
            'BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL': request.app.state.config.BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL,
            'BYPASS_WEB_SEARCH_WEB_LOADER': request.app.state.config.BYPASS_WEB_SEARCH_WEB_LOADER,
            'KAGI_SEARCH_API_KEY': request.app.state.config.KAGI_SEARCH_API_KEY,
            'WEB_LOADER_ENGINE': request.app.state.config.WEB_LOADER_ENGINE,
            'WEB_LOADER_TIMEOUT': request.app.state.config.WEB_LOADER_TIMEOUT,
            'ENABLE_WEB_LOADER_SSL_VERIFICATION': request.app.state.config.ENABLE_WEB_LOADER_SSL_VERIFICATION,
            'PLAYWRIGHT_WS_URL': request.app.state.config.PLAYWRIGHT_WS_URL,
            'PLAYWRIGHT_TIMEOUT': request.app.state.config.PLAYWRIGHT_TIMEOUT,
            'FIRECRAWL_API_KEY': request.app.state.config.FIRECRAWL_API_KEY,
            'FIRECRAWL_API_BASE_URL': request.app.state.config.FIRECRAWL_API_BASE_URL,
            'FIRECRAWL_TIMEOUT': request.app.state.config.FIRECRAWL_TIMEOUT,
            'TAVILY_API_KEY': request.app.state.config.TAVILY_API_KEY,
            'TAVILY_EXTRACT_DEPTH': request.app.state.config.TAVILY_EXTRACT_DEPTH,
            'EXTERNAL_WEB_LOADER_URL': request.app.state.config.EXTERNAL_WEB_LOADER_URL,
            'EXTERNAL_WEB_LOADER_API_KEY': request.app.state.config.EXTERNAL_WEB_LOADER_API_KEY,
            'YOUTUBE_LOADER_LANGUAGE': request.app.state.config.YOUTUBE_LOADER_LANGUAGE,
            'YOUTUBE_LOADER_PROXY_URL': request.app.state.config.YOUTUBE_LOADER_PROXY_URL,
            'YOUTUBE_LOADER_TRANSLATION': request.app.state.YOUTUBE_LOADER_TRANSLATION,
        },
    }


####################################
#
# Document process and retrieval
#
####################################


def can_merge_chunks(a: Document, b: Document) -> bool:
    if a.metadata.get('source') != b.metadata.get('source'):
        return False

    a_file_id = a.metadata.get('file_id')
    b_file_id = b.metadata.get('file_id')

    if a_file_id is not None and b_file_id is not None:
        return a_file_id == b_file_id

    return True


def merge_docs_to_target_size(
    request: Request,
    chunks: list[Document],
) -> list[Document]:
    """
    Best-effort normalization of chunk sizes.

    Attempts to grow small chunks up to a desired minimum size,
    without exceeding the maximum size or crossing source/file
    boundaries.

    Uses forward merging first (absorb the next chunk), then
    backward merging (append into the previous emitted chunk)
    for undersized chunks that can't grow forward.
    """
    min_size = request.app.state.config.CHUNK_MIN_SIZE_TARGET
    max_size = request.app.state.config.CHUNK_SIZE

    if min_size <= 0:
        return chunks

    measure: Callable[[str], int] = len
    if request.app.state.config.TEXT_SPLITTER == 'token':
        encoding = tiktoken.get_encoding(str(request.app.state.config.TIKTOKEN_ENCODING_NAME))
        measure = lambda text: len(encoding.encode(text))

    def _merge_backward(result: list[Document], content: str, chunk: Document) -> bool:
        """Try to append content into the last emitted chunk. Returns True on success."""
        if not result:
            return False
        prev = result[-1]
        if not can_merge_chunks(prev, chunk):
            return False
        merged = f'{prev.page_content}\n\n{content}'
        if measure(merged) > max_size:
            return False
        result[-1] = Document(page_content=merged, metadata={**prev.metadata})
        return True

    def _emit(result: list[Document], content: str, chunk: Document) -> None:
        """Emit a chunk, trying backward merge first if it's undersized."""
        is_undersized = measure(content) < min_size
        if is_undersized and _merge_backward(result, content, chunk):
            return
        result.append(Document(page_content=content, metadata={**chunk.metadata}))

    result: list[Document] = []
    current_chunk: Document | None = None
    current_content: str = ''

    for next_chunk in chunks:
        if current_chunk is None:
            current_chunk = next_chunk
            current_content = next_chunk.page_content
            continue

        # Forward merge: absorb next chunk into current if undersized and fits
        merged_content = f'{current_content}\n\n{next_chunk.page_content}'
        can_merge_forward = (
            can_merge_chunks(current_chunk, next_chunk)
            and measure(current_content) < min_size
            and measure(merged_content) <= max_size
        )

        if can_merge_forward:
            current_content = merged_content
        else:
            _emit(result, current_content, current_chunk)
            current_chunk = next_chunk
            current_content = next_chunk.page_content

    if current_chunk is not None:
        _emit(result, current_content, current_chunk)

    return result


def save_docs_to_vector_db(
    request: Request,
    docs,
    collection_name,
    metadata: dict | None = None,
    overwrite: bool = False,
    split: bool = True,
    add: bool = False,
    user=None,
) -> bool:
    def _get_docs_info(docs: list[Document]) -> str:
        docs_info = set()

        # Trying to select relevant metadata identifying the document.
        for doc in docs:
            metadata = getattr(doc, 'metadata', {})
            doc_name = metadata.get('name', '')
            if not doc_name:
                doc_name = metadata.get('title', '')
            if not doc_name:
                doc_name = metadata.get('source', '')
            if doc_name:
                docs_info.add(doc_name)

        return ', '.join(docs_info)

    log.debug(f'save_docs_to_vector_db: document {_get_docs_info(docs)} {collection_name}')

    # Check if entries with the same hash (metadata.hash) already exist
    if metadata and 'hash' in metadata:
        result = VECTOR_DB_CLIENT.query(
            collection_name=collection_name,
            filter={'hash': metadata['hash']},
        )

        if result is not None and result.ids and len(result.ids) > 0:
            existing_doc_ids = result.ids[0]
            if existing_doc_ids:
                # Check if the existing document belongs to the same file
                # If same file_id, this is a re-add/reindex - allow it
                # If different file_id, this is a duplicate - block it
                existing_file_id = None
                if result.metadatas and result.metadatas[0]:
                    existing_file_id = result.metadatas[0][0].get('file_id')

                if existing_file_id != metadata.get('file_id'):
                    log.info(f'Document with hash {metadata["hash"]} already exists')
                    raise ValueError(ERROR_MESSAGES.DUPLICATE_CONTENT)

    if split:
        # AST-aware code splitting. If every doc in this batch shares a
        # filename whose extension maps to a tree-sitter language, walk
        # the parse tree and emit one chunk per function/class/method
        # instead of the default character / markdown-header pipeline.
        # The default chunkers cut in the middle of function bodies,
        # which destroys retrieval quality for code (the embedder sees
        # half-functions with no signature). On a parse failure or
        # unknown extension, ``split_code`` returns an empty list and
        # we fall through to the existing pipeline below.
        if (
            getattr(request.app.state.config, 'KB_CODE_AST_SPLIT_ENABLED', True)
            and docs
        ):
            from open_webui.retrieval.loaders.code_splitter import (
                ext_to_language,
                split_code,
            )

            # Resolve filename / source for language detection. We try
            # the canonical metadata keys in the order they're most
            # likely to carry the actual filename (name = original
            # upload, title = derived, source = often a URL).
            def _doc_filename(d: Document) -> str:
                md = getattr(d, 'metadata', {}) or {}
                return str(md.get('name') or md.get('title') or md.get('source') or '')

            languages = {ext_to_language(_doc_filename(d)) for d in docs}
            # Only take the AST path when ALL docs in the batch agree
            # on a single non-None language. A mixed batch (e.g. a doc
            # carrying its own README ingested alongside code) would
            # otherwise have the README mis-routed through a Python
            # parser.
            if len(languages) == 1 and (lang := next(iter(languages))):
                code_chunks: list[Document] = []
                for doc in docs:
                    chunks = split_code(
                        doc.page_content,
                        lang,
                        chunk_size=request.app.state.config.CHUNK_SIZE,
                        chunk_overlap=request.app.state.config.CHUNK_OVERLAP,
                        base_metadata=dict(doc.metadata or {}),
                    )
                    code_chunks.extend(chunks)
                if code_chunks:
                    log.info(
                        'code_splitter: %d AST chunks for language=%s (%d docs in)',
                        len(code_chunks), lang, len(docs),
                    )
                    # AST splitter handled this batch end-to-end.
                    # Skip the markdown-header + recursive-character
                    # pipeline below by jumping past the splitting
                    # block via the early-bind to ``docs`` and the
                    # ``split = False``-ish state machine.
                    docs = code_chunks
                    split = False  # short-circuit the rest of the split branch

    if split:
        if request.app.state.config.ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER:
            log.info('Using markdown header text splitter')
            # Define headers to split on - covering most common markdown header levels
            markdown_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=[
                    ('#', 'Header 1'),
                    ('##', 'Header 2'),
                    ('###', 'Header 3'),
                    ('####', 'Header 4'),
                    ('#####', 'Header 5'),
                    ('######', 'Header 6'),
                ],
                strip_headers=False,  # Keep headers in content for context
            )

            split_docs = []
            for doc in docs:
                split_docs.extend(
                    [
                        Document(
                            page_content=split_chunk.page_content,
                            metadata={**doc.metadata},
                        )
                        for split_chunk in markdown_splitter.split_text(doc.page_content)
                    ]
                )

            docs = split_docs
            if request.app.state.config.CHUNK_MIN_SIZE_TARGET > 0:
                docs = merge_docs_to_target_size(request, docs)

        if request.app.state.config.TEXT_SPLITTER in ['', 'character']:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=request.app.state.config.CHUNK_SIZE,
                chunk_overlap=request.app.state.config.CHUNK_OVERLAP,
                add_start_index=True,
            )
            docs = text_splitter.split_documents(docs)
        elif request.app.state.config.TEXT_SPLITTER == 'token':
            log.info(f'Using token text splitter: {request.app.state.config.TIKTOKEN_ENCODING_NAME}')

            tiktoken.get_encoding(str(request.app.state.config.TIKTOKEN_ENCODING_NAME))
            text_splitter = TokenTextSplitter(
                encoding_name=str(request.app.state.config.TIKTOKEN_ENCODING_NAME),
                chunk_size=request.app.state.config.CHUNK_SIZE,
                chunk_overlap=request.app.state.config.CHUNK_OVERLAP,
                add_start_index=True,
            )
            docs = text_splitter.split_documents(docs)
        else:
            raise ValueError(ERROR_MESSAGES.DEFAULT('Invalid text splitter'))

    if len(docs) == 0:
        raise ValueError(ERROR_MESSAGES.EMPTY_CONTENT)

    texts = [sanitize_text_for_db(doc.page_content) for doc in docs]
    metadatas = [
        {
            **doc.metadata,
            **(metadata if metadata else {}),
            'embedding_config': {
                'engine': request.app.state.config.RAG_EMBEDDING_ENGINE,
                'model': request.app.state.config.RAG_EMBEDDING_MODEL,
            },
        }
        for doc in docs
    ]

    try:
        if VECTOR_DB_CLIENT.has_collection(collection_name=collection_name):
            log.info(f'collection {collection_name} already exists')

            if overwrite:
                VECTOR_DB_CLIENT.delete_collection(collection_name=collection_name)
                log.info(f'deleting existing collection {collection_name}')
            elif add is False:
                log.info(f'collection {collection_name} already exists, overwrite is False and add is False')
                return True

        log.info(f'generating embeddings for {collection_name}')
        embedding_function = get_embedding_function(
            request.app.state.config.RAG_EMBEDDING_ENGINE,
            request.app.state.config.RAG_EMBEDDING_MODEL,
            request.app.state.ef,
            (
                request.app.state.config.RAG_OPENAI_API_BASE_URL
                if request.app.state.config.RAG_EMBEDDING_ENGINE == 'openai'
                else (
                    request.app.state.config.RAG_OLLAMA_BASE_URL
                    if request.app.state.config.RAG_EMBEDDING_ENGINE == 'ollama'
                    else request.app.state.config.RAG_AZURE_OPENAI_BASE_URL
                )
            ),
            (
                request.app.state.config.RAG_OPENAI_API_KEY
                if request.app.state.config.RAG_EMBEDDING_ENGINE == 'openai'
                else (
                    request.app.state.config.RAG_OLLAMA_API_KEY
                    if request.app.state.config.RAG_EMBEDDING_ENGINE == 'ollama'
                    else request.app.state.config.RAG_AZURE_OPENAI_API_KEY
                )
            ),
            request.app.state.config.RAG_EMBEDDING_BATCH_SIZE,
            azure_api_version=(
                request.app.state.config.RAG_AZURE_OPENAI_API_VERSION
                if request.app.state.config.RAG_EMBEDDING_ENGINE == 'azure_openai'
                else None
            ),
            enable_async=request.app.state.config.ENABLE_ASYNC_EMBEDDING,
            concurrent_requests=request.app.state.config.RAG_EMBEDDING_CONCURRENT_REQUESTS,
        )

        # Run async embedding in sync context using the main event loop
        # This allows the main loop to stay responsive to health checks during long operations
        embedding_timeout = RAG_EMBEDDING_TIMEOUT

        future = asyncio.run_coroutine_threadsafe(
            embedding_function(
                list(map(lambda x: x.replace('\n', ' '), texts)),
                prefix=RAG_EMBEDDING_CONTENT_PREFIX,
                user=user,
            ),
            request.app.state.main_loop,
        )
        embeddings = future.result(timeout=embedding_timeout)
        log.info(f'embeddings generated {len(embeddings)} for {len(texts)} items')

        items = [
            {
                'id': str(uuid.uuid4()),
                'text': text,
                'vector': embeddings[idx],
                'metadata': metadatas[idx],
            }
            for idx, text in enumerate(texts)
        ]

        log.info(f'adding to collection {collection_name}')
        VECTOR_DB_CLIENT.insert(
            collection_name=collection_name,
            items=items,
        )

        log.info(f'added {len(items)} items to collection {collection_name}')
        try:
            from open_webui.retrieval.concepts.integration.ingest_hook import (
                on_docs_saved,
            )

            on_docs_saved(
                request.app.state,
                collection_name=collection_name,
                docs=docs,
                metadata=metadata,
            )
        except Exception:
            log.exception('concept_graph ingest hook failed; vector write unaffected')
        return True
    except Exception as e:
        log.exception(e)
        raise e


class ProcessFileForm(BaseModel):
    file_id: str
    content: str | None = None
    collection_name: str | None = None


@router.post('/process/file')
async def process_file(
    request: Request,
    form_data: ProcessFileForm,
    user=Depends(get_verified_user),
    db: AsyncSession = Depends(get_async_session),
):
    """
    Process a file and save its content to the vector database.
    Process a file and save its content to the vector database.
    Note: granular session management is used to prevent connection pool exhaustion.
    The session is committed before external API calls, and updates use a fresh session.
    """
    if user.role == 'admin':
        file = await Files.get_file_by_id(form_data.file_id, db=db)
    else:
        file = await Files.get_file_by_id_and_user_id(form_data.file_id, user.id, db=db)

    if file:
        try:
            collection_name = form_data.collection_name

            if collection_name is None:
                collection_name = f'file-{file.id}'
            else:
                await _validate_collection_access([collection_name], user, access_type='write')

            if form_data.content:
                # Update the content in the file
                # Usage: /files/{file_id}/data/content/update, /files/ (audio file upload pipeline)

                try:
                    # /files/{file_id}/data/content/update
                    await ASYNC_VECTOR_DB_CLIENT.delete_collection(collection_name=f'file-{file.id}')
                except Exception:
                    # Audio file upload pipeline
                    pass

                docs = [
                    Document(
                        page_content=form_data.content.replace('<br/>', '\n'),
                        metadata={
                            **file.meta,
                            'name': file.filename,
                            'created_by': file.user_id,
                            'file_id': file.id,
                            'source': file.filename,
                        },
                    )
                ]

                text_content = form_data.content
            elif form_data.collection_name:
                # Check if the file has already been processed and save the content
                # Usage: /knowledge/{id}/file/add, /knowledge/{id}/file/update

                result = await ASYNC_VECTOR_DB_CLIENT.query(
                    collection_name=f'file-{file.id}', filter={'file_id': file.id}
                )

                if result is not None and len(result.ids[0]) > 0:
                    docs = [
                        Document(
                            page_content=result.documents[0][idx],
                            metadata=result.metadatas[0][idx],
                        )
                        for idx, id in enumerate(result.ids[0])
                    ]
                else:
                    docs = [
                        Document(
                            page_content=file.data.get('content', ''),
                            metadata={
                                **file.meta,
                                'name': file.filename,
                                'created_by': file.user_id,
                                'file_id': file.id,
                                'source': file.filename,
                            },
                        )
                    ]

                text_content = file.data.get('content', '')
            else:
                # Process the file and save the content
                # Usage: /files/
                file_path = file.path
                if file_path:
                    file_path = await asyncio.to_thread(Storage.get_file, file_path)
                    loader = build_loader_from_config(request)
                    loader.user = user
                    docs = await loader.aload(file.filename, file.meta.get('content_type'), file_path)

                    docs = [
                        Document(
                            page_content=doc.page_content,
                            metadata={
                                **filter_metadata(doc.metadata),
                                'name': file.filename,
                                'created_by': file.user_id,
                                'file_id': file.id,
                                'source': file.filename,
                            },
                        )
                        for doc in docs
                    ]
                else:
                    docs = [
                        Document(
                            page_content=file.data.get('content', ''),
                            metadata={
                                **file.meta,
                                'name': file.filename,
                                'created_by': file.user_id,
                                'file_id': file.id,
                                'source': file.filename,
                            },
                        )
                    ]
                text_content = ' '.join([doc.page_content for doc in docs])

            log.debug(f'text_content: {text_content}')
            await Files.update_file_data_by_id(
                file.id,
                {'content': text_content},
                db=db,
            )
            hash = calculate_sha256_string(text_content)

            if request.app.state.config.BYPASS_EMBEDDING_AND_RETRIEVAL:
                await Files.update_file_data_by_id(file.id, {'status': 'completed'}, db=db)
                await Files.update_file_hash_by_id(file.id, hash, db=db)
                return {
                    'status': True,
                    'collection_name': None,
                    'filename': file.filename,
                    'content': text_content,
                }
            else:
                try:
                    # Commit any pending changes before the slow embedding step.
                    # Note: file is already a Pydantic model (not ORM), so no expunge needed.
                    await db.commit()

                    # External embedding API takes time (5-60s+).
                    # Subsequent updates use fresh async sessions.
                    # NOTE: save_docs_to_vector_db is a sync function that
                    # calls asyncio.run_coroutine_threadsafe(..., main_loop).result()
                    # which blocks the calling thread.  We MUST run it in a
                    # worker thread to avoid deadlocking the event loop.
                    result = await run_in_threadpool(
                        save_docs_to_vector_db,
                        request,
                        docs=docs,
                        collection_name=collection_name,
                        metadata={
                            'file_id': file.id,
                            'name': file.filename,
                            'hash': hash,
                        },
                        add=(True if form_data.collection_name else False),
                        user=user,
                    )
                    log.info(f'added {len(docs)} items to collection {collection_name}')

                    if result:
                        # Fresh session for the final update.
                        async with get_async_db() as session:
                            await Files.update_file_metadata_by_id(
                                file.id,
                                {
                                    'collection_name': collection_name,
                                },
                                db=session,
                            )

                            await Files.update_file_data_by_id(
                                file.id,
                                {'status': 'completed'},
                                db=session,
                            )
                            await Files.update_file_hash_by_id(file.id, hash, db=session)

                            return {
                                'status': True,
                                'collection_name': collection_name,
                                'filename': file.filename,
                                'content': text_content,
                            }
                    else:
                        raise Exception('Error saving document to vector database')
                except Exception as e:
                    raise e

        except Exception as e:
            log.exception(e)
            # Fresh session for error status update.
            async with get_async_db() as session:
                await Files.update_file_data_by_id(
                    file.id,
                    {'status': 'failed'},
                    db=session,
                )
                # Clear the hash so the file can be re-uploaded after fixing the issue
                await Files.update_file_hash_by_id(file.id, None, db=session)

            if 'No pandoc was found' in str(e):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=ERROR_MESSAGES.PANDOC_NOT_INSTALLED,
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(e),
                )

    else:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=ERROR_MESSAGES.NOT_FOUND)


class ProcessTextForm(BaseModel):
    name: str
    content: str
    collection_name: str | None = None


@router.post('/process/text')
async def process_text(
    request: Request,
    form_data: ProcessTextForm,
    user=Depends(get_verified_user),
):
    collection_name = form_data.collection_name
    if collection_name is None:
        collection_name = calculate_sha256_string(form_data.content)
    else:
        await _validate_collection_access([collection_name], user, access_type='write')

    docs = [
        Document(
            page_content=form_data.content,
            metadata={'name': form_data.name, 'created_by': user.id},
        )
    ]
    text_content = form_data.content
    log.debug(f'text_content: {text_content}')

    result = await run_in_threadpool(save_docs_to_vector_db, request, docs, collection_name, user=user)
    if result:
        return {
            'status': True,
            'collection_name': collection_name,
            'content': text_content,
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(),
        )


@router.post('/process/youtube')
@router.post('/process/web')
async def process_web(
    request: Request,
    form_data: ProcessUrlForm,
    process: bool = Query(True, description='Whether to process and save the content'),
    overwrite: bool = Query(True, description='Whether to overwrite existing collection'),
    user=Depends(get_verified_user),
):
    try:
        content, docs = await run_in_threadpool(get_content_from_url, request, form_data.url)
        log.debug(f'text_content: {content}')

        if process:
            collection_name = form_data.collection_name
            if not collection_name:
                collection_name = calculate_sha256_string(form_data.url)[:63]
            else:
                await _validate_collection_access([collection_name], user, access_type='write')

            if not request.app.state.config.BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL:
                await run_in_threadpool(
                    save_docs_to_vector_db,
                    request,
                    docs,
                    collection_name,
                    overwrite=overwrite,
                    add=(not overwrite),
                    user=user,
                )
            else:
                collection_name = None

            return {
                'status': True,
                'collection_name': collection_name,
                'filename': form_data.url,
                'file': {
                    'data': {
                        'content': content,
                    },
                    'meta': {
                        'name': form_data.url,
                        'source': form_data.url,
                    },
                },
            }
        else:
            return {
                'status': True,
                'content': content,
            }
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )


class _EnginePlan(NamedTuple):
    """One leg of a (possibly fanned-out) dispatch plan.

    Each plan entry is dispatched concurrently via :func:`asyncio.gather`;
    a single-entry plan reduces to the pre-fanout behavior. ``query`` is
    the engine-specific query string (bangs stripped where the engine's
    router fired; original otherwise).
    """

    engine: str
    query: str
    lens_id: Optional[str] = None
    arxiv_category: Optional[str] = None
    mslearn_product: Optional[str] = None
    godot_version: Optional[str] = None
    hf_author: Optional[str] = None
    hf_sort: Optional[str] = None
    hf_pipeline_tag: Optional[str] = None
    # Bitbucket: workspace is pulled from app.state.config at dispatch time
    # (so a config change without a restart takes effect immediately).
    # repo_slug is the second half of a `workspace/reposlug` reference; when
    # None, the PR-search leg of the Bitbucket fanout is skipped because
    # Bitbucket Cloud has no workspace-wide PR API.
    bb_repo_slug: Optional[str] = None


async def _build_kagi_plan(request: Request, query: str) -> _EnginePlan:
    """Resolve Kagi lens routing and return a Kagi ``_EnginePlan``.

    Pulled out as a helper because both the "Kagi-only" path and every
    fanout path need an EnginePlan for Kagi with the lens YAML applied.
    """
    lens_id: Optional[str] = None
    cleaned = query
    if getattr(request.app.state.config, 'ENABLE_KAGI_LENS_ROUTING', True):
        try:
            lens_id, cleaned, lens_name = await asyncio.to_thread(
                route_kagi_lens,
                query,
                config_path=getattr(
                    request.app.state.config, 'KAGI_LENSES_CONFIG_PATH', None
                ),
            )
        except Exception as e:  # never let lens routing break search
            log.debug('search_web: kagi lens routing failed: %s', e)
            lens_id = None
            cleaned = query
            lens_name = None
        if lens_id:
            log.debug(
                "search_web: kagi lens routed query=%r → lens=%s (%s); cleaned=%r",
                query,
                lens_id,
                lens_name,
                cleaned,
            )
    return _EnginePlan(engine='kagi', query=cleaned, lens_id=lens_id)


async def _build_dispatch_plan(
    request: Request, default_engine: str, query: str
) -> list[_EnginePlan]:
    """Decide which engines to fan out to for ``query``.

    Returns a non-empty list of :class:`_EnginePlan` entries. A single entry
    is the no-fanout case (configured default, single bang match, or fanout
    disabled). Multiple entries trigger parallel dispatch + merge.

    Routing precedence (any layer fails open):

    1. **arXiv** — if a bang fires, return arXiv-only. If a keyword fires
       and ``WEB_SEARCH_FANOUT_KAGI`` is enabled, return [arXiv, Kagi].
    2. **Doc portals (MDN / Microsoft Learn)** — same bang-vs-keyword
       split. Only consulted when arXiv didn't already win.
    3. **Hugging Face** — bang / portal keyword / open-weights family
       auto-route. Family matches forward a canonical HF org as
       ``hf_author`` so we surface official releases (``google/gemma-3``,
       ``meta-llama/Llama-3.3``) and false-positive matches collapse to
       ~zero results.
    4. **Kagi** — fallback when no specialty router fired. Lens routing is
       applied here.
    """
    config = request.app.state.config
    fanout_enabled = getattr(config, 'WEB_SEARCH_FANOUT_KAGI', True)

    if default_engine == 'kagi' and getattr(config, 'ENABLE_ARXIV_SEARCH', True):
        try:
            arxiv = await asyncio.to_thread(route_arxiv, query)
        except Exception as e:  # never let arxiv routing break search
            log.debug('search_web: arxiv routing failed: %s', e)
            arxiv = None  # type: ignore[assignment]

        if arxiv is not None and arxiv.matched:
            log.debug(
                "search_web: arxiv intent routed query=%r cat=%s cleaned=%r exclusive=%s",
                query,
                arxiv.category,
                arxiv.query,
                arxiv.exclusive,
            )
            plan: list[_EnginePlan] = [
                _EnginePlan(
                    engine='arxiv',
                    query=arxiv.query,
                    arxiv_category=arxiv.category,
                )
            ]
            if not arxiv.exclusive and fanout_enabled:
                plan.append(await _build_kagi_plan(request, query))
            return plan

    if default_engine == 'kagi' and getattr(config, 'ENABLE_DOCS_ROUTING', True):
        try:
            docs = await asyncio.to_thread(route_docs, query)
        except Exception as e:  # never let docs routing break search
            log.debug('search_web: docs routing failed: %s', e)
            docs = None  # type: ignore[assignment]

        if docs is not None and docs.engine is not None:
            portal_enabled = (
                docs.engine == 'mdn'
                and getattr(config, 'ENABLE_MDN_SEARCH', True)
            ) or (
                docs.engine == 'mslearn'
                and getattr(config, 'ENABLE_MSLEARN_SEARCH', True)
            ) or (
                docs.engine == 'godot'
                and getattr(config, 'ENABLE_GODOT_SEARCH', True)
            )
            if portal_enabled:
                log.debug(
                    "search_web: docs routing → %s (product=%s version=%s); "
                    "query=%r cleaned=%r exclusive=%s",
                    docs.engine,
                    docs.product,
                    docs.version,
                    query,
                    docs.query,
                    docs.exclusive,
                )
                plan = [
                    _EnginePlan(
                        engine=docs.engine,
                        query=docs.query,
                        mslearn_product=docs.product,
                        godot_version=docs.version,
                    )
                ]
                if not docs.exclusive and fanout_enabled:
                    plan.append(await _build_kagi_plan(request, query))
                return plan
            log.debug(
                'search_web: docs routing matched %s but engine is disabled; falling back to kagi',
                docs.engine,
            )

    if default_engine == 'kagi' and getattr(config, 'ENABLE_HF_SEARCH', True):
        try:
            hf = await asyncio.to_thread(route_hf, query)
        except Exception as e:  # never let hf routing break search
            log.debug('search_web: hf routing failed: %s', e)
            hf = None  # type: ignore[assignment]

        if hf is not None and hf.matched:
            log.debug(
                "search_web: hf intent routed query=%r author=%s cleaned=%r exclusive=%s",
                query,
                hf.author,
                hf.query,
                hf.exclusive,
            )
            plan = [
                _EnginePlan(
                    engine='huggingface',
                    query=hf.query,
                    hf_author=hf.author,
                    hf_sort=hf.sort,
                    hf_pipeline_tag=hf.pipeline_tag,
                )
            ]
            if not hf.exclusive and fanout_enabled:
                plan.append(await _build_kagi_plan(request, query))
            return plan

    # Bitbucket: internal-codebase intent. Gated on both the feature flag
    # and the presence of a workspace+token (no point routing to a backend
    # we can't authenticate to). Sits after the public-portal routers but
    # before the Kagi fallback because "our codebase" / `workspace/repo`
    # references are unambiguous and should win over a generic web search.
    if (
        default_engine == 'kagi'
        and getattr(config, 'ENABLE_BITBUCKET_SEARCH', False)
        and getattr(config, 'BITBUCKET_ACCESS_TOKEN', '')
        and getattr(config, 'BITBUCKET_WORKSPACE', '')
    ):
        try:
            bb = await asyncio.to_thread(
                route_bitbucket, query, getattr(config, 'BITBUCKET_WORKSPACE', '')
            )
        except Exception as e:
            log.debug('search_web: bitbucket routing failed: %s', e)
            bb = None  # type: ignore[assignment]

        if bb is not None and bb.matched:
            log.debug(
                'search_web: bitbucket intent routed query=%r cleaned=%r '
                'repo_slug=%s exclusive=%s',
                query,
                bb.query,
                bb.repo_slug,
                bb.exclusive,
            )
            plan = [
                _EnginePlan(
                    engine='bitbucket',
                    query=bb.query,
                    bb_repo_slug=bb.repo_slug,
                )
            ]
            if not bb.exclusive and fanout_enabled:
                plan.append(await _build_kagi_plan(request, query))
            return plan

    # Kagi default (or non-Kagi configured engine with no specialty routing).
    if default_engine == 'kagi':
        return [await _build_kagi_plan(request, query)]
    return [_EnginePlan(engine=default_engine, query=query)]


def _dedup_key(link: str) -> str:
    """Normalize a result URL for cross-engine dedup.

    arXiv exposes the same paper at both ``arxiv.org/abs/X.Y`` and
    ``arxiv.org/pdf/X.Y``, and Kagi may return either form. Strip the
    ``/abs/`` vs ``/pdf/`` distinction and the trailing version suffix so
    they collapse to the same key.
    """
    if not link:
        return ''
    parsed = urlparse(link)
    netloc = parsed.netloc.lower()
    path = parsed.path
    if netloc.endswith('arxiv.org'):
        path = path.replace('/pdf/', '/abs/')
        # Strip trailing version suffix (``v1`` / ``v2``) + optional .pdf
        path = re.sub(r'v\d+(\.pdf)?$', '', path)
    path = path.rstrip('/')
    return f'{netloc}{path}'


def _merge_results(
    bundles: list[list[SearchResult]], *, max_count: int
) -> list[SearchResult]:
    """Round-robin interleave + dedup-by-URL + cap at ``max_count``.

    Round-robin preserves the top result from every provider in the merged
    list, so portal-specific authoritative results don't get crowded out by
    Kagi's broader index (or vice-versa). The dedup keeps cross-engine
    overlap from wasting the result budget.
    """
    if max_count <= 0:
        return []
    seen: set[str] = set()
    merged: list[SearchResult] = []
    indices = [0] * len(bundles)
    while len(merged) < max_count and any(
        indices[i] < len(bundles[i]) for i in range(len(bundles))
    ):
        for i, bundle in enumerate(bundles):
            if indices[i] >= len(bundle):
                continue
            result = bundle[indices[i]]
            indices[i] += 1
            key = _dedup_key(result.link)
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            merged.append(result)
            if len(merged) >= max_count:
                break
    return merged


async def _run_plan(
    request: Request,
    plan: list[_EnginePlan],
    *,
    search_filter: Optional[WebSearchFilter],
    user=None,
) -> list[SearchResult]:
    """Execute every leg of ``plan`` concurrently and merge the results.

    Per-engine post-filtering (``apply_to_results``) runs on each leg
    independently — Kagi already does the filter natively
    (``has_full_native_support``) so it's skipped there, while the portal
    adapters get the generic post-filter. Single-engine plans short-circuit
    the merge so the no-fanout path stays a single ``apply_to_results`` call.
    """

    async def run_one(entry: _EnginePlan) -> list[SearchResult]:
        try:
            results = await _dispatch_web_search(
                request,
                entry.engine,
                entry.query,
                user=user,
                search_filter=search_filter,
                lens_id=entry.lens_id,
                arxiv_category=entry.arxiv_category,
                mslearn_product=entry.mslearn_product,
                godot_version=entry.godot_version,
                hf_author=entry.hf_author,
                hf_sort=entry.hf_sort,
                hf_pipeline_tag=entry.hf_pipeline_tag,
                bb_repo_slug=entry.bb_repo_slug,
            )
        except Exception as e:
            # Fanout partial-failure tolerance: an upstream outage on one
            # engine shouldn't black-hole the whole search. Log loudly so
            # the failure is visible, then return [] and let the merge
            # surface results from the other legs.
            log.warning(
                'search_web: %s engine failed in fanout (%d-leg plan): %s',
                entry.engine,
                len(plan),
                e,
            )
            return []

        if (
            search_filter is not None
            and not search_filter.is_empty()
            and not has_full_native_support(entry.engine)
        ):
            results = search_filter.apply_to_results(results)
        return results

    bundles = await asyncio.gather(*(run_one(entry) for entry in plan))

    if len(bundles) == 1:
        return bundles[0]

    max_count = request.app.state.config.WEB_SEARCH_RESULT_COUNT
    merged = _merge_results(bundles, max_count=max_count)
    log.debug(
        'search_web: fanout merged %s -> %d result(s) (engines=%s)',
        [len(b) for b in bundles],
        len(merged),
        [entry.engine for entry in plan],
    )
    return merged


async def search_web(
    request: Request, engine: str, query: str, user=None
) -> list[SearchResult]:
    """Run a web search, applying natural-language filters where possible.

    Pipeline:

    1. **Plan build** (:func:`_build_dispatch_plan`) — consults the arXiv,
       doc-portal, and Kagi-lens routers in priority order. Bang matches
       are exclusive (portal-only). Keyword matches fan out to the portal
       *and* Kagi in parallel for broader coverage, gated by
       ``WEB_SEARCH_FANOUT_KAGI``.
    2. **NL filter** — :func:`extract_filter_from_query` parses
       date/region/domain/keyword intent from the primary (cleaned) query
       and is shared across every leg of the plan.
    3. **Dispatch + merge** (:func:`_run_plan`) — every leg of the plan is
       dispatched concurrently via ``asyncio.gather``. Per-engine
       post-filtering runs on each leg; results are round-robin interleaved
       and deduplicated.

    Each layer fails open: a missing lens config, a downed NL-filter model,
    a routing parse error, or even a complete engine outage all degrade
    gracefully — a fanout sibling can carry the search alone.
    """
    plan = await _build_dispatch_plan(request, engine, query)

    # The "primary" leg is the first plan entry — for fanout queries this
    # is the specialty engine (arXiv / MDN / MS Learn); for single-engine
    # plans it's the only engine. Use its (potentially bang-stripped) query
    # as the NL filter input so the filter operates on the user's actual
    # intent rather than the routing marker.
    nl_query = plan[0].query
    engines_label = '+'.join(entry.engine for entry in plan)

    # Skip the NL filter call when the plan has no Kagi leg. The portal
    # adapters (arXiv / MDN / MS Learn) don't benefit much from the filter
    # (their result domains are pinned; their date handling is engine-
    # specific) and a hallucinated ``after`` from the task model can silently
    # filter every result out — observed: granite4.1:8b emitting
    # ``after=today`` for "!arxiv constraining LLM context windows", which
    # then made the arxiv post-filter drop all 24 returned papers. When a
    # Kagi leg IS in the plan (default search or keyword fanout), the
    # filter still runs and benefits Kagi's native filter params.
    needs_nl_filter = any(entry.engine == 'kagi' for entry in plan)
    if needs_nl_filter:
        search_filter = await asyncio.to_thread(extract_filter_from_query, nl_query)
    else:
        search_filter = None
        log.debug(
            "search_web: skipping nl filter for portal-only plan engines=%s",
            engines_label,
        )

    # Recency-intent → shorter page-cache TTL. The NL filter sets ``after``
    # to ~30d before today for "recent/latest/news" queries and ~1-2d for
    # "breaking news". We treat any ``after`` within ~31 days of today as
    # recency intent and surface that to ``process_web_search`` via the
    # request scope so the loader downstream can opt into a shorter cache
    # TTL. Done at request scope (not contextvar) so multiple concurrent
    # sub-query tasks under ``asyncio.gather`` can all contribute.
    try:
        if (
            search_filter is not None
            and search_filter.after is not None
            and (date.today() - search_filter.after).days <= 31
        ):
            request.state.web_page_cache_recency_hint = True
    except Exception:  # defensive: never fail a search over a cache hint
        pass

    # Surface what the NL parser produced so "LLM said empty" vs "LLM call
    # failed" is visible without cross-referencing nl_filter's own debug logs.
    if search_filter is not None and not search_filter.is_empty():
        log.debug(
            "search_web: parsed nl filter for query=%r engines=%s -> %s",
            nl_query,
            engines_label,
            search_filter.model_dump(exclude_none=True, exclude_defaults=True),
        )
    else:
        log.debug(
            "search_web: no nl filter applied for query=%r engines=%s "
            "(parser returned empty or failed open)",
            nl_query,
            engines_label,
        )

    return await _run_plan(request, plan, search_filter=search_filter, user=user)


async def _dispatch_web_search(
    request: Request,
    engine: str,
    query: str,
    user=None,
    search_filter: Optional[WebSearchFilter] = None,
    lens_id: Optional[str] = None,
    arxiv_category: Optional[str] = None,
    mslearn_product: Optional[str] = None,
    godot_version: Optional[str] = None,
    hf_author: Optional[str] = None,
    hf_sort: Optional[str] = None,
    hf_pipeline_tag: Optional[str] = None,
    bb_repo_slug: Optional[str] = None,
) -> list[SearchResult]:
    """Dispatch a web search query to the configured engine and return results.

    Providers that have been migrated to async (aiohttp) are awaited natively.
    Legacy sync providers are offloaded via ``asyncio.to_thread`` to avoid
    blocking the event loop.
    """

    if engine == 'kagi':
        if not request.app.state.config.KAGI_SEARCH_API_KEY:
            raise Exception('No KAGI_SEARCH_API_KEY found in environment variables')
        return await asyncio.to_thread(
            search_kagi,
            request.app.state.config.KAGI_SEARCH_API_KEY,
            query,
            request.app.state.config.WEB_SEARCH_RESULT_COUNT,
            request.app.state.config.WEB_SEARCH_DOMAIN_FILTER_LIST,
            search_filter,
            lens_id,
        )

    if engine == 'arxiv':
        return await asyncio.to_thread(
            search_arxiv,
            query,
            request.app.state.config.WEB_SEARCH_RESULT_COUNT,
            search_filter,
            arxiv_category,
        )

    if engine == 'mdn':
        return await asyncio.to_thread(
            search_mdn,
            query,
            request.app.state.config.WEB_SEARCH_RESULT_COUNT,
            search_filter,
        )

    if engine == 'mslearn':
        return await asyncio.to_thread(
            search_mslearn,
            query,
            request.app.state.config.WEB_SEARCH_RESULT_COUNT,
            search_filter,
            mslearn_product,
        )

    if engine == 'godot':
        return await asyncio.to_thread(
            search_godot,
            query,
            request.app.state.config.WEB_SEARCH_RESULT_COUNT,
            search_filter,
            godot_version,
        )

    if engine == 'huggingface':
        return await asyncio.to_thread(
            search_huggingface,
            query,
            request.app.state.config.WEB_SEARCH_RESULT_COUNT,
            search_filter,
            hf_author,
            hf_sort,
            hf_pipeline_tag,
        )

    if engine == 'bitbucket':
        # Pull workspace + token at dispatch time so an admin config edit
        # (via the WebUI settings UI) takes effect on the next search
        # without needing a restart. The router already verified both are
        # set before producing a 'bitbucket' plan entry, but re-check
        # defensively in case config changed between plan-build and
        # dispatch.
        workspace = request.app.state.config.BITBUCKET_WORKSPACE
        token = request.app.state.config.BITBUCKET_ACCESS_TOKEN
        if not (workspace and token):
            log.warning(
                'search_web: bitbucket dispatch missing workspace/token at exec time; '
                'returning empty results'
            )
            return []
        return await asyncio.to_thread(
            search_bitbucket,
            query,
            workspace,
            token,
            request.app.state.config.WEB_SEARCH_RESULT_COUNT,
            repo_slug=bb_repo_slug,
        )

    # This fork only supports Kagi + the subject-specific portal adapters
    # (arXiv / MDN / MS Learn / Godot / Hugging Face / Bitbucket); legacy
    # engines are intentionally unsupported.
    raise Exception(
        f'Unsupported web search engine: {engine!r} '
        '(supported: "kagi", "arxiv", "mdn", "mslearn", "godot", '
        '"huggingface", "bitbucket")'
    )


@router.post('/process/web/search')
async def process_web_search(request: Request, form_data: SearchForm, user=Depends(get_verified_user)):
    if not request.app.state.config.ENABLE_WEB_SEARCH:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )

    if user.role != 'admin' and not await has_permission(
        user.id, 'features.web_search', request.app.state.config.USER_PERMISSIONS
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )

    # Per-stage stopwatch for the web-search pipeline. One structured line
    # lands in the logs per call so we can answer "where did the seconds
    # go" without grepping across services. Stages roughly match the
    # control flow below: search = Kagi fan-out (incl. NL filter LLM
    # calls), load = Playwright fetch of every URL, embed = MLX embedding
    # + vector-db write. Numbers are wall-clock ms.
    t_pipeline_start = time.perf_counter()
    stage_ms: dict[str, float] = {}

    def _mark(stage: str, t0: float) -> None:
        stage_ms[stage] = round((time.perf_counter() - t0) * 1000, 1)

    def _emit_timing(outcome: str, **extra) -> None:
        total_ms = round((time.perf_counter() - t_pipeline_start) * 1000, 1)
        log.info(
            'web_search_timing outcome=%s engine=%s n_queries=%d total_ms=%.1f stages=%s extra=%s',
            outcome,
            request.app.state.config.WEB_SEARCH_ENGINE,
            len(form_data.queries or []),
            total_ms,
            stage_ms,
            extra,
        )

    urls = []
    result_items = []

    try:
        logging.debug(f'trying to web search with {request.app.state.config.WEB_SEARCH_ENGINE, form_data.queries}')
        t_search = time.perf_counter()

        # Use semaphore to limit concurrent requests based on WEB_SEARCH_CONCURRENT_REQUESTS
        # 0 or None = unlimited (previous behavior), positive number = limited concurrency
        # Set to 1 for sequential execution (rate-limited APIs like Brave free tier)
        concurrent_limit = request.app.state.config.WEB_SEARCH_CONCURRENT_REQUESTS

        if concurrent_limit:
            # Limited concurrency with semaphore
            semaphore = asyncio.Semaphore(concurrent_limit)

            async def search_query_with_semaphore(query):
                async with semaphore:
                    return await search_web(
                        request,
                        request.app.state.config.WEB_SEARCH_ENGINE,
                        query,
                        user,
                    )

            search_tasks = [search_query_with_semaphore(query) for query in form_data.queries]
        else:
            # Unlimited parallel execution
            search_tasks = [
                search_web(
                    request,
                    request.app.state.config.WEB_SEARCH_ENGINE,
                    query,
                    user,
                )
                for query in form_data.queries
            ]

        # ``return_exceptions=True`` so one rejected sub-query (Kagi 400 on a
        # bad lens, transient 5xx, rate limit, etc.) doesn't tank the entire
        # chat turn. Without this, the first failing query's exception
        # propagates out of ``gather`` and the user sees "An error occurred
        # while searching the web" even when 2/3 sibling queries had
        # perfectly good results. Observed on a "Kentucky" prompt where the
        # NL filter inferred ``region: US`` on every sub-query and Kagi
        # 400'd all three uniformly -- but the same pattern would have hit
        # the user even if only one of three queries had been malformed.
        search_results_raw = await asyncio.gather(*search_tasks, return_exceptions=True)
        _mark('search', t_search)

        search_results = []
        search_errors: list[str] = []
        for query, result in zip(form_data.queries, search_results_raw):
            if isinstance(result, BaseException):
                log.warning(
                    'search_web: sub-query failed query=%r err=%s: %s',
                    query, type(result).__name__, result,
                )
                search_errors.append(f'{query!r}: {result}')
                continue
            search_results.append(result)
            if result:
                for item in result:
                    if item and item.link:
                        result_items.append(item)
                        urls.append(item.link)

        urls = list(dict.fromkeys(urls))
        log.debug(f'urls: {urls}')

        # Only abort the turn if EVERY sub-query failed. If at least one
        # came back with results, proceed with what we have and report
        # the failures structurally so they show up in web_search_timing
        # without poisoning the chat reply.
        if search_errors and not search_results:
            raise Exception(
                'all '
                f'{len(form_data.queries)} sub-queries failed: '
                + ' | '.join(search_errors)
            )
        if search_errors:
            log.info(
                'search_web: %d/%d sub-queries failed but %d succeeded; proceeding',
                len(search_errors), len(form_data.queries), len(search_results),
            )

    except Exception as e:
        log.exception('Web search failed')
        _emit_timing('error_search', error=str(e))
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.WEB_SEARCH_ERROR(e))

    if len(urls) == 0:
        _emit_timing('no_results')
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.DEFAULT('No results found from web search'),
        )

    try:
        if request.app.state.config.BYPASS_WEB_SEARCH_WEB_LOADER:
            search_results = [item for result in search_results for item in result if result]

            docs = [
                Document(
                    page_content=result.snippet,
                    metadata={
                        'source': result.link,
                        'title': result.title,
                        'snippet': result.snippet,
                        'link': result.link,
                    },
                )
                for result in search_results
                if hasattr(result, 'snippet') and result.snippet is not None
            ]
        else:
            # If ANY sub-query in this batch surfaced a recency-intent hint
            # (NL filter parsed ``after=`` within ~31 days of today), use the
            # shorter recency TTL for the whole batch. Mixing TTLs per-URL
            # would be possible but a single batch-wide TTL is plenty for the
            # heuristic and keeps the cache plumbing trivial.
            cache_ttl_override = None
            if getattr(request.state, 'web_page_cache_recency_hint', False):
                cache_ttl_override = page_cache.recency_ttl_seconds()
                log.debug(
                    'process_web_search: recency intent detected, page cache TTL=%ss',
                    cache_ttl_override,
                )
            # Plumb the full set of sub-queries into the loader so the
            # heading-trim post-processor (see SafeTrafilaturaLoader._extract)
            # can match on the union of every query term in the batch.
            # Joining with spaces is fine -- the trimmer tokenizes via a
            # non-alnum regex so "[query A] [query B]" is indistinguishable
            # from "query A B" for matching purposes.
            heading_trim_query = ' '.join(form_data.queries or [])
            loader = get_web_loader(
                urls,
                verify_ssl=request.app.state.config.ENABLE_WEB_LOADER_SSL_VERIFICATION,
                requests_per_second=request.app.state.config.WEB_LOADER_CONCURRENT_REQUESTS,
                trust_env=request.app.state.config.WEB_SEARCH_TRUST_ENV,
                cache_ttl_seconds=cache_ttl_override,
                query=heading_trim_query,
                heading_trim_enabled=getattr(
                    request.app.state.config, 'WEB_HEADING_TRIM_ENABLED', True
                ),
                heading_trim_min_token_len=getattr(
                    request.app.state.config, 'WEB_HEADING_TRIM_MIN_TOKEN_LEN', 3
                ),
                heading_trim_keep_intro=getattr(
                    request.app.state.config, 'WEB_HEADING_TRIM_KEEP_INTRO', True
                ),
                js_fallback_enabled=getattr(
                    request.app.state.config, 'WEB_JS_FALLBACK_ENABLED', True
                ),
                js_fallback_min_extract_chars=getattr(
                    request.app.state.config, 'WEB_JS_FALLBACK_MIN_EXTRACT_CHARS', 200
                ),
            )
            t_load = time.perf_counter()
            docs = await loader.aload()
            _mark('load', t_load)

            # Per-doc gemma-3-1b compress pass. Only meaningful on the
            # bypass-embedding path -- when web docs go straight into
            # chat context, the compress pass slashes input tokens
            # without losing citation fidelity (the original source URL
            # is preserved on each compressed doc's metadata). On the
            # vector-store path the chunker + reranker already does
            # the equivalent narrowing, so skipping the LLM call there
            # avoids paying the latency twice.
            if (
                request.app.state.config.BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL
                and getattr(
                    request.app.state.config, 'WEB_SEARCH_COMPRESS_ENABLED', True
                )
                and docs
            ):
                from open_webui.retrieval.web.llm_compress import compress_docs

                t_compress = time.perf_counter()
                compress_query = ' '.join(form_data.queries or [])
                try:
                    docs = await compress_docs(
                        docs,
                        compress_query,
                        request=request,
                    )
                except Exception as e:
                    # compress_docs is documented as never raising, but
                    # we keep this last-resort guard so a bug in the
                    # compressor can never tank a search. The original
                    # docs from ``loader.aload()`` are already in
                    # ``docs`` if we reach this branch (the exception
                    # would only fire from import errors or similar).
                    log.warning(
                        'compress_docs raised unexpectedly; using uncompressed docs: %s', e
                    )
                _mark('compress', t_compress)

        urls = [
            doc.metadata.get('source') for doc in docs if doc.metadata.get('source')
        ]  # only keep the urls returned by the loader
        result_items = [
            dict(item) for item in result_items if item.link in urls
        ]  # only keep the search results that have been loaded

        if request.app.state.config.BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL:
            _emit_timing('ok_bypass_embed', n_docs=len(docs), n_urls=len(urls))
            return {
                'status': True,
                'collection_name': None,
                'filenames': urls,
                'items': result_items,
                'docs': [
                    {
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                    }
                    for doc in docs
                ],
                'loaded_count': len(docs),
            }
        else:
            # Create a single collection for all documents
            collection_name = f'web-search-{calculate_sha256_string("-".join(form_data.queries))}'[:63]

            t_embed = time.perf_counter()
            try:
                await run_in_threadpool(
                    save_docs_to_vector_db,
                    request,
                    docs,
                    collection_name,
                    overwrite=True,
                    user=user,
                )
            except Exception as e:
                log.debug(f'error saving docs: {e}')
            _mark('embed', t_embed)

            _emit_timing('ok', n_docs=len(docs), n_urls=len(urls))
            return {
                'status': True,
                'collection_names': [collection_name],
                'items': result_items,
                'filenames': urls,
                'loaded_count': len(docs),
            }
    except Exception as e:
        log.exception('Web search content loading failed')
        _emit_timing('error_load_or_embed', error=str(e))
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.DEFAULT(e))


async def _validate_collection_access(collection_names: list[str], user, access_type: str = 'read') -> None:
    """
    Raise 403 if the user lacks access to any of the requested collections.
    Delegates to the shared filter_accessible_collections utility so the
    access rules stay in one place.
    """
    requested = set(collection_names)
    allowed = await filter_accessible_collections(requested, user, access_type=access_type)
    denied = requested - allowed
    if denied:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )


class QueryDocForm(BaseModel):
    collection_name: str
    query: str
    k: int | None = None
    k_reranker: int | None = None
    r: float | None = None
    hybrid: bool | None = None


@router.post('/query/doc')
async def query_doc_handler(
    request: Request,
    form_data: QueryDocForm,
    user=Depends(get_verified_user),
):
    await _validate_collection_access([form_data.collection_name], user)

    try:
        if request.app.state.config.ENABLE_RAG_HYBRID_SEARCH and (form_data.hybrid is None or form_data.hybrid):
            collection_results = {}
            collection_results[form_data.collection_name] = await ASYNC_VECTOR_DB_CLIENT.get(
                collection_name=form_data.collection_name
            )
            return await query_doc_with_hybrid_search(
                collection_name=form_data.collection_name,
                collection_result=collection_results[form_data.collection_name],
                query=form_data.query,
                embedding_function=lambda query, prefix: request.app.state.EMBEDDING_FUNCTION(
                    query, prefix=prefix, user=user
                ),
                k=form_data.k if form_data.k else request.app.state.config.TOP_K,
                reranking_function=(
                    (lambda query, documents: request.app.state.RERANKING_FUNCTION(query, documents, user=user))
                    if request.app.state.RERANKING_FUNCTION
                    else None
                ),
                k_reranker=form_data.k_reranker or request.app.state.config.TOP_K_RERANKER,
                r=(form_data.r if form_data.r else request.app.state.config.RELEVANCE_THRESHOLD),
                hybrid_bm25_weight=(
                    form_data.hybrid_bm25_weight
                    if form_data.hybrid_bm25_weight
                    else request.app.state.config.HYBRID_BM25_WEIGHT
                ),
                concept_graph_store=getattr(request.app.state, 'concept_graph_store', None),
                **_build_concept_graph_extras(request.app.state),  # W6.10: sync embed_fn + name-only cosine reranker
                concept_graph_tiebreaker=None,
                concept_graph_embed_alpha=None,
                concept_graph_catrag_alpha=None,
                user=user,
            )
        else:
            query_embedding = await request.app.state.EMBEDDING_FUNCTION(
                form_data.query, prefix=RAG_EMBEDDING_QUERY_PREFIX, user=user
            )
            # query_doc wraps a blocking VECTOR_DB_CLIENT.search call;
            # offload so the request's event loop stays responsive.
            return await asyncio.to_thread(
                query_doc,
                collection_name=form_data.collection_name,
                query_embedding=query_embedding,
                k=form_data.k if form_data.k else request.app.state.config.TOP_K,
                user=user,
            )
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )


class QueryCollectionsForm(BaseModel):
    collection_names: list[str]
    query: str
    k: int | None = None
    k_reranker: int | None = None
    r: float | None = None
    hybrid: bool | None = None
    hybrid_bm25_weight: float | None = None
    enable_enriched_texts: bool | None = None


@router.post('/query/collection')
async def query_collection_handler(
    request: Request,
    form_data: QueryCollectionsForm,
    user=Depends(get_verified_user),
):
    await _validate_collection_access(form_data.collection_names, user)

    try:
        if request.app.state.config.ENABLE_RAG_HYBRID_SEARCH and (form_data.hybrid is None or form_data.hybrid):
            return await query_collection_with_hybrid_search(
                collection_names=form_data.collection_names,
                queries=[form_data.query],
                embedding_function=lambda query, prefix: request.app.state.EMBEDDING_FUNCTION(
                    query, prefix=prefix, user=user
                ),
                k=form_data.k if form_data.k else request.app.state.config.TOP_K,
                reranking_function=(
                    (lambda query, documents: request.app.state.RERANKING_FUNCTION(query, documents, user=user))
                    if request.app.state.RERANKING_FUNCTION
                    else None
                ),
                k_reranker=form_data.k_reranker or request.app.state.config.TOP_K_RERANKER,
                r=(form_data.r if form_data.r else request.app.state.config.RELEVANCE_THRESHOLD),
                hybrid_bm25_weight=(
                    form_data.hybrid_bm25_weight
                    if form_data.hybrid_bm25_weight
                    else request.app.state.config.HYBRID_BM25_WEIGHT
                ),
                enable_enriched_texts=(
                    form_data.enable_enriched_texts
                    if form_data.enable_enriched_texts is not None
                    else request.app.state.config.ENABLE_RAG_HYBRID_SEARCH_ENRICHED_TEXTS
                ),
                concept_graph_store=getattr(request.app.state, 'concept_graph_store', None),
                **_build_concept_graph_extras(request.app.state),  # W6.10: sync embed_fn + name-only cosine reranker
                concept_graph_tiebreaker=None,
                concept_graph_embed_alpha=None,
                concept_graph_catrag_alpha=None,
            )
        else:
            return await query_collection(
                request,
                collection_names=form_data.collection_names,
                queries=[form_data.query],
                embedding_function=lambda query, prefix: request.app.state.EMBEDDING_FUNCTION(
                    query, prefix=prefix, user=user
                ),
                k=form_data.k if form_data.k else request.app.state.config.TOP_K,
            )

    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )


####################################
#
# Vector DB operations
#
####################################


class DeleteForm(BaseModel):
    collection_name: str
    file_id: str


@router.post('/delete')
async def delete_entries_from_collection(
    form_data: DeleteForm,
    user=Depends(get_admin_user),
    db: AsyncSession = Depends(get_async_session),
):
    try:
        if await ASYNC_VECTOR_DB_CLIENT.has_collection(collection_name=form_data.collection_name):
            file = await Files.get_file_by_id(form_data.file_id, db=db)
            if not file:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=ERROR_MESSAGES.NOT_FOUND,
                )
            hash = file.hash

            # Refuse to issue a `filter={'hash': None}` query — the
            # match semantics of a null filter value are
            # backend-dependent (some backends ignore the key, some
            # match every row whose metadata lacks `hash`) and risk
            # deleting unrelated entries. Files without a hash are
            # typically unprocessed / failed / legacy records that
            # can't be targeted by hash anyway.
            if hash is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=ERROR_MESSAGES.DEFAULT('File has no hash; cannot delete vector entries by hash.'),
                )

            # Pre-existing bug: this used `metadata=` which is not a
            # parameter on `VectorDBBase.delete` nor on any backend
            # implementation, so the call always raised TypeError that
            # was silently swallowed by the surrounding `except
            # Exception` and the endpoint reported `{'status': False}`
            # for every request. Use `filter` to actually do what the
            # endpoint name promises.
            await ASYNC_VECTOR_DB_CLIENT.delete(
                collection_name=form_data.collection_name,
                filter={'hash': hash},
            )
            return {'status': True}
        else:
            return {'status': False}
    except HTTPException:
        # Caller-meaningful errors (404/400 above) must not be
        # swallowed and re-shaped as `{'status': False}`.
        raise
    except Exception as e:
        log.exception(e)
        return {'status': False}


@router.post('/reset/db')
async def reset_vector_db(user=Depends(get_admin_user), db: AsyncSession = Depends(get_async_session)):
    await ASYNC_VECTOR_DB_CLIENT.reset()
    await Knowledges.delete_all_knowledge(db=db)


@router.post('/reset/uploads')
async def reset_upload_dir(user=Depends(get_admin_user)) -> bool:
    folder = f'{UPLOAD_DIR}'
    try:
        # Check if the directory exists
        if os.path.exists(folder):
            # Iterate over all the files and directories in the specified directory
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)  # Remove the file or link
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # Remove the directory
                except Exception as e:
                    log.exception(f'Failed to delete {file_path}. Reason: {e}')
        else:
            log.warning(f'The directory {folder} does not exist')
    except Exception as e:
        log.exception(f'Failed to process the directory {folder}. Reason: {e}')
    return True


if ENV == 'dev':

    @router.get('/ef/{text}')
    async def get_embeddings(request: Request, text: str | None = 'Hello World!'):
        return {'result': await request.app.state.EMBEDDING_FUNCTION(text, prefix=RAG_EMBEDDING_QUERY_PREFIX)}


class BatchProcessFilesForm(BaseModel):
    files: list[FileModel]
    collection_name: str


class BatchProcessFilesResult(BaseModel):
    file_id: str
    status: str
    error: str | None = None


class BatchProcessFilesResponse(BaseModel):
    results: list[BatchProcessFilesResult]
    errors: list[BatchProcessFilesResult]


@router.post('/process/files/batch')
async def process_files_batch(
    request: Request,
    form_data: BatchProcessFilesForm,
    user=Depends(get_verified_user),
    db=None,
) -> BatchProcessFilesResponse:
    """
    Process a batch of files and save them to the vector database.

    NOTE: We intentionally do NOT use Depends(get_async_session) here.
    The save_docs_to_vector_db() call makes external embedding API calls which
    can take 5-60+ seconds for batch operations. Database operations after
    embedding (Files.update_file_by_id) manage their own short-lived sessions.
    """

    collection_name = form_data.collection_name

    if collection_name:
        await _validate_collection_access([collection_name], user, access_type='write')

    file_results: list[BatchProcessFilesResult] = []
    file_errors: list[BatchProcessFilesResult] = []
    file_updates: list[FileUpdateForm] = []

    # Prepare all documents first
    all_docs: list[Document] = []

    for file in form_data.files:
        try:
            # Ownership check: verify the requesting user owns the file or is an admin
            db_file = await Files.get_file_by_id(file.id, db=db)
            if not db_file:
                file_errors.append(
                    BatchProcessFilesResult(
                        file_id=file.id,
                        status='failed',
                        error='File not found',
                    )
                )
                continue
            if db_file.user_id != user.id and user.role != 'admin':
                file_errors.append(
                    BatchProcessFilesResult(
                        file_id=file.id,
                        status='failed',
                        error='Permission denied: not file owner',
                    )
                )
                continue

            text_content = file.data.get('content', '')
            docs: list[Document] = [
                Document(
                    page_content=text_content.replace('<br/>', '\n'),
                    metadata={
                        **file.meta,
                        'name': file.filename,
                        'created_by': file.user_id,
                        'file_id': file.id,
                        'source': file.filename,
                    },
                )
            ]

            all_docs.extend(docs)

            file_updates.append(
                FileUpdateForm(
                    hash=calculate_sha256_string(text_content),
                    data={'content': text_content},
                )
            )
            file_results.append(BatchProcessFilesResult(file_id=file.id, status='prepared'))

        except Exception as e:
            log.error(f'process_files_batch: Error processing file {file.id}: {str(e)}')
            file_errors.append(BatchProcessFilesResult(file_id=file.id, status='failed', error=str(e)))

    # Save all documents in one batch
    if all_docs:
        try:
            await run_in_threadpool(
                save_docs_to_vector_db,
                request,
                all_docs,
                collection_name,
                add=True,
                user=user,
            )

            # Update all files with collection name
            for file_update, file_result in zip(file_updates, file_results):
                await Files.update_file_by_id(id=file_result.file_id, form_data=file_update, db=db)
                file_result.status = 'completed'

        except Exception as e:
            log.error(f'process_files_batch: Error saving documents to vector DB: {str(e)}')
            for file_result in file_results:
                file_result.status = 'failed'
                file_errors.append(BatchProcessFilesResult(file_id=file_result.file_id, status='failed', error=str(e)))

    return BatchProcessFilesResponse(results=file_results, errors=file_errors)
