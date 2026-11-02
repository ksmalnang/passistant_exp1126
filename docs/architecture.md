# Architecture

## System Overview

PASsistant is a Retrieval-Augmented Generation (RAG) chatbot built on LangGraph. It processes academic documents (PDF) through OCR, indexes them into a vector database, and answers student questions by retrieving relevant context and generating grounded responses.

## Component Diagram

```mermaid
flowchart TB
    subgraph API["API Layer"]
        FastAPI[FastAPI REST + WebSocket (SSE Support)]
        Telegram[Telegram Bot Adapter]
    end

    subgraph Graph["LangGraph Workflow"]
        Router[Router Node]
        Retrieval[Retrieval Node]
        Response[Response Node]
        Fallback[Fallback Response Node]
        ErrorNode[Error Handler Node]
        OutputGuard[Output Guard]
        DocProcessor[Document Processor]
        StudentHandler[Student Record Handler]
    end

    subgraph Services["Service Layer"]
        Intent[IntentClassifier]
        ResponseGen[ResponseGeneration]
        DocProcessing[DocumentProcessing]
        IngestionHealth[IngestionHealth]
        Sessions[InMemorySessionManager]
        RateLimit[InMemoryRateLimiter]
    end

    subgraph Infra["Infrastructure"]
        Qdrant[(Qdrant)]
        Redis[(Redis Cache)]
        LLM[OpenAI / OpenRouter]
        Zhipu[Zhipu AI GLM-4 OCR]
    end

    API --> Graph
    Graph --> Services
    Services --> Infra
```

## Request Lifecycle

```mermaid
sequenceDiagram
    participant U as User
    participant API as FastAPI
    participant IG as Input Guard
    participant R as Router
    participant RN as Retrieval Node
    participant LLM as LLM Provider
    participant Q as Qdrant
    participant OG as Output Guard

    U->>API: POST /chat (or SSE /chat/stream)
    API->>IG: validate(message)
    IG-->>API: safe ✓
    API->>R: classify intent
    R-->>API: query_document
    API->>RN: run(state)
    RN->>LLM: rewrite query
    LLM-->>RN: keyword query
    par Parallel Search
        RN->>Q: search(query_1)
        RN->>Q: search(query_2)
        RN->>Q: search(query_3)
    end
    Q-->>RN: candidates
    RN->>LLM: rerank candidates
    LLM-->>RN: scored results
    RN-->>API: retrieved_chunks
    API->>LLM: generate response
    LLM-->>API: answer
    API->>OG: filter(answer)
    OG-->>API: safe answer
    API-->>U: ChatResponse (or SSE Events)
```

## SSE Streaming Support

The FastAPI layer supports streaming interactions via Server-Sent Events (SSE). It emits 6 event types during a run:
1. `run.started`
2. `run.status`
3. `message.delta`
4. `run.completed`
5. `run.failed`
6. `run.cancelled`

It also supports seamless resumption using the `Last-Event-ID` header or query parameter.
HTTP, SSE, and WebSocket chat runs share the same process-local registry.
`POST /chat/{thread_id}/cancel` cancels the workflow task, closes its agent
iterator, and publishes a terminal cancellation event. Disconnecting a subscriber
preserves the run for replay. See [Stop API](api.md#post-chatthread_idcancel--stop)
for run-ID checks and cooperative-cancellation limits.

## Key Design Decisions

### Checkpoint Serialization

Default in-memory checkpointers share the factory in `src/graphs/checkpointing.py`.
Its `JsonPlusSerializer` explicitly allows PASsistant's state models, including
`Citation`, `DocumentUpload`, and `StudentRecord`. Checkpoint reads preserve these
types without unregistered-type warnings, including with
`LANGGRAPH_STRICT_MSGPACK=true`. Caller-injected checkpointers retain their own
serializer configuration.

### Chunked Document Indexing

Documents are OCR-processed and split into overlapping chunks using LangChain's
`RecursiveCharacterTextSplitter`. Each chunk is embedded and upserted into Qdrant
with page-number metadata derived from OCR page boundaries, enabling page-level
citations in retrieval results.

### Contextual Embedding

Each chunk payload includes document-level metadata (`doc_title`, `filename`,
`document_type`) and page-level `source_locations`, keeping broad context
available to semantic search and enabling accurate citations.

### Hybrid Retrieval with Reranking

Three retrieval strategies are supported:
- **similarity** — Dense cosine similarity only
- **rrf** — Reciprocal Rank Fusion of dense + BM25 sparse vectors
- **reranker** — First-stage RRF/similarity candidates re-scored by a cross-encoder

### Parallel Multi-Query

Up to 3 query variants (original, LLM-rewritten, expanded) are searched in parallel via `asyncio.gather`, reducing retrieval latency from 3x to 1x the single-query time.
