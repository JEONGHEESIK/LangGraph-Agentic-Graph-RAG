# LangGraph-Agentic-Graph RAG

<div align="center">

<img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python">
<img src="https://img.shields.io/badge/Framework-LangGraph-orange.svg" alt="LangGraph"> 
<img src="https://img.shields.io/badge/Inference-SGLang-green.svg" alt="SGLang"> 
<img src="https://img.shields.io/badge/DB-Neo4j-blue.svg" alt="Neo4j"> 
<img src="https://img.shields.io/badge/DB-Weaviate-green.svg" alt="Weaviate"> 
<br>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

English | [한국어](README/README_KO.md) | [中文](README/README_ZH.md)
</div> 

Traditional single-vector RAG systems face inherent limitations when dealing with complex, multi-hop queries, often remaining confined to passive information retrieval. To overcome this, I initially designed the Agentic Graph RAG—powered by LangGraph and SGLang—featuring an adaptive 3-way retrieval routing system to maximize reasoning accuracy and minimize unnecessary database traffic.

However, during the architectural design phase, I realized that by simply adding an intent-based Tool Routing layer and Model Context Protocol (MCP) on top of this robust retrieval foundation, it would be entirely possible to build a fully autonomous Agentic AI system.

With this high scalability in mind, the architecture was expanded. The resulting framework dynamically classifies user intent to seamlessly switch between deep knowledge retrieval and computational tasks (Calculator, SQL, APIs). Coupled with intelligent backtracking and Quality Gate validation, this project lays a highly scalable foundation for AI systems that not only retrieve information but actively think and execute tools.

LangGraph-Agentic-Graph RAG is a fully autonomous Agentic AI and vector–graph hybrid RAG platform powered by LangGraph + SGLang. The ingestion pipeline converts raw documents into Markdown chunks and graph metadata via LangGraph state machines with checkpoint persistence. During query time, an intent-based tool router dynamically delegates computational tasks to external tools (via MCP), while knowledge queries are routed through a hop-based router with quality-gate backtracking, selecting among three retrieval paths—Vector, Weaviate GraphRAG, or Neo4j GraphDB—to generate precise answers.

The system features:

- **Dual-mode query processing**: Automatically routes between knowledge-based RAG retrieval and computational tool execution based on LLM-powered intent classification
- **Advanced document ingestion**: Converts raw documents (PDF/images/audio) into Markdown chunks and structured graph metadata via LangGraph state machines with checkpoint persistence
- **Intelligent retrieval routing**: Hop-based router with quality-gate backtracking dynamically selects among three retrieval paths (Vector, Weaviate Cross-Reference GraphRAG, or Neo4j Deep Graph Traversal) based on query complexity
- **Tool calling framework**: MCP (Model Context Protocol) server integration with local fallback for calculator, API calls, code execution, and database queries
- **Graph-of-Thought reasoning**: Multi-branch exploration with snapshot-based backtracking for complex analytical queries

<div align="center">
<img src="https://github.com/user-attachments/assets/504ea0fa-ed9a-4664-9095-042e01debc65" width="512" height="768" ></img><br/>
</div> 

---

## Key Capabilities

- **LangGraph state machines**: All workflows (ingestion, query reasoning, tool execution, summarization, mindmap generation) run on LangGraph `StateGraph` with `MemorySaver` checkpointing for full state persistence and recovery.

- **Intelligent query routing**: LLM-powered intent classification automatically determines whether a query requires knowledge retrieval or computational tool execution:
  - **Knowledge queries** → RAG pipeline with 3-way retrieval routing
  - **Calculation queries** → Calculator tool with AST-based safe evaluation supporting advanced math (sqrt, log, trig, sigma)
  - **Database queries** → SQL executor (planned)
  - **API calls** → HTTP API caller with configurable endpoints
  - **Code execution** → Python sandbox with restricted built-ins

- **Checkpoint & intelligent backtracking**: Every node transition is checkpointed; the quality gate evaluates retrieval results and triggers intelligent path selection when quality is insufficient:
  - **Quality evaluation**: Observer LLM scores each path result (0.0–1.0) against `QUALITY_GATE_THRESHOLD`
  - **Smart path selection**: `PathSelector` analyzes remaining untried paths based on query keywords, hop count, and path characteristics to select the most suitable alternative
  - **Backtrack limits**: Configurable via `MAX_BACKTRACK_COUNT` to prevent infinite loops
  - **State tracking**: `tried_paths` field prevents re-attempting failed strategies

- **3-way retrieval routing**: Query complexity (hop count) determines the optimal retrieval strategy:
  - **Path 1 – Vector RAG (≤ 2 hops)**: Fast semantic similarity search on late-chunked TextDocument corpus via BM25 + reranker. Ideal for direct factual questions.
  - **Path 2 – Weaviate Cross-Reference GraphRAG (3–5 hops)**: BM25 seed entity search followed by multi-hop cross-reference traversal (source/target/event refs) within Weaviate. Surfaces query-adjacent entities and events through relationship walking.
  - **Path 3 – Neo4j Deep Graph Traversal (≥ 6 hops)**: Cypher-based deep graph exploration for schema-intensive relationship reasoning. Handles complex multi-entity queries requiring extensive graph traversal.
  - **Hop classification**: Hybrid LLM + heuristic approach estimates query complexity, with LLM primary classification and keyword-based fallback

- **Graph-of-Thought expansion**: Multi-branch reasoning with snapshot-based backtracking for complex analytical queries:
  - **Branch exploration**: Each step fans out `GOT_BRANCH_FACTOR` candidate queries in parallel
  - **Quality scoring**: Observer LLM evaluates each branch (0.0–1.0) for relevance, coverage, and novelty
  - **Intelligent merging**: Branches above `GOT_THOUGHT_SCORE_THRESHOLD` are merged via configurable strategy (`top_k`/`weighted_union`/`vote`)
  - **Edge pruning**: Low-quality connections removed by keyword-overlap scoring (`GOT_EDGE_PRUNE_THRESHOLD`)
  - **Failure recovery**: Consecutive all-branch failures trigger snapshot rollback to last successful merge point

- **SGLang inference ecosystem**: All LLM operations (generation, embedding, reranking, hop classification, quality evaluation) run on SGLang servers with intelligent lifecycle management:
  - **Lazy-loading architecture**: Servers auto-start on first request, eliminating cold-start overhead during initialization
  - **Idle timeout**: GPU memory automatically released after `SGLANG_IDLE_TIMEOUT` (default 60s) of inactivity
  - **GPU allocation**: Configurable device assignment and memory fractions per server (generator, embedding, reranker, refiner)
  - **Keepalive mechanism**: Background thread maintains server health during long-running operations
  - **Chunk retry logic**: LLM metadata extraction auto-retries failed chunks with server restart (configurable via `GRAPH_EXTRACTOR_RETRY_ON_FAILURE`)

- **Tool calling framework**: MCP (Model Context Protocol) server integration with local fallback:
  - **MCP-first architecture**: Primary execution via MCP server when available (`MCP_SERVER_ENABLED=true`)
  - **Local fallback**: Automatic fallback to local implementations when MCP server is unavailable
  - **Calculator**: AST-based safe expression evaluation supporting advanced math functions (sqrt, cbrt, log, exp, sin, cos, tan, sigma)
  - **Natural language parsing**: Converts Korean/English math expressions ("144의 제곱근", "루트 144") to executable code
  - **API caller**: HTTP request execution with configurable endpoints and methods
  - **Code runner**: Python sandbox with restricted built-ins and token filtering for security
  - **SQL executor**: Placeholder for future database query support

- **Automatic graph construction**: End-to-end pipeline from raw documents to queryable knowledge graph:
  - **OCR processing**: SGLang-powered OCRFlux converts documents to Markdown
  - **LLM extraction**: Entity/event/relation extraction from Markdown chunks with configurable chunk size and timeout
  - **Dual storage**: Simultaneous upsert to Weaviate (cross-reference graph) and Neo4j (deep graph) with deterministic UUIDs
  - **Schema management**: Automatic Weaviate collection creation with cross-reference definitions

- **Async job monitoring**: Full task lifecycle tracking through REST APIs:
  - **Upload progress**: Real-time status for file upload and processing stages
  - **OCR progress**: Per-page OCR completion tracking
  - **Embedding progress**: Chunk-level indexing status
  - **Task cancellation**: Graceful termination of long-running operations

---

## Architecture

```
┌──────────────┐    ┌─────────────────────────────┐
│ Input Layer  │ →  │ LangGraph Upload Pipeline   │ →  Markdown + *.graph.json
│ (PDF/IMG/…)  │    │ (MemorySaver checkpoint)    │
└──────────────┘    └─────────────────────────────┘
                                  │
┌──────────────┐    ┌─────────────────────────────┐
│ User Query   │ →  │ LangGraph RAG Workflow      │
└──────────────┘    │ (MemorySaver checkpoint)    │
                    └─────────────────────────────┘
                                  │
                    ┌─────────────┴──────────────┐
                    │  Tool Router (LLM intent)  │
                    │  (ToolExecutor.classify)   │
                    └─────────────┬──────────────┘
                                  │
                    ┌─────────────┴──────────────┐
                    │                            │
              Knowledge Query              Computational Task
                    │                            │
        ┌───────────┴──────────────┐        ┌───────┴────────┐
        │   RAG Router (reasoner)  │        │ Tool Executor  │
        │ HopClassifier+PathSelect │        │ (MCP/Local)    │
        └───────────┬──────────────┘        └────────────────┘
                    │
     ┌──────────────┼──────────────┐
     │              │              │
 Path 1        Path 2         Path 3
 Vector        CrossRef       GraphDB
 Search        Weaviate Ref   Neo4j Cypher
 (≤ 2 hop)     (3–5 hop)      (≥ 6 hop)
                    │
                    ▼
   ┌───────────────────────────────────────┐
   │ Quality Gate (QualityEvaluator)       │
   │ Observer LLM (QUALITY_GATE_THRESHOLD) │
   └────────────────┬──────────────────────┘
                    │
          ┌─────────▼──────────────┐
          │ GoT Thought Expander   │
          │ Branch merge + pruning │
          └─────────┬──────────────┘
                    │
             ┌──────▼──────┐
             │ LLM Answer  │
             └─────────────┘
```

### Core Components

**Reasoner Module** (`backend/notebooklm/reasoner/`):
- `HopClassifier`: Query complexity estimation (LLM + heuristic fallback)
- `PathSelector`: Optimal retrieval path selection for backtracking
- `QualityEvaluator`: Observer LLM-based result quality assessment
- `VectorRetriever`, `CrossRefRetriever`, `GraphDBRetriever`: Modular retrieval implementations

### Workflow Graph (Query)

```
planner → tool_router ┬→ rag_router →┬→ vector_retriever  ──→┐
                      │               ├→ crossref_retriever ─→├→ quality_gate →┬→ thought_expander → aggregator → END
                      │               └→ graphdb_retriever ──→┘                └→ rag_router (backtrack)
                      │
                      └→ tool_executor ────────────────────────────────────────→ aggregator → END
```

**Tool Router**: LLM-powered intent classification determines query routing strategy:
- **Classification endpoint**: Configurable via `TOOL_INTENT_CLASSIFIER_ENDPOINT` (defaults to SGLang generator)
- **Intent categories**: `knowledge`, `calculation`, `database`, `api_call`, `code_exec`
- **Routing logic**:
  - `knowledge` → RAG pipeline (3-way retrieval routing)
  - Other intents → Tool executor (MCP/local fallback)
- **Fallback mechanism**: Heuristic keyword matching when LLM classification fails

**Tool Executor** (`backend/notebooklm/tools/tool_executor.py`):
- **MCP integration**: Primary execution via MCP server REST API when `MCP_SERVER_ENABLED=true`
- **Local fallback**: Automatic fallback to local implementations when MCP unavailable
- **Intent classification**: LLM-based with heuristic fallback for reliability
- **Supported tools**:
  - **Calculator** (fully implemented):
    - AST-based safe expression evaluation (no `eval()` security risks)
    - Advanced math functions: `sqrt`, `cbrt`, `abs`, `log`, `ln`, `log10`, `exp`, `sin`, `cos`, `tan`, `sigma`
    - Natural language parsing: "144의 제곱근" → `sqrt(144)`, "2의 3제곱" → `2 ** 3`
    - Operator support: `+`, `-`, `*`, `/`, `**`, `%`
  - **API Caller** (lightweight implementation):
    - HTTP GET/POST request execution
    - URL extraction from natural language prompts
    - Configurable timeout and error handling
  - **Code Runner** (sandbox implementation):
    - Restricted Python execution environment
    - Filtered built-ins (no `eval`, `exec`, `__import__`)
    - Token-based security filtering
  - **SQL Executor** (placeholder):
    - Returns `not_implemented` status
    - Reserved for future database integration

---

## Input / Preprocessing

Handled by `LangGraphUploadPipeline` (`langgraph_upload_pipeline.py`) with `MemorySaver` checkpointing across all nodes:

1. **Conversion & Layout**: `run_file_processor.py` handles PDF/Office/image/audio inputs → `Results/1.Converted_images` + `Results/2.LayoutDetection`.
2. **OCR & Markdown**: `run_ocr_processing()` with SGLang-powered OCRFlux produces per-page Markdown → `Results/4.OCR_results`.
3. **LLM Metadata Extraction**: `LLMMetadataExtractor` extracts entities/events/relations from Markdown → `Results/8.graph_metadata/*.graph.json`.
   - **Chunk size**: configurable via `GRAPH_EXTRACTOR_CHUNK_SIZE`
   - **Timeout**: configurable via `GRAPH_EXTRACTOR_API_TIMEOUT`
   - **Retry logic**: On timeout, SGLang generator server restarts and retries the same chunk once
   - **Keepalive**: Background thread touches server every `SGLANG_KEEPALIVE_INTERVAL` seconds during processing
4. **Graph Upsert**:
   - `GraphSchemaManager` ensures Weaviate GraphEntity/GraphEvent/GraphRelation collections exist with cross-references (source/target/event)
   - `LegacyGraphIngestor` / `Neo4jManager` MERGEs nodes/relationships into Neo4j with deterministic UUIDs
5. **Late Chunking & Embedding**: `embedding_text.py` splits Markdown into chunks and uploads into the Weaviate TextDocument collection via `SharedEmbeddingModel` (model configurable via `EMBEDDING_MODEL`).

---

## Module Map

```
backend/
├── main.py                          # [Excluded] FastAPI server entry point
├── config.py                        # [Excluded] Server-level configuration
├── logging_config.py                # [Excluded] Logging configuration
│
├── api/                             # API layer
│   ├── routes.py                    # [Excluded] Main upload/file/session routes
│   ├── chat.py                      # [Excluded] POST /v1/chat endpoint
│   ├── ocr_routes.py                # [Excluded] OCR processing endpoints
│   └── pause_api.py                 # [Excluded] Task pause/resume API
│
├── notebooklm/                      # RAG core modules
│   ├── config.py                    # Model/path/graph configuration
│   ├── rag_pipeline.py              # LangGraph RAG workflow orchestrator
│   ├── graph_reasoner.py            # LangGraph workflow orchestration
│   ├── graph_schema.py              # Weaviate Entity/Event/Relation schema
│   ├── hop_classifier.py            # Query complexity estimator
│   ├── reasoner/                    # Refactored GraphReasoner modules
│   │   ├── state.py                 # GraphReasonerState definition
│   │   ├── routing.py               # PathSelector, HopClassifier
│   │   ├── quality.py               # QualityEvaluator
│   │   ├── retrievers.py            # VectorRetriever, CrossRefRetriever, GraphDBRetriever
│   │   └── __init__.py
│   ├── legacy_graph_client.py       # Neo4j Cypher traversal client
│   ├── legacy_graph_ingestor.py     # Neo4j upsert helper
│   ├── embedding_text.py            # Late chunking + Weaviate text indexing
│   ├── embedding_image.py           # Image embedding + Weaviate image indexing
│   ├── image_processor.py           # Image processing utilities
│   ├── shared_embedding.py          # SGLang embedding/reranker client (singleton)
│   ├── sglang_server_manager.py     # SGLang server lifecycle manager
│   ├── generator.py                 # LLM answer generation
│   ├── refiner.py                   # Answer refinement
│   ├── evaluator.py                 # Answer quality evaluation
│   ├── router.py                    # Query type routing
│   ├── query_rewriter.py            # Query rewriting
│   ├── parallel_search.py           # Parallel text+image search
│   ├── weaviate_utils.py            # Weaviate client utilities
│   ├── clean_weaviate.py            # [Excluded] Weaviate + Neo4j data cleanup script
│   ├── tools/                       # Tool calling & MCP integration
│   │   ├── mcp_client.py            # MCP server REST client
│   │   ├── tool_executor.py         # Tool execution (MCP/local fallback)
│   │   └── __init__.py
│   ├── rag_text/                    # Text search + reranker
│   └── rag_image/                   # Image search + reranker
│
├── mcp_server/                      # MCP (Model Context Protocol) Tool Server
│   └── main.py                      # FastAPI server for tool execution
│
├── data_pipeline/                    # Data processing pipeline
│   └── pipe/
│       ├── langgraph_upload_pipeline.py  # LangGraph upload workflow (checkpointed)
│       ├── llm_metadata_extractor.py     # Entity/event/relation extraction
│       ├── neo4j_manager.py              # Neo4j upsert manager
│       ├── run_file_processor.py         # Convert/layout/OCR orchestrator
│       ├── pipeline_image.py             # Image pipeline
│       ├── pipeline_sound.py             # Audio pipeline
│       └── main_pipe/
│           ├── ocr_pipe/                 # SGLang-based OCRFlux engine
│           ├── udp_pdftopng_300dpi.py    # PDF → PNG conversion
│           └── udp_layoutdetection.py    # Layout detection
│
├── services/                        # Business logic services
│   ├── model_manager.py             # [Excluded] LazyModelManager (GPU lifecycle)
│   ├── ocr_vision_manager.py        # [Excluded] OCR engine management
│   └── rag_service.py               # [Excluded] RAG service orchestration
│
└── utils/
    ├── task_queue.py                # [Excluded] GPU task queue (async job management)
    ├── helpers.py                   # [Excluded] Shared utility functions
    ├── path_helpers.py              # [Excluded] Path calculation helpers
    └── file_utils.py                # [Excluded] File operation utilities
```

---

## Data Pipeline

1. **File upload** (`POST /upload/files`)
   - `api/routes.py` stores files in per-session folders and enqueues `run_processing_pipeline` via `task_queue.py`.
2. **GPU task queue**
   - `task_queue.py` manages sequential GPU-bound tasks (convert → layout → OCR) with progress tracking.
3. **Text indexing** (`run_text_indexing` / `run_text_indexing_v2`)
   - Initializes `SharedEmbeddingModel` → runs `process_markdown_files` for Weaviate late-chunking indexing.
4. **Graph extraction & ingestion**
   - `LLMMetadataExtractor` produces `*.graph.json` → `GraphSchemaManager` upserts to Weaviate → `Neo4jManager` / `LegacyGraphIngestor` upserts to Neo4j.
5. **Storage state**
   - Weaviate: TextDocument + GraphEntity/Event/Relation collections.
   - Neo4j: Entity/Event nodes + relation edges with deterministic UUIDs and auto-created constraints.

---

## Query Processing Flow

1. **Request initiation**: `POST /v1/chat` → `RAGPipeline.process_query()` invokes `GraphReasoner.retrieve()`

2. **LangGraph workflow execution** (`graph_reasoner.py`):

   * **Step 1: Planner** (`planner` node)
     - Analyzes user query and extracts key concepts
     - Records high-level search plan and reasoning steps
     - Initializes query history for multi-turn context
     - **Output**: `plan`, `query_analysis`

   * **Step 2: Tool Router** (`tool_router` node)
     - LLM classifies query intent via `ToolExecutor.classify_intent()`:
       - `knowledge` → routes to **Branch A** (RAG Router)
       - `calculation`, `database`, `api_call`, `code_exec` → routes to **Branch B** (Tool Executor)
     - **Output**: `intent`, routing decision

   **[Branch A: Knowledge Query Path]**

   * **Step 3a: RAG Router** (`rag_router` node, for knowledge queries)
     - Performs hop classification (LLM + heuristic hybrid)
     - Sets `max_hops` = `min(llm_estimate, heuristic_estimate, GRAPH_MAX_HOPS)`
     - Selects initial retrieval path:
       - hop ≤ 2 → `vector_retriever` (Path 1)
       - hop 3–5 → `crossref_retriever` (Path 2)
       - hop ≥ 6 → `graphdb_retriever` (Path 3)
     - **Output**: `max_hops`, `retrieval_path`, `tried_paths`

   * **Step 4a: Retrieval Execution** (Path 1/2/3)
     - **Path 1**: semantic search + reranker on TextDocument
     - **Path 2**: BM25 seed search + Weaviate cross-reference multi-hop traversal
     - **Path 3**: Neo4j Cypher deep graph traversal
     - **Output**: `context_snippets`, `entities`, `events`, `relations`

   * **Step 5a: Quality Gate** (`quality_gate` node)
     - Observer LLM scores retrieval result (0.0–1.0)
     - **If quality ≥ `QUALITY_GATE_THRESHOLD`**: Proceeds to aggregator or thought expander
     - **If quality < threshold**: Triggers intelligent backtracking (`PathSelector` analyzes remaining paths, selects alternative, and returns to Step 4a)
     - **Termination**: After `MAX_BACKTRACK_COUNT` retries or all paths exhausted
     - **Output**: `retrieval_quality`, `backtrack_count`, `tried_paths`

   * **Step 6a: Thought Expander** (`thought_expander` node, if `GOT_MODE_ENABLED=true`)
     - Fans out `GOT_BRANCH_FACTOR` candidate queries in parallel
     - Observer LLM scores each branch (0.0–1.0)
     - Merges branches above `GOT_THOUGHT_SCORE_THRESHOLD` and prunes low-quality edges
     - Snapshot-based backtracking on consecutive failures
     - **Output**: `thought_steps`, expanded context

   **[Branch B: Tool Execution Path]**

   * **Step 3b: Tool Executor** (`tool_executor` node, for computational tasks)
     - Maps intent to specific tool (`calculation`, `api_call`, `code_exec`, `database`)
     - Prepares tool inputs (e.g., converts natural language "144의 제곱근" → `sqrt(144)`)
     - Executes tool (MCP-first architecture with local fallback)
     - **Output**: `tool_result` with status (`ok`/`error`), result value, metadata (Note: Tool failures return error messages; does NOT fall back to RAG)

   **[Merge Node]**

   * **Step 7: Aggregator** (`aggregator` node)
     - **For RAG queries**: Builds context snippets from entities/events/relations/thoughts
     - **For tool queries**: Formats tool result into natural language (`"expression = result"`, etc.)
     - Collects metadata (`context_snippets`, `thought_steps`, `backtrack_count`, `tried_paths`, `tool_result`)
     - **Output**: Aggregated context or formatted tool result

3. **Answer generation**:
   - **For RAG queries**: `generator.py` synthesizes answer from original query + context snippets
   - **For tool queries**: Returns formatted tool result directly (no LLM generation needed)
   - **Optional post-processing**: `refiner.py` polishes answer, `evaluator.py` logs quality notes

4. **Response construction** (`RAGPipeline._build_response()` or `_build_tool_only_response()`):
   - **For RAG queries**: Full response with answer, context, snippets, search results
   - **For tool queries**: Streamlined response with tool result as answer
   - **Metadata included**: `plan`, `max_hops`, `retrieval_quality`, `backtrack_count`, `tried_paths`, `thought_steps`, `tool_result`
   - **Debugging support**: All workflow state exposed for traceability

5. **Return to client**: JSON response with answer, metadata, and debugging information

---

## Key Settings (`notebooklm/config.py`)

- **Graph RAG toggles**: `GRAPH_RAG_ENABLED`, `LANGGRAPH_ENABLED`, `GOT_MODE_ENABLED`, `GRAPH_MAX_HOPS`.
- **GoT tuning**: `GOT_BRANCH_FACTOR`, `GOT_MERGE_STRATEGY` (`top_k`/`weighted_union`/`vote`), `GOT_MERGE_TOP_K`, `GOT_THOUGHT_SCORE_THRESHOLD`, `GOT_EDGE_PRUNE_THRESHOLD`, `GOT_MAX_STEPS`, `GOT_MAX_CONSECUTIVE_FAILURES`, `GOT_OBSERVER_ENDPOINT`/`GOT_OBSERVER_MODEL`.
- **Neo4j**: `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `GRAPH_MAX_HOPS`.
- **Weaviate**: `WEAVIATE_HOST/PORT`, `WEAVIATE_TEXT_CLASS` (TextDocument), `WEAVIATE_VECTORIZER` (text2vec-model2vec).
- **SGLang models**: `LLM_MODEL`, `EMBEDDING_MODEL`, `RERANKER_MODEL`, `REFINER_MODEL`, `QUERY_REWRITER_MODEL`.
- **SGLang servers**: `SGLANG_GENERATOR_ENDPOINT`, `SGLANG_EMBEDDING_ENDPOINT`, `SGLANG_RERANKER_ENDPOINT`, `SGLANG_REFINER_ENDPOINT`, `SGLANG_QUERY_REWRITER_ENDPOINT`.
- **SGLang lifecycle**: `SGLANG_IDLE_TIMEOUT` (60s), `SGLANG_KEEPALIVE_INTERVAL` (20s).
- **Graph extractor**: `GRAPH_EXTRACTOR_API_TIMEOUT` (60s), `GRAPH_EXTRACTOR_CHUNK_SIZE` (800), `GRAPH_EXTRACTOR_RETRY_ON_FAILURE` (true).
- **Session directories**: `DATA_ROOT/Results`, `sessions/<id>` layout.

---

## MCP Tool Server

The MCP (Model Context Protocol) server is an independent FastAPI service that handles tool execution for the LangGraph RAG system. It provides a REST API for executing computational tools like calculators, SQL queries, API calls, and code execution.

### Features

- **Calculator**: AST-based safe mathematical expression evaluation
- **SQL Executor**: SQL query execution (extensible)
- **API Caller**: External API invocation (extensible)
- **Code Runner**: Code execution sandbox (extensible)

---

### Adding New Tools

1. Implement executor function in `main.py`
2. Register in `TOOL_REGISTRY`
3. Restart server

```
def execute_my_tool(inputs: Dict[str, Any]) -> ToolExecuteResponse:
    # Implementation
    return ToolExecuteResponse(status="ok", result=...)

TOOL_REGISTRY["my_tool"] = {
    "executor": execute_my_tool,
    "info": ToolInfo(
        name="my_tool",
        description="Description of my tool",
        parameters={...}
    )
}
```

---

## API Highlights

| Endpoint | Description |
|---|---|
| `POST /upload/files` | Triggers the LangGraph upload pipeline |
| `POST /v1/chat` | Runs 3-way RAG with checkpoint/backtracking |
| `GET /api/v1/tasks/{task_id}` | Monitors queued upload/OCR tasks |
| `GET /files` | Lists session artifacts |
| `POST /pause` | Pauses/resumes background tasks |

---

## Logging & Operations

- `sglang_embedding_server.log`, `sglang_reranker_server.log` – SGLang model server health.
- `Results/8.graph_metadata/*.graph.json` – archive of LLM extraction results.
- `LegacyGraphIngestor` auto-creates constraints on first run; no manual setup required.
- `SGLangServerManager` releases GPU memory after 60 seconds of idling (configurable via `SGLANG_IDLE_TIMEOUT`).
- All LangGraph workflows log checkpoint IDs and backtrack counts for traceability.
- **SGLang cold start**: lazy loading means the first `/v1/chat` (or hop-classifier) request must warm each SGLang server, which can take 20–60s VRAM load time; issue a warm-up request or keep-alive cron to avoid client timeouts.
- **Chunk retry mechanism**: If LLM metadata extraction times out (default 60s), the generator server is automatically restarted and the same chunk is retried once. This prevents hanging on problematic chunks while maintaining extraction quality.
- **Volatile checkpoints**: `MemorySaver` stores graph snapshots in-process, so any FastAPI restart drops in-flight state until the planned migration to `SqliteSaver`/`PostgresSaver` lands.
- **Weaviate v4 API**: Uses `weaviate.connect_to_custom()` with gRPC support (port 50051).

---

## Roadmap

1. ~~**GoT (Graph of Thought)**~~
   - `thought_expander` now performs graph-shaped exploration: each step fans out `GOT_BRANCH_FACTOR` branches, an observer LLM scores each branch, and the best results are merged via `GOT_MERGE_STRATEGY`. Low-quality edges are pruned, and consecutive failures trigger snapshot-based backtracking.
2. **Advanced hop classifier**
   - Augment with query metadata (token length, entity counts) for a hybrid router.
3. **Multi-graph retrieval optimization**
   - Improve context filtering/dedup for 3–5 hop Weaviate traversals and add Cypher templates for ≥ 6 hop Neo4j exploration.
4. **LangGraph workflow observability**
   - Emit per-node latency/error metrics and integrate retry policies inside `GraphReasoner` and `LegacyGraphClient`.
5. **Persistent checkpointer**
   - Migrate from `MemorySaver` to `SqliteSaver` / `PostgresSaver` for cross-session state recovery.

---

## Contribution & Contact

Issues and PRs are welcome. For questions or concerns, please open an issue on GitHub or email us at **koto144@gmail.com**.

---

## License

This project is dual-licensed under:
- **MIT License** - see the [LICENSE](LICENSE) file for details
- **Apache License 2.0** - see the [LICENSE-APACHE](LICENSE-APACHE) file for details

You may choose either license to govern your use of this software.

---

## Citation

If you use this project in your research, please cite the following:

### SGLang
```bibtex
@misc{zheng2023sglang,
  title={SGLang: Efficient Execution of Structured Language Model Programs},
  author={Lianmin Zheng and Liangsheng Yin and Zhiqiang Xie and Jeff Huang and Chuyue Sun and Cody Hao Yu and Shiyi Cao and Christos Kozyrakis and Ion Stoica and Joseph E. Gonzalez and Clark Barrett and Ying Sheng},
  year={2023},
  url={https://github.com/sgl-project/sglang}
}
```

### LangGraph
```bibtex
@software{langgraph2024,
  title={LangGraph: A Framework for Building Stateful Multi-Actor Applications},
  author={LangChain AI},
  year={2024},
  url={https://github.com/langchain-ai/langgraph}
}
```

### Weaviate
```bibtex
@software{weaviate2024,
  title={Weaviate: An Open-Source Vector Database},
  author={Weaviate B.V.},
  year={2024},
  url={https://github.com/weaviate/weaviate}
}
```

### Neo4j
```bibtex
@software{neo4j2024,
  title={Neo4j: The Graph Database Platform},
  author={Neo4j, Inc.},
  year={2024},
  url={https://github.com/neo4j/neo4j}
}
```

### OCRFlux
```bibtex
@software{ocrflux2024,
  title={OCRFlux: Vision-Language Model for OCR},
  author={ChatDOC},
  year={2024},
  url={https://huggingface.co/ChatDOC/OCRFlux-3B}
}
```