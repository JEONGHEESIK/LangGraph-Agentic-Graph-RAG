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

[English](../README.md) | 한국어 | [中文](README_ZH.md)
</div>

기존의 단일 벡터(Single-vector) RAG 시스템은 복잡한 다단계(Multi-hop) 질의를 처리하는 데 한계가 있었으며, 수동적인 정보 검색 역할에 머무르는 문제가 있었습니다. 이를 극복하기 위해 처음에는 LangGraph와 SGLang을 기반으로 추론 정확도를 높이고 불필요한 데이터베이스 트래픽을 최소화하는 3-Way 라우팅 기반의 Agentic Graph RAG를 설계했습니다.

하지만 아키텍처를 구체화하는 과정에서, 이 견고한 검색 파이프라인 위에 사용자의 의도를 파악하는 **라우팅 레이어(Tool Router)**와 **MCP(Model Context Protocol)**만 더 얹는다면 완전한 자율형 Agentic AI 시스템을 구축하는 것이 가능하겠다는 판단이 들었습니다.

이러한 확장성을 깊이 고려하여 아키텍처를 발전시킨 결과, 현재의 프레임워크가 완성되었습니다. 이 시스템은 사용자의 의도를 동적으로 분류하여 깊이 있는 지식 검색(Graph RAG)과 연산 작업(계산기, SQL, API 등) 사이를 매끄럽게 전환합니다. 지능형 백트래킹(Intelligent backtracking) 및 품질 검증(Quality Gate) 로직과 결합된 이 프로젝트는, 단순한 질의응답을 넘어 능동적으로 사고하고 도구를 실행하는 AI 시스템을 위한 가장 확장성 높은 기반을 제공합니다.

LangGraph-Agentic-Graph RAG는 LangGraph와 SGLang을 기반으로 구동되는 완전 자율형 에이전틱 AI(Agentic AI)이자 벡터-그래프 하이브리드 RAG 플랫폼입니다. 데이터 수집(Ingestion) 파이프라인은 체크포인트 영속성을 지원하는 LangGraph 상태 머신을 통해 원본 문서를 마크다운 청크(Chunk)와 그래프 메타데이터로 변환합니다. 쿼리 시점에는 의도 기반 툴 라우터가 계산 및 외부 작업을 MCP를 통해 외부 도구로 동적 위임합니다. 반면 지식 기반 쿼리는 퀄리티 게이트 백트래킹이 적용된 홉(Hop) 기반 라우터를 거쳐 세 가지 검색 경로(Vector, Weaviate GraphRAG, Neo4j GraphDB) 중 최적의 경로를 선택하여 정교한 답변을 생성합니다.

시스템 주요 특징:

- **이중 모드 쿼리 처리**: LLM 기반 의도 분류를 통해 지식 기반 RAG 검색과 연산 도구 실행 사이를 자동으로 라우팅합니다.
- **고급 문서 수집**: 체크포인트 영속성을 지원하는 LangGraph 상태 머신을 통해 원본 문서(PDF/이미지/오디오)를 마크다운 청크(Chunk)와 구조화된 그래프 메타데이터로 변환합니다.
- **지능형 검색 라우팅**: 쿼리 복잡도에 따라 세 가지 검색 경로(Vector, Weaviate Cross-Reference GraphRAG, Neo4j Deep Graph Traversal) 중 하나를 동적으로 선택하며, 품질 게이트 백트래킹을 지원하는 홉(Hop) 기반 라우터입니다.
- **도구 호출 프레임워크**: 계산기, API 호출, 코드 실행 및 데이터베이스 쿼리를 위한 로컬 폴백(Fallback) 기능이 포함된 MCP(Model Context Protocol) 서버 통합.
- **Graph-of-Thought 추론**: 복잡한 분석 쿼리를 위해 스냅샷 기반 백트래킹을 지원하는 다중 분기(Multi-branch) 탐색 기능.

<div align="center">
<img src="https://github.com/user-attachments/assets/504ea0fa-ed9a-4664-9095-042e01debc65" width="512" height="768" ></img><br/>
</div>

---

## 주요 기능 (Key Capabilities)

- **LangGraph 상태 머신**: 모든 워크플로우(수집, 쿼리 추론, 툴 실행, 요약, 마인드맵 생성)가 전체 상태 영속성과 복구를 위해 `MemorySaver` 체크포인팅을 사용하는 LangGraph `StateGraph` 위에서 동작합니다.

- **지능형 쿼리 라우팅**: LLM 기반 의도 분류가 쿼리에 지식 검색이 필요한지 연산 도구 실행이 필요한지 자동으로 결정합니다:
  - **지식 쿼리** → 3-Way 검색 라우팅이 적용된 RAG 파이프라인
  - **계산 쿼리** → 고급 수학(sqrt, log, trig, sigma)을 지원하는 AST 기반 안전 평가 계산기 도구
  - **데이터베이스 쿼리** → SQL 실행기 (예정)
  - **API 호출** → 구성 가능한 엔드포인트를 갖춘 HTTP API 호출기
  - **코드 실행** → 제한된 내장 함수를 갖춘 Python 샌드박스

- **체크포인트 & 지능형 백트래킹**: 모든 노드 전환이 체크포인트됩니다; 품질 게이트가 검색 결과를 평가하고 품질이 불충분할 때 지능형 경로 선택을 트리거합니다:
  - **품질 평가**: 관찰자 LLM이 각 경로 결과를 `QUALITY_GATE_THRESHOLD`를 기준으로 평가 (0.0–1.0)
  - **스마트 경로 선택**: `PathSelector`가 쿼리 키워드, 홉 수 및 경로 특성을 기반으로 남은 미시도 경로를 분석하여 가장 적합한 대안을 선택
  - **백트래킹 제한**: 무한 루프 방지를 위해 `MAX_BACKTRACK_COUNT`로 설정 가능
  - **상태 추적**: `tried_paths` 필드가 실패한 전략의 재시도를 방지

- **3-Way 검색 라우팅**: 쿼리 복잡도(홉 수)가 최적의 검색 전략을 결정합니다:
  - **Path 1 – Vector RAG (≤ 2 홉)**: BM25 + 리랭커를 통한 Late-chunked TextDocument 코퍼스 상의 빠른 의미 유사성 검색. 직접적인 사실 확인 질문에 이상적입니다.
  - **Path 2 – Weaviate Cross-Reference GraphRAG (3–5 홉)**: BM25 시드 엔티티 검색 후 Weaviate 내에서 다중 홉 상호 참조 탐색(source/target/event refs). 관계를 탐색하여 쿼리와 인접한 엔티티 및 이벤트를 발굴합니다.
  - **Path 3 – Neo4j Deep Graph Traversal (≥ 6 홉)**: 스키마 중심의 관계 추론을 위한 Cypher 기반 심층 그래프 탐색. 광범위한 그래프 탐색이 필요한 복잡한 다중 엔티티 쿼리를 처리합니다.
  - **홉 분류**: 하이브리드 LLM + 휴리스틱 접근법으로 쿼리 복잡도를 추정하며, LLM 1차 분류 및 키워드 기반 폴백을 사용합니다.

- **Graph-of-Thought 확장**: 복잡한 분석 쿼리를 위한 스냅샷 기반 백트래킹을 갖춘 다중 분기 추론:
  - **분기 탐색**: 각 단계에서 `GOT_BRANCH_FACTOR` 개의 후보 쿼리를 병렬로 전개
  - **품질 점수화**: 관찰자 LLM이 각 분기의 관련성, 범위, 참신성을 평가 (0.0–1.0)
  - **지능형 병합**: `GOT_THOUGHT_SCORE_THRESHOLD` 이상의 분기를 설정된 전략(`top_k`/`weighted_union`/`vote`)을 통해 병합
  - **엣지 가지치기**: 키워드 오버랩 점수(`GOT_EDGE_PRUNE_THRESHOLD`)를 통해 저품질 연결 제거
  - **실패 복구**: 연속적인 전체 분기 실패 시 마지막 성공적인 병합 지점으로 스냅샷 롤백

- **SGLang 추론 에코시스템**: 모든 LLM 작업(생성, 임베딩, 리랭킹, 홉 분류, 품질 평가)이 지능형 생명주기 관리를 갖춘 SGLang 서버에서 실행됩니다:
  - **지연 로딩(Lazy-loading) 아키텍처**: 첫 요청 시 서버가 자동 시작되어 초기화 중 콜드 스타트 오버헤드 제거
  - **유휴 타임아웃**: 비활성 상태가 `SGLANG_IDLE_TIMEOUT` (기본 60초) 지속되면 GPU 메모리 자동 해제
  - **GPU 할당**: 서버별 구성 가능한 장치 할당 및 메모리 비율 지정
  - **청크 재시도 로직**: LLM 메타데이터 추출 실패 시 서버 재시작과 함께 청크 자동 재시도 (`GRAPH_EXTRACTOR_RETRY_ON_FAILURE` 설정)

- **도구 호출 프레임워크 (Tool Calling)**: 로컬 폴백을 지원하는 MCP(Model Context Protocol) 서버 통합:
  - **MCP 우선 아키텍처**: 사용 가능 시 MCP 서버를 통한 1차 실행 (`MCP_SERVER_ENABLED=true`)
  - **로컬 폴백**: MCP 서버를 사용할 수 없는 경우 로컬 구현체로 자동 전환
  - **자연어 파싱**: 한국어/영어 수학 표현식("144의 제곱근", "루트 144")을 실행 가능한 코드로 변환

- **자동 그래프 구축**: 원본 문서에서 쿼리 가능한 지식 그래프까지의 End-to-end 파이프라인:
  - **OCR 처리**: SGLang 기반 OCRFlux가 문서를 마크다운으로 변환
  - **LLM 추출**: 구성 가능한 청크 크기 및 타임아웃으로 마크다운 청크에서 엔티티/이벤트/관계 추출
  - **이중 저장소**: 결정론적 UUID를 사용하여 Weaviate(상호 참조 그래프) 및 Neo4j(심층 그래프)에 동시 업서트

- **비동기 작업 모니터링**: REST API를 통한 전체 작업 생명주기 추적.

---

## 아키텍처

```
┌──────────────┐    ┌─────────────────────────────┐
│  입력 레이어  │ →  │  LangGraph 업로드 파이프라인   │ →  Markdown + *.graph.json
│ (PDF/IMG/…)  │    │   (MemorySaver 체크포인트)    │
└──────────────┘    └─────────────────────────────┘
                                  │
┌──────────────┐    ┌─────────────────────────────┐
│  사용자 쿼리  │ →  │  LangGraph RAG 워크플로우     │
└──────────────┘    │   (MemorySaver 체크포인트)    │
                    └─────────────────────────────┘
                                  │
                    ┌─────────────┴──────────────┐
                    │   홉 라우터 (LLM + 규칙)     │
                    └─────────────┬──────────────┘
                                  │
     ┌────────────────────────────┼────────────────────────────┐
     │                            │                            │
  Path 1                       Path 2                        Path 3
  Vector RAG                   Weaviate Cross-Ref            Neo4j GraphDB
  TextSearcher+Reranker         GraphQL QueryReference        Deep Traversal
  (≤ 2 홉)                      (3–5 홉)                      (≥ 6 홉)
                                  │
                                  ▼
                 ┌──────────────────────────────────────────┐
                 │  품질 게이트 + 관찰자 LLM (0~1 점수)        │
                 └────────────────┬─────────────────────────┘
                                  │
                        ┌─────────▼──────────────┐
                        │  GoT Thought Expander  │
                        │  분기 병합 + 가지치기     │
                        └─────────┬──────────────┘
                                  │
                           ┌──────▼──────┐
                           │  LLM 답변    │
                           └─────────────┘
```

### 워크플로우 그래프 (쿼리)

```
planner → tool_router ┬→ rag_router →┬→ vector_retriever  ──→┐
                      │               ├→ crossref_retriever ─→├→ quality_gate →┬→ thought_expander → aggregator → END
                      │               └→ graphdb_retriever ──→┘                └→ rag_router (backtrack)
                      │
                      └→ tool_executor ────────────────────────────────────────→ aggregator → END
```

**Tool Router**: LLM 기반 의도 분류가 쿼리 라우팅 전략을 결정합니다:
* **의도 카테고리**: `knowledge`, `calculation`, `database`, `api_call`, `code_exec`
* **라우팅 로직**:
  * `knowledge` → RAG 파이프라인 (3-Way 검색 라우팅)
  * 기타 의도 → Tool executor (MCP / 로컬 폴백)
* **폴백 메커니즘**: LLM 분류 실패 시 휴리스틱 키워드 매칭 수행

**Tool Executor**:
* **MCP 통합**: `MCP_SERVER_ENABLED=true`일 때 MCP 서버 REST API를 통한 1차 실행
* **로컬 폴백**: MCP를 사용할 수 없을 때 로컬 구현체로 자동 전환
* **지원 도구**: Calculator (AST 안전 평가, 자연어 파싱), API Caller, Code Runner (샌드박스), SQL Executor (예정)

---

## 입력 / 전처리

`LangGraphUploadPipeline` (`langgraph_upload_pipeline.py`)이 모든 노드에 걸쳐 `MemorySaver` 체크포인팅과 함께 처리합니다:

1. **변환 & 레이아웃**: `run_file_processor.py`가 PDF/Office/이미지/오디오 입력을 처리 → `Results/1.Converted_images` + `Results/2.LayoutDetection`.
2. **OCR & Markdown**: SGLang 기반 OCRFlux를 사용하는 `run_ocr_processing()`이 페이지별 Markdown 생성 → `Results/4.OCR_results`.
3. **LLM 메타데이터 추출**: `LLMMetadataExtractor`가 Markdown에서 엔티티/이벤트/관계 추출 → `Results/8.graph_metadata/*.graph.json`.
   - **청크 크기**: configurable via `GRAPH_EXTRACTOR_CHUNK_SIZE`
   - **타임아웃**: configurable via `GRAPH_EXTRACTOR_API_TIMEOUT`
   - **재시도 로직**: 타임아웃 발생 시 SGLang generator 서버 재시작 후 동일 청크 1회 재시도
   - **Keepalive**: 처리 중 20초마다 서버 touch하는 백그라운드 스레드
4. **그래프 업서트**:
   - `GraphSchemaManager`가 Weaviate GraphEntity/GraphEvent/GraphRelation 컬렉션을 Cross-Reference(source/target/event)와 함께 생성
   - `LegacyGraphIngestor` / `Neo4jManager`가 결정론적 UUID로 Neo4j에 노드/관계를 MERGE
5. **Late Chunking & 임베딩**: `embedding_text.py`가 Markdown을 청크로 분할하고 `SharedEmbeddingModel`을 통해 Weaviate TextDocument 컬렉션에 업로드합니다(`EMBEDDING_MODEL`로 모델 설정 가능).

---

## 모듈 맵

```
backend/
├── main.py                  # [제외] FastAPI 서버 진입점
├── config.py                # [제외] 서버 수준 설정
├── logging_config.py        # [제외] 로깅 설정
│
├── api/                     # API 레이어
│   ├── routes.py            # [제외] 주요 업로드/파일/세션 라우트
│   ├── chat.py              # [제외] POST /v1/chat 엔드포인트
│   ├── ocr_routes.py        # [제외] OCR 처리 엔드포인트
│   └── pause_api.py         # [제외] 작업 일시정지/재개 API
│
├── notebooklm/              # RAG 핵심 모듈
│   ├── config.py            # 모델/경로/그래프 설정
│   ├── rag_pipeline.py      # LangGraph RAG 워크플로우 오케스트레이터
│   ├── graph_reasoner.py    # LangGraph 워크플로우 오케스트레이션
│   ├── graph_schema.py      # Weaviate Entity/Event/Relation 스키마
│   ├── hop_classifier.py    # 쿼리 복잡도 추정기
│   ├── reasoner/            # 리팩터링된 GraphReasoner 모듈
│   │   ├── state.py         # GraphReasonerState 정의
│   │   ├── routing.py       # PathSelector, HopClassifier
│   │   ├── quality.py       # QualityEvaluator
│   │   ├── retrievers.py    # VectorRetriever, CrossRefRetriever, GraphDBRetriever
│   │   └── __init__.py
│   ├── legacy_graph_client.py # Neo4j Cypher 탐색 클라이언트
│   ├── legacy_graph_ingestor.py # Neo4j 업서트 헬퍼
│   ├── embedding_text.py    # Late chunking + Weaviate 텍스트 인덱싱
│   ├── embedding_image.py   # 이미지 임베딩 + Weaviate 이미지 인덱싱
│   ├── image_processor.py   # 이미지 처리 유틸리티
│   ├── shared_embedding.py  # SGLang 임베딩/리랭커 클라이언트 (싱글톤)
│   ├── sglang_server_manager.py # SGLang 서버 생명주기 관리자
│   ├── generator.py         # LLM 답변 생성
│   ├── refiner.py           # 답변 정제
│   ├── evaluator.py         # 답변 품질 평가
│   ├── router.py            # 쿼리 유형 라우팅
│   ├── query_rewriter.py    # 쿼리 재작성
│   ├── parallel_search.py   # 병렬 텍스트+이미지 검색
│   ├── weaviate_utils.py    # Weaviate 클라이언트 유틸리티
│   ├── clean_weaviate.py    # [제외] Weaviate + Neo4j 데이터 정리 스크립트
│   ├── tools/               # 툴 호출 & MCP 통합
│   │   ├── mcp_client.py    # MCP 서버 REST 클라이언트
│   │   ├── tool_executor.py # 툴 실행 (MCP/로컬 폴백)
│   │   └── __init__.py
│   ├── rag_text/            # 텍스트 검색 + 리랭커
│   └── rag_image/           # 이미지 검색 + 리랭커
│
├── mcp_server/              # MCP (Model Context Protocol) 툴 서버
│   └── main.py              # 툴 실행을 위한 FastAPI 서버
│
├── data_pipeline/           # 데이터 처리 파이프라인
│   └── pipe/
│       ├── langgraph_upload_pipeline.py # LangGraph 업로드 워크플로우 (체크포인팅)
│       ├── llm_metadata_extractor.py    # 엔티티/이벤트/관계 추출
│       ├── neo4j_manager.py             # Neo4j 업서트 매니저
│       ├── run_file_processor.py        # 변환/레이아웃/OCR 오케스트레이터
│       ├── pipeline_image.py            # 이미지 파이프라인
│       ├── pipeline_sound.py            # 오디오 파이프라인
│       └── main_pipe/
│           ├── ocr_pipe/                # SGLang 기반 OCRFlux 엔진
│           ├── udp_pdftopng_300dpi.py   # PDF → PNG 변환
│           └── udp_layoutdetection.py   # 레이아웃 감지
│
├── services/                
│   ├── model_manager.py     # [제외] LazyModelManager (GPU 생명주기)
│   ├── ocr_vision_manager.py# [제외] OCR 엔진 관리
│   └── rag_service.py       # [제외] RAG 서비스 오케스트레이션
│
└── utils/
    ├── task_queue.py        # [제외] GPU 작업 큐 (비동기 작업 관리)
    ├── helpers.py           # [제외] 공유 유틸리티 함수
    ├── path_helpers.py      # [제외] 경로 계산 헬퍼
    └── file_utils.py        # [제외] 파일 작업 유틸리티
```

---

## 데이터 파이프라인

1. **파일 업로드** (`POST /upload/files`)
   - `api/routes.py`가 파일을 세션별 폴더에 저장하고 `task_queue.py`를 통해 `run_processing_pipeline`을 큐에 추가합니다.
2. **GPU 작업 큐**
   - `task_queue.py`가 진행 상황 추적과 함께 순차적 GPU 바운드 작업(변환 → 레이아웃 → OCR)을 관리합니다.
3. **텍스트 인덱싱** (`run_text_indexing` / `run_text_indexing_v2`)
   - `SharedEmbeddingModel` 초기화 → Weaviate late-chunking 인덱싱을 위한 `process_markdown_files` 실행.
4. **그래프 추출 & 수집**
   - `LLMMetadataExtractor`가 `*.graph.json` 생성 → `GraphSchemaManager`가 Weaviate에 업서트 → `Neo4jManager` / `LegacyGraphIngestor`가 Neo4j에 업서트.
5. **저장소 상태**
   - Weaviate: TextDocument + GraphEntity/Event/Relation 컬렉션.
   - Neo4j: Entity/Event 노드 + 결정론적 UUID와 자동 생성 제약 조건을 가진 관계 엣지.

---

## 쿼리 처리 흐름 (Query Processing Flow)

`GraphReasoner` (`graph_reasoner.py`)가 `MemorySaver` 체크포인팅과 지능형 백트래킹으로 처리합니다. 실행은 다음과 같은 시간순 흐름을 따릅니다:

1. **요청 시작**: `POST /v1/chat` → `RAGPipeline.process_query()`가 `GraphReasoner.retrieve()` 호출.

2. **LangGraph 워크플로우 실행** (`graph_reasoner.py`):

   * **Step 1: Planner** (`planner` 노드)
     - 사용자 쿼리를 분석하고 핵심 개념을 추출합니다.
     - 상위 수준의 검색 계획과 추론 단계를 기록합니다.
     - 다중 턴 컨텍스트를 위한 쿼리 히스토리를 초기화합니다.
     - **출력**: `plan`, `query_analysis`

   * **Step 2: Tool Router** (`tool_router` 노드)
     - LLM이 `ToolExecutor.classify_intent()`를 통해 쿼리 의도를 분류합니다:
       - `knowledge` → **Branch A** (RAG Router)로 라우팅
       - `calculation`, `database`, `api_call`, `code_exec` → **Branch B** (Tool Executor)로 라우팅
     - **출력**: `intent`, 라우팅 결정 사항

   **[Branch A: 지식 쿼리 경로]**

   * **Step 3a: RAG Router** (`rag_router` 노드)
     - 홉 분류 수행 (LLM + 휴리스틱 하이브리드)
     - `max_hops` = `min(llm_estimate, heuristic_estimate, GRAPH_MAX_HOPS)`로 설정
     - 초기 검색 경로 선택:
       - 홉 ≤ 2 → `vector_retriever` (Path 1)
       - 홉 3–5 → `crossref_retriever` (Path 2)
       - 홉 ≥ 6 → `graphdb_retriever` (Path 3)
     - **출력**: `max_hops`, `retrieval_path`, `tried_paths`

   * **Step 4a: 검색 실행** (Path 1/2/3)
     - **Path 1**: TextDocument 상의 의미 검색 + 리랭커
     - **Path 2**: BM25 시드 검색 + Weaviate 상호 참조 다중 홉 탐색
     - **Path 3**: Neo4j Cypher 심층 그래프 탐색
     - **출력**: `context_snippets`, `entities`, `events`, `relations`

   * **Step 5a: 품질 게이트** (`quality_gate` 노드)
     - 관찰자 LLM이 검색 결과를 평가합니다 (0.0–1.0).
     - **품질 ≥ `QUALITY_GATE_THRESHOLD`**: 어그리게이터 또는 Thought Expander로 진행.
     - **품질 < 임계값**: 지능형 백트래킹 트리거 (`PathSelector`가 남은 경로를 분석하여 대안을 선택하고 Step 4a로 돌아감).
     - **종료 조건**: `MAX_BACKTRACK_COUNT` 재시도 초과 또는 모든 경로 소진 시.
     - **출력**: `retrieval_quality`, `backtrack_count`, `tried_paths`

   * **Step 6a: Thought Expander** (GoT 활성화 시)
     - `GOT_BRANCH_FACTOR` 개의 후보 쿼리를 병렬로 전개.
     - 관찰자 LLM이 각 분기를 점수화하고 임계값 이상의 분기를 병합.
     - 연속 실패 시 스냅샷 기반 백트래킹 수행.
     - **출력**: `thought_steps`, 확장된 컨텍스트

   **[Branch B: 도구 실행 경로]**

   * **Step 3b: Tool Executor** (`tool_executor` 노드)
     - 의도를 특정 도구(`calculation`, `api_call`, `code_exec`, `database`)에 매핑.
     - 도구 입력 준비 (예: 자연어 "144의 제곱근" → `sqrt(144)` 변환).
     - 도구 실행 (MCP 서버 우선 실행, 실패 시 로컬 폴백).
     - **출력**: 상태(`ok`/`error`), 결과값, 메타데이터가 포함된 `tool_result` (도구 실패 시 오류 메시지 반환; RAG로 돌아가지 않음).

   **[병합 노드 (Merge Node)]**

   * **Step 7: Aggregator** (`aggregator` 노드)
     - **RAG 쿼리**: 엔티티/이벤트/관계/사고 단계에서 컨텍스트 스니펫을 구성.
     - **Tool 쿼리**: 툴 결과를 자연어로 포맷팅 (예: `"수식 = 결과"`).
     - 디버깅 메타데이터 수집 (`context_snippets`, `backtrack_count`, `tool_result` 등).
     - **출력**: 통합된 컨텍스트 또는 포맷팅된 도구 결과

3. **답변 생성**:
   - **RAG 쿼리**: `generator.py`가 원본 쿼리와 컨텍스트 스니펫을 합성하여 답변 생성.
   - **Tool 쿼리**: 포맷팅된 도구 결과를 직접 반환 (LLM 생성 생략).
   - **선택적 후처리**: `refiner.py`가 답변을 정제하고, `evaluator.py`가 품질 노트를 기록.

4. **응답 구성**:
   - 디버깅을 지원하기 위해 전체 워크플로우 상태(`plan`, `max_hops`, `backtrack_count`, `tried_paths` 등)가 노출됩니다.

5. **클라이언트 반환**: 답변, 메타데이터, 디버깅 정보가 포함된 JSON 응답 반환.

---

## 주요 설정 (`notebooklm/config.py`)

- **Graph RAG 토글**: `GRAPH_RAG_ENABLED`, `LANGGRAPH_ENABLED`, `GOT_MODE_ENABLED`, `GRAPH_MAX_HOPS`.
- **GoT 튜닝**: `GOT_BRANCH_FACTOR`, `GOT_MERGE_STRATEGY` (`top_k`/`weighted_union`/`vote`), `GOT_MERGE_TOP_K`, `GOT_THOUGHT_SCORE_THRESHOLD`, `GOT_EDGE_PRUNE_THRESHOLD`, `GOT_MAX_STEPS`, `GOT_MAX_CONSECUTIVE_FAILURES`, `GOT_OBSERVER_ENDPOINT`/`GOT_OBSERVER_MODEL`.
- **Neo4j**: `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `GRAPH_MAX_HOPS`.
- **Weaviate**: `WEAVIATE_HOST/PORT`, `WEAVIATE_TEXT_CLASS` (TextDocument), `WEAVIATE_VECTORIZER` (text2vec-model2vec).
- **SGLang 모델**: `LLM_MODEL`, `EMBEDDING_MODEL`, `RERANKER_MODEL`, `REFINER_MODEL`, `QUERY_REWRITER_MODEL`.
- **SGLang 서버**: `SGLANG_GENERATOR_ENDPOINT`, `SGLANG_EMBEDDING_ENDPOINT`, `SGLANG_RERANKER_ENDPOINT`, `SGLANG_REFINER_ENDPOINT`, `SGLANG_QUERY_REWRITER_ENDPOINT`.
- **SGLang 생명주기**: `SGLANG_IDLE_TIMEOUT` (60초), `SGLANG_KEEPALIVE_INTERVAL` (20초).
- **그래프 추출기**: `GRAPH_EXTRACTOR_API_TIMEOUT` (60초), `GRAPH_EXTRACTOR_CHUNK_SIZE` (800), `GRAPH_EXTRACTOR_RETRY_ON_FAILURE` (true).
- **세션 디렉토리**: `DATA_ROOT/Results`, `sessions/<id>` 레이아웃.

---

## MCP Tool Server

MCP (Model Context Protocol) 서버는 LangGraph RAG 시스템의 툴 실행을 담당하는 독립적인 FastAPI 서비스입니다. 계산기, SQL 쿼리, API 호출, 코드 실행과 같은 계산 툴을 실행하기 위한 REST API를 제공합니다.

### 기능
- **Calculator**: AST 기반 안전한 수학 표현식 평가
- **SQL Executor**: SQL 쿼리 실행 (확장 가능)
- **API Caller**: 외부 API 호출 (확장 가능)
- **Code Runner**: 코드 실행 샌드박스 (확장 가능)

### 새 툴 추가하기

1. `main.py`에 executor 함수를 구현합니다.
2. `TOOL_REGISTRY`에 등록합니다.
3. 서버를 재시작합니다.

```
def execute_my_tool(inputs: Dict[str, Any]) -> ToolExecuteResponse:
    # 구현
    return ToolExecuteResponse(status="ok", result=...)

TOOL_REGISTRY["my_tool"] = {
    "executor": execute_my_tool,
    "info": ToolInfo(
        name="my_tool",
        description="도구 설명",
        parameters={...}
    )
}
```

---

## API 요약

| 엔드포인트 | 설명 |
|---|---|
| `POST /upload/files` | LangGraph 업로드 파이프라인 트리거 |
| `POST /v1/chat` | 체크포인트/백트래킹이 적용된 3-Way RAG 실행 |
| `GET /api/v1/tasks/{task_id}` | 큐에 등록된 업로드/OCR 작업 모니터링 |
| `GET /files` | 세션 아티팩트 목록 조회 |
| `POST /pause` | 백그라운드 작업 일시정지/재개 |

---

## 로깅 & 운영

- `sglang_embedding_server.log`, `sglang_reranker_server.log` – SGLang 모델 서버 상태.
- `Results/8.graph_metadata/*.graph.json` – LLM 추출 결과 아카이브.
- `LegacyGraphIngestor`가 첫 실행 시 제약 조건을 자동 생성합니다; 수동 설정 불필요.
- `SGLangServerManager`가 약 60초(설정 가능) 유휴 후 GPU 메모리를 자동 해제합니다.
- 모든 LangGraph 워크플로우가 추적성을 위해 체크포인트 ID와 백트래킹 횟수를 기록합니다.
- **SGLang 콜드 스타트**: 지연 로딩 구조라 첫 `/v1/chat`(또는 hop classifier) 요청 시 각 SGLang 서버가 웜업을 위해 20–60초의 VRAM 적재 시간이 소요될 수 있습니다. 웜업 요청이나 keep-alive 설정으로 타임아웃을 방지하세요.
- **청크 재시도 메커니즘**: LLM 메타데이터 추출 시 타임아웃이 발생하면, 생성기 서버가 자동으로 재시작되고 해당 청크를 한 번 더 재시도합니다.
- **휘발성 체크포인트**: `MemorySaver`는 프로세스 내부에 그래프 스냅샷을 저장하므로 FastAPI가 재시작되면 진행 중인 상태가 모두 소실됩니다. `SqliteSaver`/`PostgresSaver` 마이그레이션 적용을 계획 중입니다.

---

## 로드맵

1. ~~**GoT (Graph of Thought)**~~
   - `thought_expander`가 이제 그래프 형태의 탐색을 수행합니다: 각 단계에서 `GOT_BRANCH_FACTOR`개의 분기를 확장하고, 관찰자 LLM이 각 분기를 점수화하며, 최적 결과가 `GOT_MERGE_STRATEGY`로 병합됩니다. 저품질 엣지는 가지치기되고, 연속 실패 시 스냅샷 기반 백트래킹이 트리거됩니다.
2. **고급 홉 분류기**
   - 하이브리드 라우터를 위해 쿼리 메타데이터(토큰 길이, 엔티티 수)로 보강합니다.
3. **멀티 그래프 검색 최적화**
   - 3–5 홉 Weaviate 탐색의 컨텍스트 필터링/중복 제거를 개선하고 ≥ 6 홉 Neo4j 탐색을 위한 Cypher 템플릿을 추가합니다.
4. **LangGraph 워크플로우 관찰 가능성**
   - 노드별 지연/오류 메트릭을 내보내고 `GraphReasoner` 및 `LegacyGraphClient` 내에 재시도 정책을 통합합니다.
5. **영속적 체크포인터**
   - 크로스 세션 상태 복구를 위해 `MemorySaver`에서 `SqliteSaver` / `PostgresSaver`로 마이그레이션합니다.

---

## 기여 & 연락처

이슈와 PR을 환영합니다. 문의 사항은 koto144@gmail.com으로 연락해 주세요.

---

## 라이선스

이 프로젝트는 이중 라이선스로 제공됩니다:
- **MIT 라이선스** - 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.
- **Apache 라이선스 2.0** - 자세한 내용은 [LICENSE-APACHE](LICENSE-APACHE) 파일을 참조하세요.

이 소프트웨어 사용에 적용할 라이선스를 선택할 수 있습니다.

---

## 인용

이 프로젝트를 연구에 사용하는 경우 다음을 인용해 주세요:

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
