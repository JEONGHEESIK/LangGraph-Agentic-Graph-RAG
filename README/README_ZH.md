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

[English](../README.md) | [한국어](README_KO.md) | 中文
</div>

传统的单向量（Single-vector）RAG 系统在处理复杂的多跳（Multi-hop）查询时面临固有局限，往往只能停留在被动的信息检索阶段。为了克服这一问题，我最初设计了基于 LangGraph 和 SGLang 的 Agentic Graph RAG，通过自适应的 3-Way 路由机制来最大化推理准确性并减少不必要的数据库流量。

然而，在架构设计过程中我意识到，只要在这个坚实的检索基础之上，增加一个用于意图分类的**路由层（Tool Router）**并结合 MCP（模型上下文协议），就完全有可能构建出一个高度自主的 Agentic AI 系统。

基于这种对高可扩展性的考量，我扩展了整体架构。最终形成的框架能够动态对用户意图进行分类，在深度知识检索（Graph RAG）和计算任务（计算器、SQL、API 等）之间实现无缝切换。结合智能回溯和 Quality Gate 质量验证，本项目不仅打破了传统问答的限制，更为构建能够主动思考并执行工具的 AI 系统奠定了高度可扩展的坚实基础。

LangGraph-Agentic-Graph RAG 是一个基于 LangGraph + SGLang 驱动的完全自主的 Agentic AI 与向量-图混合 RAG 平台。数据摄取 (Ingestion) 流水线通过具有检查点持久化功能的 LangGraph 状态机，将原始文档转换为 Markdown 分块和图元数据。在查询阶段，基于意图的工具路由器会动态地将计算任务通过 MCP 委托给外部工具；而知识类查询则会通过带有质量网关回溯（Quality-gate backtracking）的基于跳数（Hop-based）的路由器，在三条检索路径（Vector、Weaviate GraphRAG 或 Neo4j GraphDB）中进行选择，从而生成精确的答案。

系统主要特性：

- **双模式查询处理**：基于 LLM 的意图分类，自动在基于知识的 RAG 检索和计算工具执行之间进行路由。
- **高级文档摄取**：通过具有检查点持久化功能的 LangGraph 状态机，将原始文档（PDF/图像/音频）转换为 Markdown 分块和结构化图元数据。
- **智能检索路由**：基于跳数（Hop-based）的路由器，结合质量门控回溯，动态在三条检索路径（Vector、Weaviate GraphRAG 或 Neo4j GraphDB）中选择最优路径。
- **工具调用框架**：集成 MCP 服务器，并为计算器、API 调用、代码执行和数据库查询提供本地备用方案（Fallback）。
- **Graph-of-Thought 推理**：支持基于快照回溯的多分支（Multi-branch）探索，专为复杂分析查询设计。

<div align="center">
<img src="https://github.com/user-attachments/assets/504ea0fa-ed9a-4664-9095-042e01debc65" width="512" height="768" ></img><br/>
</div>

---

## 核心功能 (Key Capabilities)

- **LangGraph 状态机**：所有工作流（数据摄取、查询推理、工具执行、摘要生成、思维导图生成）均运行在带有 `MemorySaver` 检查点的 LangGraph `StateGraph` 上，实现完整的状态持久化与恢复。

- **智能查询路由**：基于 LLM 的意图分类，自动确定查询需要知识检索还是计算工具执行：
  - **知识查询** → 带有 3 路检索路由的 RAG 管道
  - **计算查询** → 支持高级数学（sqrt, log, trig, sigma）的基于 AST 的安全评估计算器工具
  - **数据库查询** → SQL 执行器（计划中）
  - **API 调用** → 具有可配置端点的 HTTP API 调用器
  - **代码执行** → 具有受限内置函数的 Python 沙箱

- **检查点与智能回溯**：每个节点转换都会被记录检查点；质量门控会评估检索结果，并在质量不足时触发智能路径选择：
  - **质量评估**：观察者 LLM 对每个路径结果（0.0–1.0）进行评分，以对比 `QUALITY_GATE_THRESHOLD`
  - **智能路径选择**：`PathSelector` 根据查询关键词、跳数和路径特征分析剩余未尝试的路径，以选择最合适的替代方案
  - **回溯限制**：可通过 `MAX_BACKTRACK_COUNT` 配置，防止无限循环
  - **状态跟踪**：`tried_paths` 字段防止重试失败的策略

- **3 路检索路由**：查询复杂度（跳数）决定最佳检索策略：
  - **路径 1 – Vector RAG (≤ 2 跳)**：在延迟分块（Late-chunked）的 TextDocument 语料库上执行基于 。非常适合直接的事实性问题。
  - **路径 2 – Weaviate Cross-Reference GraphRAG (3–5 跳)**：BM25 种子实体搜索，随后在 Weaviate 内进行多跳交叉引用遍历（源/目标/事件引用）。通过关系遍历挖掘与查询相邻的实体和事件。
  - **路径 3 – Neo4j Deep Graph Traversal (≥ 6 跳)**：用于模式密集型关系推理的基于 Cypher 的深度图探索。处理需要广泛图遍历的复杂多实体查询。
  - **跳数分类**：混合 LLM + 启发式方法估计查询复杂度，以 LLM 主要分类和基于关键词的降级方案（Fallback）为辅助。

- **Graph-of-Thought 思维扩展**：具有基于快照回溯的多分支推理，用于复杂的分析查询：
  - **分支探索**：每步并行扩展出 `GOT_BRANCH_FACTOR` 个候选查询
  - **质量评分**：观察者 LLM 评估每个分支的相关性、覆盖度和新颖性（0.0–1.0）
  - **智能合并**：高于 `GOT_THOUGHT_SCORE_THRESHOLD` 的分支通过可配置的策略（`top_k`/`weighted_union`/`vote`）进行合并
  - **边缘剪枝**：通过关键词重叠评分（`GOT_EDGE_PRUNE_THRESHOLD`）移除低质量连接
  - **失败恢复**：连续的全分支失败会触发快照回滚到最后一次成功合并的点

- **SGLang 推理生态系统**：所有 LLM 操作（生成、嵌入、重排序、跳数分类、质量评估）均运行在具有智能生命周期管理的 SGLang 服务器上：
  - **延迟加载架构**：服务器在首次请求时自动启动，消除初始化期间的冷启动开销
  - **空闲超时**：在非活动状态持续 `SGLANG_IDLE_TIMEOUT`（默认 60 秒）后自动释放 GPU 内存
  - **GPU 分配**：可为每台服务器配置设备分配和内存占比
  - **分块重试逻辑**：LLM 元数据提取失败时自动重启服务器并重试（通过 `GRAPH_EXTRACTOR_RETRY_ON_FAILURE` 配置）

- **工具调用框架**：具有本地备用方案的 MCP（模型上下文协议）服务器集成：
  - **MCP 优先架构**：可用时优先通过 MCP 服务器执行（`MCP_SERVER_ENABLED=true`）
  - **本地备用**：MCP 服务器不可用时自动回退到本地实现
  - **自然语言解析**：将韩语/英语数学表达式（“144的平方根”）转换为可执行代码

- **自动图构建**：从原始文档到可查询知识图谱的端到端管道：
  - **OCR 处理**：基于 SGLang 的 OCRFlux 将文档转换为 Markdown
  - **LLM 提取**：从 Markdown 分块中提取实体/事件/关系，支持配置分块大小和超时
  - **双重存储**：使用确定性 UUID 同步更新插入到 Weaviate（交叉引用图）和 Neo4j（深度图）

- **异步任务监控**：通过 REST API 进行完整的任务生命周期跟踪。

---

## 架构

```
┌──────────────┐    ┌─────────────────────────────┐
│   输入层     │ →  │  LangGraph 上传管道          │ →  Markdown + *.graph.json
│ (PDF/IMG/…)  │    │  (MemorySaver 检查点)        │
└──────────────┘    └─────────────────────────────┘
                                  │
┌──────────────┐    ┌─────────────────────────────┐
│   用户查询   │ →  │  LangGraph RAG 工作流        │
└──────────────┘    │  (MemorySaver 检查点)        │
                    └─────────────────────────────┘
                                  │
                    ┌─────────────┴──────────────┐
                    │   跳数路由器（LLM + 规则）   │
                    └─────────────┬──────────────┘
                                  │
     ┌────────────────────────────┼────────────────────────────┐
     │                            │                            │
  路径 1                        路径 2                       路径 3
  Vector                       Weaviate Cross-Ref           Neo4j GraphDB
  TextSearcher+Reranker        QueryReference               深度遍历
  (≤ 2 跳)                      (3–5 跳)                      (≥ 6 跳)
                                  │
                                  ▼
                 ┌──────────────────────────────────────────┐
                 │  质量门控 + 观察者 LLM（0~1 评分）         │
                 └────────────────┬─────────────────────────┘
                                  │
                        ┌─────────▼──────────────┐
                        │  GoT 思维扩展器         │
                        │  分支合并 + 剪枝        │
                        └─────────┬──────────────┘
                                  │
                           ┌──────▼──────┐
                           │  LLM 答案   │
                           └─────────────┘
```

### 工作流图（查询）

```
planner → tool_router ┬→ rag_router →┬→ vector_retriever  ──→┐
                      │               ├→ crossref_retriever ─→├→ quality_gate →┬→ thought_expander → aggregator → END
                      │               └→ graphdb_retriever ──→┘                └→ rag_router (backtrack)
                      │
                      └→ tool_executor ────────────────────────────────────────→ aggregator → END
```

**Tool Router**：基于 LLM 的意图分类决定查询路由策略：
* **意图类别**：`knowledge`, `calculation`, `database`, `api_call`, `code_exec`
* **路由逻辑**：
  * `knowledge` → RAG 管道（3 路检索路由）
  * 其他意图 → 工具执行器（MCP / 本地备用）
* **备用机制（Fallback）**：当 LLM 分类失败时执行启发式关键词匹配

**Tool Executor**：
* **MCP 集成**：当 `MCP_SERVER_ENABLED=true` 时，优先通过 MCP 服务器 REST API 执行
* **本地备用**：当 MCP 不可用时自动回退到本地实现
* **支持工具**：Calculator（AST 安全评估、自然语言解析），API Caller，Code Runner（沙箱），SQL Executor（计划中）

---

## 输入 / 预处理层

由 `LangGraphUploadPipeline`（`langgraph_upload_pipeline.py`）在所有节点上使用 `MemorySaver` 检查点进行处理：

1. **转换与布局**：`run_file_processor.py` 处理 PDF/Office/图像/音频输入 → `Results/1.Converted_images` + `Results/2.LayoutDetection`。
2. **OCR 与 Markdown**：由 SGLang 驱动的 OCRFlux 执行 `run_ocr_processing()`，生成逐页的 Markdown → `Results/4.OCR_results`。
3. **LLM 元数据提取**：`LLMMetadataExtractor` 从 Markdown 中提取实体/事件/关系 → `Results/8.graph_metadata/*.graph.json`。
   - **分块大小**：可通过 `GRAPH_EXTRACTOR_CHUNK_SIZE` 配置
   - **超时**：可通过 `GRAPH_EXTRACTOR_API_TIMEOUT` 配置
   - **重试逻辑**：超时时，SGLang 生成器服务器将重启，并对同一分块重试一次
   - **保活机制**：后台线程在处理期间每隔指定时间“触碰”服务器以保持活跃
4. **图更新插入**：
   - `GraphSchemaManager` 确保包含交叉引用（source/target/event）的 Weaviate GraphEntity/GraphEvent/GraphRelation 集合存在
   - `LegacyGraphIngestor` / `Neo4jManager` 使用确定性 UUID 将节点/关系 MERGE（合并）到 Neo4j
5. **延迟分块与嵌入**：`embedding_text.py` 将 Markdown 拆分为分块，并通过 `SharedEmbeddingModel` 上传到 Weaviate TextDocument 集合。

---

## 模块地图

```
backend/
├── main.py                          # [Excluded] FastAPI 服务器入口点
├── config.py                        # [Excluded] 服务器级配置
├── logging_config.py                # [Excluded] 日志配置
│
├── api/                             # API 层
│   ├── routes.py                    # [Excluded] 主要上传/文件/会话路由
│   ├── chat.py                      # [Excluded] POST /v1/chat 端点
│   ├── ocr_routes.py                # [Excluded] OCR 处理端点
│   └── pause_api.py                 # [Excluded] 任务暂停/恢复 API
│
├── notebooklm/                      # RAG 核心模块
│   ├── config.py                    # 模型/路径/图配置
│   ├── rag_pipeline.py              # LangGraph RAG 工作流编排器
│   ├── graph_reasoner.py            # LangGraph 工作流编排
│   ├── graph_schema.py              # Weaviate Entity/Event/Relation 模式
│   ├── hop_classifier.py            # 查询复杂度估计器
│   ├── reasoner/                    # 重构的 GraphReasoner 模块
│   │   ├── state.py                 # GraphReasonerState 定义
│   │   ├── routing.py               # PathSelector, HopClassifier
│   │   ├── quality.py               # QualityEvaluator
│   │   ├── retrievers.py            # VectorRetriever, CrossRefRetriever, GraphDBRetriever
│   │   └── __init__.py
│   ├── legacy_graph_client.py       # Neo4j Cypher 遍历客户端
│   ├── legacy_graph_ingestor.py     # Neo4j 更新插入助手
│   ├── embedding_text.py            # 延迟分块 + Weaviate 文本索引
│   ├── embedding_image.py           # 图像嵌入 + Weaviate 图像索引
│   ├── image_processor.py           # 图像处理工具
│   ├── shared_embedding.py          # SGLang 嵌入/重排序客户端（单例）
│   ├── sglang_server_manager.py     # SGLang 服务器生命周期管理器
│   ├── generator.py                 # LLM 答案生成
│   ├── refiner.py                   # 答案精炼
│   ├── evaluator.py                 # 答案质量评估
│   ├── router.py                    # 查询类型路由
│   ├── query_rewriter.py            # 查询重写
│   ├── parallel_search.py           # 并行文本+图像搜索
│   ├── weaviate_utils.py            # Weaviate 客户端工具
│   ├── clean_weaviate.py            # [Excluded] Weaviate + Neo4j data cleanup script
│   ├── tools/                       # 工具调用 & MCP 集成
│   │   ├── mcp_client.py            # MCP 服务器 REST 客户端
│   │   ├── tool_executor.py         # 工具执行（MCP/本地 fallback）
│   │   └── __init__.py
│   ├── rag_text/                    # 文本搜索 + 重排序器
│   └── rag_image/                   # 图像搜索 + 重排序器
│
├── mcp_server/                      # MCP（模型上下文协议）工具服务器
│   ├── main.py                      # 用于工具执行的 FastAPI 服务器
│   └── requirements.txt             # MCP 服务器依赖项
│
├── data_pipeline/                    # 数据处理管道
│   └── pipe/
│       ├── langgraph_upload_pipeline.py  # LangGraph 上传工作流（带检查点）
│       ├── llm_metadata_extractor.py     # 实体/事件/关系提取
│       ├── neo4j_manager.py              # Neo4j 更新插入管理器
│       ├── run_file_processor.py         # 转换/布局/OCR 编排器
│       ├── pipeline_image.py             # 图像管道
│       ├── pipeline_sound.py             # 音频管道
│       └── main_pipe/
│           ├── ocr_pipe/                 # 基于 SGLang 的 OCRFlux 引擎
│           ├── udp_pdftopng_300dpi.py    # PDF → PNG 转换
│           └── udp_layoutdetection.py    # 布局检测
│
├── services/                        # 业务逻辑服务
│   ├── model_manager.py             # [Excluded] LazyModelManager（GPU 生命周期）
│   ├── ocr_vision_manager.py        # [Excluded] OCR 引擎管理
│   └── rag_service.py               # [Excluded] RAG 服务编排
│
└── utils/
    ├── task_queue.py                # [Excluded] GPU 任务队列（异步任务管理）
    ├── helpers.py                   # [Excluded] 共享工具函数
    ├── path_helpers.py              # [Excluded] 路径计算助手
    └── file_utils.py                # [Excluded] 文件操作工具
```

---

## 数据管道

1. **文件上传**（`POST /upload/files`）
   - `api/routes.py` 将文件存储在会话专属文件夹中，并通过 `task_queue.py` 将 `run_processing_pipeline` 加入队列。
2. **GPU 任务队列**
   - `task_queue.py` 管理顺序 GPU 绑定任务（转换 → 布局 → OCR），并跟踪进度。
3. **文本索引**（`run_text_indexing` / `run_text_indexing_v2`）
   - 初始化 `SharedEmbeddingModel` → 运行 `process_markdown_files` 进行 Weaviate 延迟分块（late-chunking）索引。
4. **图提取与摄取**
   - `LLMMetadataExtractor` 生成 `*.graph.json` → `GraphSchemaManager` 更新插入到 Weaviate → `Neo4jManager` / `LegacyGraphIngestor` 更新插入到 Neo4j。
5. **存储状态**
   - Weaviate：TextDocument + GraphEntity/Event/Relation 集合。
   - Neo4j：Entity/Event 节点 + 带有确定性 UUID 和自动创建约束的关系边。

---

## 查询处理流程 (Query Processing Flow)

由 `GraphReasoner` (`graph_reasoner.py`) 结合 `MemorySaver` 检查点和智能回溯进行处理。执行流程按以下时间顺序进行：

1. **请求初始化**：`POST /v1/chat` → `RAGPipeline.process_query()` 调用 `GraphReasoner.retrieve()`。

2. **LangGraph 工作流执行** (`graph_reasoner.py`)：

   * **Step 1: Planner（规划器节点）**
     - 分析用户查询并提取核心概念。
     - 记录顶层搜索计划和推理步骤。
     - 初始化查询历史以提供多轮对话上下文。
     - **输出**：`plan`, `query_analysis`

   * **Step 2: Tool Router（工具路由节点）**
     - LLM 通过 `ToolExecutor.classify_intent()` 分类查询意图：
       - `knowledge` → 路由到 **Branch A**（RAG Router）
       - `calculation`, `database`, `api_call`, `code_exec` → 路由到 **Branch B**（Tool Executor）
     - **输出**：`intent`，路由决定

   **[Branch A: 知识查询路径]**

   * **Step 3a: RAG Router（RAG 路由节点）**
     - 执行跳数分类（LLM + 启发式混合）
     - 设置 `max_hops` = `min(llm_estimate, heuristic_estimate, GRAPH_MAX_HOPS)`
     - 选择初始检索路径：
       - 跳数 ≤ 2 → `vector_retriever` (路径 1)
       - 跳数 3–5 → `crossref_retriever` (路径 2)
       - 跳数 ≥ 6 → `graphdb_retriever` (路径 3)
     - **输出**：`max_hops`, `retrieval_path`, `tried_paths`

   * **Step 4a: 检索执行** (路径 1/2/3)
     - **路径 1**：TextDocument 上的 语义搜索 + 重排序器
     - **路径 2**：BM25 种子搜索 + Weaviate 交叉引用多跳遍历
     - **路径 3**：Neo4j Cypher 深度图遍历
     - **输出**：`context_snippets`, `entities`, `events`, `relations`

   * **Step 5a: 质量门控 (Quality Gate)**
     - 观察者 LLM 对检索结果进行评分（0.0–1.0）。
     - **如果质量 ≥ `QUALITY_GATE_THRESHOLD`**：继续进入聚合器（Aggregator）或思维扩展器（Thought Expander）。
     - **如果质量 < 阈值**：触发智能回溯（`PathSelector` 分析剩余路径，选择替代方案，并返回 Step 4a）。
     - **终止条件**：超过 `MAX_BACKTRACK_COUNT` 重试次数或所有路径均已耗尽。
     - **输出**：`retrieval_quality`, `backtrack_count`, `tried_paths`

   * **Step 6a: 思维扩展器 (Thought Expander)**（如果启用 GoT）
     - 并行扇出（Fan-out）`GOT_BRANCH_FACTOR` 个候选查询。
     - 观察者 LLM 对每个分支进行评分，并合并高质量分支、修剪低质量边缘。
     - 连续失败时触发基于快照的回溯。
     - **输出**：`thought_steps`, 扩展的上下文

   **[Branch B: 工具执行路径]**

   * **Step 3b: 工具执行器 (Tool Executor)**
     - 将意图映射到特定工具（`calculation`, `api_call`, `code_exec`, `database`）。
     - 准备工具输入（例如，将自然语言“144的平方根”转换为 `sqrt(144)`）。
     - 执行工具（优先使用 MCP 架构，带有本地备用方案）。
     - **输出**：包含状态（`ok`/`error`）、结果值和元数据的 `tool_result`（注：工具失败将返回错误信息，**不会**回退到 RAG）。

   **[合并节点 (Merge Node)]**

   * **Step 7: 聚合器 (Aggregator)**
     - **对于 RAG 查询**：从实体/事件/关系/思维步骤构建上下文片段。
     - **对于工具查询**：将工具结果格式化为自然语言（例如，`"表达式 = 结果"`）。
     - 收集元数据（`context_snippets`, `backtrack_count`, `tool_result` 等）。
     - **输出**：聚合的上下文或格式化的工具结果。

3. **生成答案**：
   - **对于 RAG 查询**：`generator.py` 综合原始查询和上下文片段生成答案。
   - **对于工具查询**：直接返回格式化后的工具结果（跳过 LLM 生成）。
   - **可选后处理**：`refiner.py` 润色答案，`evaluator.py` 记录质量说明。

4. **构建响应**：
   - 为支持调试，将暴露整个工作流状态（`plan`, `max_hops`, `backtrack_count`, `tried_paths` 等）。

5. **返回客户端**：返回包含答案、元数据和调试信息的 JSON 响应。

---

## 关键配置 (`notebooklm/config.py`)

- **Graph RAG 开关**：`GRAPH_RAG_ENABLED`, `LANGGRAPH_ENABLED`, `GOT_MODE_ENABLED`, `GRAPH_MAX_HOPS`。
- **GoT 调优**：`GOT_BRANCH_FACTOR`, `GOT_MERGE_STRATEGY`（`top_k`/`weighted_union`/`vote`）, `GOT_MERGE_TOP_K`, `GOT_THOUGHT_SCORE_THRESHOLD`, `GOT_EDGE_PRUNE_THRESHOLD`, `GOT_MAX_STEPS`, `GOT_MAX_CONSECUTIVE_FAILURES`, `GOT_OBSERVER_ENDPOINT`/`GOT_OBSERVER_MODEL`。
- **Neo4j**：`NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `GRAPH_MAX_HOPS`。
- **Weaviate**：`WEAVIATE_HOST/PORT`, `WEAVIATE_TEXT_CLASS` (TextDocument), `WEAVIATE_VECTORIZER` (text2vec-model2vec)。
- **SGLang 模型**：`LLM_MODEL`, `EMBEDDING_MODEL`, `RERANKER_MODEL`, `REFINER_MODEL`, `QUERY_REWRITER_MODEL`。
- **SGLang 服务器**：`SGLANG_GENERATOR_ENDPOINT`, `SGLANG_EMBEDDING_ENDPOINT`, `SGLANG_RERANKER_ENDPOINT`, `SGLANG_REFINER_ENDPOINT`, `SGLANG_QUERY_REWRITER_ENDPOINT`。
- **SGLang 生命周期**：`SGLANG_IDLE_TIMEOUT` (60秒), `SGLANG_KEEPALIVE_INTERVAL` (20秒)。
- **图提取器**：`GRAPH_EXTRACTOR_API_TIMEOUT` (60秒), `GRAPH_EXTRACTOR_CHUNK_SIZE` (800), `GRAPH_EXTRACTOR_RETRY_ON_FAILURE` (true)。
- **会话目录**：`DATA_ROOT/Results`, `sessions/<id>` 布局。

---

## MCP 工具服务器 (MCP Tool Server)

MCP（模型上下文协议）服务器是一个独立的 FastAPI 服务，负责为 LangGraph RAG 系统执行工具。它提供 REST API 来执行计算工具，如计算器、SQL 查询、API 调用和代码执行。

### 功能
- **Calculator**：基于 AST 的安全数学表达式求值
- **SQL Executor**：SQL 查询执行（可扩展）
- **API Caller**：外部 API 调用（可扩展）
- **Code Runner**：代码执行沙箱（可扩展）

### 添加新工具

1. 在 `main.py` 中实现 executor 函数。
2. 在 `TOOL_REGISTRY` 中进行注册。
3. 重启服务器。

```
def execute_my_tool(inputs: Dict[str, Any]) -> ToolExecuteResponse:
    # 实现
    return ToolExecuteResponse(status="ok", result=...)

TOOL_REGISTRY["my_tool"] = {
    "executor": execute_my_tool,
    "info": ToolInfo(
        name="my_tool",
        description="工具描述",
        parameters={...}
    )
}
```

---

## API 概览

| 端点 | 描述 |
|---|---|
| `POST /upload/files` | 触发 LangGraph 上传管道 |
| `POST /v1/chat` | 运行带检查点/回溯的 3 路 RAG |
| `GET /api/v1/tasks/{task_id}` | 监控排队的上传/OCR 任务 |
| `GET /files` | 列出会话产物 |
| `POST /pause` | 暂停/恢复后台任务 |

---

## 日志与运维

- `sglang_embedding_server.log`、`sglang_reranker_server.log` – SGLang 模型服务器健康状态。
- `Results/8.graph_metadata/*.graph.json` – LLM 提取结果存档。
- `LegacyGraphIngestor` 在首次运行时自动创建约束；无需手动设置。
- `SGLangServerManager` 在空闲约 60 秒（可配置）后自动释放 GPU 内存。
- 所有 LangGraph 工作流记录检查点 ID 和回溯次数以供追踪。
- **SGLang 冷启动**：延迟加载架构意味着首次 `/v1/chat`（或跳数分类器）请求需为每个 SGLang 服务器预热，将模型加载到 VRAM 大约需要 20–60 秒。建议发送预热请求或配置 keep-alive 以避免客户端超时。
- **分块重试机制**：如果 LLM 元数据提取发生超时，生成器服务器将自动重启，并重新尝试该分块一次。
- **易失性检查点**：`MemorySaver` 将图快照存储在进程内，只要 FastAPI 重启，进行中的状态就会丢失。我们计划后续迁移至 `SqliteSaver`/`PostgresSaver`。

---

## 路线图

1. ~~**GoT（Graph of Thought）**~~
   - `thought_expander` 现在执行图形化探索：每步扩展 `GOT_BRANCH_FACTOR` 个分支，观察者 LLM 对每个分支评分，最佳结果通过 `GOT_MERGE_STRATEGY` 合并。低质量边被剪枝，连续失败触发基于快照的回溯。
2. **高级跳数分类器**
   - 使用查询元数据（词元长度、实体数量）增强混合路由器。
3. **多图检索优化**
   - 改进 3–5 跳 Weaviate 遍历的上下文过滤/去重，并为 ≥ 6 跳 Neo4j 探索添加 Cypher 模板。
4. **LangGraph 工作流可观测性**
   - 发出每节点延迟/错误指标，并在 `GraphReasoner` 和 `LegacyGraphClient` 内集成重试策略。
5. **持久化检查点**
   - 从 `MemorySaver` 迁移到 `SqliteSaver` / `PostgresSaver` 以实现跨会话状态恢复。

---

## 贡献与联系

欢迎提交 Issue 和 PR。如有问题，请联系 koto144@gmail.com。

---

## 许可证

本项目采用双重许可：
- **MIT 许可证** - 详情请参阅 [LICENSE](LICENSE) 文件
- **Apache 许可证 2.0** - 详情请参阅 [LICENSE-APACHE](LICENSE-APACHE) 文件

您可以选择其中任一许可证来管理您对本软件的使用。

---

## 引用

如果您在研究中使用了本项目，请引用以下内容：

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
