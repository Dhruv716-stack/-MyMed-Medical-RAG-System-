<div align="center">

# 🩺 MyMed AI

### Enterprise-Grade Medical Retrieval-Augmented Generation Platform

### *Building Reliable, Explainable, and Context-Aware Medical Intelligence through Multi-Stage Retrieval and Reflection.*

<br>

<p align="center">

<img src="https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>

<img src="https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi"/>

<img src="https://img.shields.io/badge/LangChain-RAG_Framework-1C3C3C?style=for-the-badge"/>

<img src="https://img.shields.io/badge/Qdrant-Vector_Database-DC244C?style=for-the-badge"/>

<img src="https://img.shields.io/badge/React-Frontend-61DAFB?style=for-the-badge&logo=react"/>

<img src="https://img.shields.io/badge/Docker-Deployment-2496ED?style=for-the-badge&logo=docker"/>

<img src="https://img.shields.io/badge/License-MIT-success?style=for-the-badge"/>

</p>

---

### 🧠 Production-Oriented Medical AI System

*A modular Retrieval-Augmented Generation platform that combines Hybrid Retrieval, Self-RAG, Corrective RAG, Reflection Pipelines, Confidence Estimation, and Medical Query Routing to generate grounded and explainable medical responses.*

---

**Hybrid Retrieval • Self-RAG • Corrective RAG • Reflection • Cross Encoder Reranking • Confidence Engine • LangSmith • FastAPI • React**

<br>

[Features](#-core-features) •
[Architecture](#-system-architecture) •
[Installation](#-installation) •
[API](#-api-reference) •
[Roadmap](#-roadmap)

</div>

---

# 📖 Table of Contents

- Introduction
- Problem Statement
- Why Traditional Medical RAG Fails
- Solution Overview
- Core Features
- System Architecture
- Complete Query Lifecycle
- Self-RAG Workflow
- Corrective RAG Workflow
- Tech Stack
- Repository Structure
- Installation
- Environment Variables
- API Reference
- Benchmarks
- Roadmap
- Contributors
- License

---

# 🎯 Problem Statement

Large Language Models have demonstrated remarkable reasoning capabilities, yet they remain fundamentally unsuitable for answering medical questions without access to reliable external knowledge.

Traditional language models suffer from several critical limitations:

- Hallucination of medical facts
- Outdated medical knowledge
- Lack of citation and explainability
- Inability to verify retrieved evidence
- Poor retrieval quality
- Context redundancy
- Overconfidence despite insufficient evidence

Conventional Retrieval-Augmented Generation (RAG) partially solves these issues by retrieving documents before generation. However, classical RAG architectures still assume that the retrieved context is always sufficient, relevant, and complete.

This assumption frequently breaks in real-world medical applications where incorrect retrieval directly affects answer quality.

---

# 💡 Solution Overview

**MyMed AI** introduces a production-inspired multi-stage retrieval pipeline that continuously evaluates the quality of retrieved information before generating an answer.

Instead of a single retrieval step, the system incorporates:

- Hybrid semantic and lexical retrieval
- Query rewriting
- Medical domain routing
- Maximum Marginal Relevance (MMR)
- Cross-Encoder reranking
- Retrieval reflection
- Corrective retrieval
- Context compression
- Answer reflection
- Confidence estimation

Every stage is designed to reduce hallucinations while improving retrieval precision and transparency.

---

# ✨ Core Features

## 🔍 Intelligent Retrieval

- Dense Vector Search
- BM25 Lexical Search
- Hybrid Retrieval Fusion
- Reciprocal Rank Fusion
- Maximum Marginal Relevance
- Metadata-aware retrieval

---

## 🧠 Advanced Reasoning

- Self-RAG Retrieval Reflection
- Corrective RAG
- Answer Reflection
- Confidence-aware Generation
- Context Sufficiency Detection
- Medical Query Classification

---

## ⚙️ Production Engineering

- Modular Architecture
- FastAPI Backend
- React Frontend
- JWT Authentication
- Docker Support
- Provider Abstraction
- LangSmith Tracing
- Structured Logging
- Config-driven Components

---

## 🤖 LLM Support

Supports multiple providers through a unified abstraction layer.

- Groq
- HuggingFace
- Ollama
- OpenAI
- Future Provider Extensions

Switching providers requires only configuration changes without modifying business logic.

---

# 🚀 Why MyMed AI?

Unlike traditional medical chatbots, MyMed treats Retrieval-Augmented Generation as a **decision pipeline** rather than a simple retrieve-and-generate workflow.

Instead of asking:

> "What documents are similar?"

MyMed continuously asks:

- Is the query well formed?
- Does this belong to a medical domain?
- Did retrieval return enough evidence?
- Is the evidence diverse?
- Can retrieval be improved?
- Is the generated answer supported?
- How confident should the system be?

This reflective reasoning pipeline substantially improves reliability compared to classical RAG implementations.

---

# 🏗 System Architecture

```text
                                        Medical Knowledge Base
                                                │
                     ┌──────────────────────────┼─────────────────────────┐
                     │                          │                         │
                     ▼                          ▼                         ▼
             PDF Documents              Clinical Notes            Medical Articles
                     │                          │                         │
                     └──────────────────────────┼─────────────────────────┘
                                                │
                                                ▼
                                      Ingestion Pipeline
                                                │
                            Cleaning • Parsing • Chunking • Metadata
                                                │
                                                ▼
                                  Embedding Generation (BGE Base)
                                                │
                                                ▼
                                   Qdrant Vector Database
                                                ▲
────────────────────────────────────────────────┼────────────────────────────────────

                                            User Query
                                                │
                                                ▼
                                       FastAPI REST API
                                                │
                                                ▼
                                      JWT Authentication
                                                │
                                                ▼
                                       Query Rewriter
                                                │
                                                ▼
                                     Medical Query Router
                                                │
                 ┌──────────────────────────────┴─────────────────────────────┐
                 │                                                            │
                 ▼                                                            ▼
          Dense Semantic Search                                       BM25 Retrieval
                 │                                                            │
                 └─────────────── Reciprocal Rank Fusion (RRF) ───────────────┘
                                                │
                                                ▼
                                   Maximum Marginal Relevance
                                                │
                                                ▼
                                   Cross Encoder Reranker
                                                │
                                                ▼
                                   Retrieval Reflection Agent
                                                │
                     ┌──────────────────────────┴─────────────────────────┐
                     │                                                    │
                     ▼                                                    ▼
             Context Sufficient                              Corrective Retrieval
                     │                                                    │
                     └──────────────────────────┬─────────────────────────┘
                                                ▼
                                      Context Compression
                                                │
                                                ▼
                                          LLM Generation
                                                │
                                                ▼
                                       Answer Reflection
                                                │
                                                ▼
                                        Confidence Engine
                                                │
                                                ▼
                                          Final Response
```

---

# 🏛 Design Principles

The architecture is built around five engineering principles.

## 1. Reliability

Every generated answer must be grounded in retrieved medical evidence rather than relying solely on LLM memory.

---

## 2. Explainability

The pipeline exposes retrieval confidence, reflection results, reranking decisions, and overall confidence rather than behaving as a black box.

---

## 3. Modularity

Every stage—including retrieval, reranking, reflection, and generation—is independently replaceable without affecting the remaining pipeline.

---

## 4. Extensibility

New embedding models, rerankers, retrieval strategies, or LLM providers can be integrated with minimal code changes.

---

## 5. Production Readiness

Configuration, logging, authentication, observability, and provider abstraction are designed following production software engineering practices.

---

# 🔄 End-to-End Query Lifecycle

Every medical question passes through multiple validation stages before an answer is produced.

```text
Receive Query
      │
      ▼
Rewrite Query
      │
      ▼
Medical Routing
      │
      ▼
Hybrid Retrieval
      │
      ▼
MMR Diversification
      │
      ▼
Cross Encoder Reranking
      │
      ▼
Reflection
      │
      ▼
Corrective Retrieval (if needed)
      │
      ▼
Compression
      │
      ▼
LLM Generation
      │
      ▼
Answer Reflection
      │
      ▼
Confidence Estimation
      │
      ▼
Final Medical Response
```

---
---

# 🧠 Core AI Pipeline

Unlike conventional Retrieval-Augmented Generation systems that retrieve documents once before prompting an LLM, **MyMed AI** treats retrieval as an adaptive reasoning process.

Instead of assuming retrieved context is always sufficient, the system continuously evaluates retrieval quality, corrects failures, and reflects on generated responses before producing the final answer.

The pipeline consists of four major stages:

1. Query Understanding
2. Intelligent Retrieval
3. Reflection & Corrective Retrieval
4. Confidence-Aware Generation

Each stage contributes independently to improving answer quality while reducing hallucinations.

---

# 🔍 Query Understanding Layer

Medical questions vary significantly in complexity.

Examples:

```
"What causes migraine?"
```

```
"What antibiotics are used for bacterial meningitis?"
```

```
"My blood pressure is 160/100. Is this dangerous?"
```

Each query requires different retrieval behavior.

Before retrieval begins, MyMed performs preprocessing to normalize and enrich the user query.

---

## Query Rewriter

The Query Rewriter improves retrieval quality by transforming ambiguous, incomplete, or conversational questions into retrieval-optimized search queries.

### Responsibilities

- Correct grammatical errors
- Expand abbreviations
- Normalize terminology
- Remove conversational noise
- Preserve medical intent
- Generate retrieval-friendly queries

### Example

Input

```
head pain after sleeping
```

↓

Rewritten

```
Possible causes of headache after sleeping
```

Another example

Input

```
what medicine for sugar
```

↓

Rewritten

```
Recommended medications for diabetes mellitus treatment
```

Benefits

- Better embedding similarity
- Improved BM25 matching
- Higher recall
- Reduced irrelevant retrieval

---

## Medical Query Router

Not every question belongs to the same medical category.

Different question types require different retrieval behavior.

The Medical Router classifies incoming queries into specialized domains.

Supported routing categories include

- Symptoms
- Diseases
- Drugs
- Diagnostics
- Treatment
- Lifestyle
- Emergency Care
- Nutrition
- General Medical Knowledge

Example

```
Symptoms of dengue
```

↓

Disease Knowledge

```
Paracetamol dosage
```

↓

Drug Information

```
Chest pain after running
```

↓

Emergency / Cardiology

Routing enables domain-specific prompting and retrieval strategies.

---

# 📚 Hybrid Retrieval Engine

Traditional RAG usually depends on a single retrieval strategy.

MyMed combines semantic and lexical retrieval to maximize both recall and precision.

## Dense Retrieval

Dense retrieval searches semantic meaning using vector embeddings.

Embedding Model

```
BAAI/bge-base-en-v1.5
```

Advantages

- Understands meaning
- Handles paraphrases
- Captures semantic similarity
- Robust against wording changes

Example

```
heart attack
```

matches

```
myocardial infarction
```

even without keyword overlap.

---

## BM25 Retrieval

BM25 performs lexical matching.

Advantages

- Excellent keyword precision
- Handles exact medical terminology
- Finds rare drug names
- Strong performance on abbreviations

Example

```
Metformin
```

BM25 immediately retrieves all documents containing the exact drug name.

---

## Why Hybrid Retrieval?

Neither dense retrieval nor BM25 is sufficient alone.

| Dense Retrieval | BM25 |
|----------------|------|
| Semantic similarity | Exact keyword matching |
| Handles paraphrases | Handles exact terms |
| Better recall | Better precision |
| Misses rare tokens | Misses semantic relationships |

Combining both provides significantly stronger retrieval performance.

---

# 🔀 Reciprocal Rank Fusion (RRF)

After Dense Retrieval and BM25 complete independently, results are merged using Reciprocal Rank Fusion.

Instead of selecting one retrieval strategy over another, RRF combines rankings to produce a stronger candidate set.

Benefits

- Better recall
- Reduced retrieval bias
- Higher document diversity
- More stable rankings

---

# 🌐 Maximum Marginal Relevance (MMR)

Even after hybrid retrieval, many retrieved chunks contain duplicate information.

MMR removes redundancy while maximizing information diversity.

Without MMR

```
Chunk 1
COVID symptoms

Chunk 2
COVID symptoms

Chunk 3
COVID symptoms

Chunk 4
COVID symptoms
```

With MMR

```
Chunk 1
Symptoms

Chunk 2
Diagnosis

Chunk 3
Treatment

Chunk 4
Complications
```

Advantages

- Diverse context
- Less repetition
- Better token efficiency
- Higher answer coverage

---

# 🎯 Cross Encoder Reranking

After MMR, candidate documents are reranked using a Cross Encoder.

Model

```
BAAI/bge-reranker-base
```

Unlike vector similarity, Cross Encoders jointly encode

(Query, Document)

before assigning a relevance score.

Pipeline

```
Retrieved Documents

↓

Cross Encoder

↓

Relevance Score

↓

Sorted Context
```

Benefits

- Higher precision
- Better ranking
- Stronger grounding
- Reduced hallucination

---

# 🪞 Retrieval Reflection

This is one of the defining features of MyMed AI.

Instead of immediately generating an answer, the system evaluates whether retrieved evidence is sufficient.

Reflection asks questions such as

- Is enough evidence available?
- Are retrieved documents relevant?
- Is context contradictory?
- Should retrieval continue?
- Should Top-K increase?
- Is another retrieval iteration necessary?

Reflection output

```
Confidence

Sufficient Context

Suggested Top K

Reasoning

Next Action
```

This stage dramatically reduces retrieval failures.

---

# 🔄 Corrective Retrieval

If Retrieval Reflection determines that evidence is insufficient, MyMed automatically performs another retrieval cycle.

Possible corrective actions include

- Increase Top-K
- Rewrite query
- Expand search
- Remove noisy documents
- Retrieve additional chunks

Instead of producing low-quality answers, the pipeline attempts to improve retrieval first.

---

# 📦 Context Compression

Large retrieval results often exceed the LLM context window.

Compression removes

- Duplicate chunks
- Irrelevant paragraphs
- Low-confidence evidence

while preserving medically important information.

Benefits

- Lower token usage
- Faster inference
- Better grounding
- Reduced cost

---

# ✍️ Generation Layer

Only after retrieval passes reflection does generation begin.

The generation layer is provider-independent.

Supported providers

- Groq
- HuggingFace
- Ollama
- OpenAI

The abstraction layer allows providers to be switched through configuration without modifying business logic.

---

# 🪞 Answer Reflection

Generation does not mark the end of the pipeline.

The generated response undergoes another reflection stage.

Questions evaluated include

- Is the answer supported by evidence?
- Does it contradict retrieved context?
- Is medical reasoning coherent?
- Are unsupported claims present?
- Is confidence acceptable?

If validation fails, corrective retrieval can be triggered again.

---

# 📈 Confidence Engine

The Confidence Engine aggregates signals from every stage.

Signals include

- Retrieval confidence
- Reranker scores
- Reflection output
- Coverage score
- Generation quality

These are combined into a final confidence estimate.

Possible levels

- Very High
- High
- Medium
- Low
- Very Low

Rather than presenting every answer as equally reliable, MyMed communicates how trustworthy the response is.

---

# 🔭 LangSmith Observability

Every stage of the pipeline is instrumented for tracing.

Tracked components include

- Query rewriting
- Retrieval latency
- Dense retrieval
- BM25 retrieval
- RRF fusion
- MMR
- Cross Encoder
- Reflection
- Generation
- Answer Reflection
- Confidence Estimation

This enables end-to-end debugging and evaluation during development.

---

# 🔁 Complete Retrieval Flow

```text
User Query
      │
      ▼
Query Rewriter
      │
      ▼
Medical Router
      │
      ▼
Dense Search ───────┐
                    │
BM25 Search ────────┤
                    ▼
          Reciprocal Rank Fusion
                    │
                    ▼
                  MMR
                    │
                    ▼
         Cross Encoder Reranker
                    │
                    ▼
        Retrieval Reflection Agent
                    │
      ┌─────────────┴──────────────┐
      ▼                            ▼
Enough Context            Corrective Retrieval
      │                            │
      └─────────────┬──────────────┘
                    ▼
          Context Compression
                    │
                    ▼
             LLM Generation
                    │
                    ▼
          Answer Reflection
                    │
                    ▼
          Confidence Engine
                    │
                    ▼
             Final Response
```

---
---

# 🏛 Repository Structure

The project follows a modular architecture where each component is responsible for a single stage of the retrieval pipeline. This separation of concerns improves maintainability, extensibility, and testing.

```text
MyMed-AI/
│
├── backend/
│   ├── api/
│   │   ├── routes/
│   │   ├── middleware/
│   │   ├── dependencies/
│   │   └── schemas/
│   │
│   ├── authentication/
│   │   ├── jwt.py
│   │   ├── security.py
│   │   └── permissions.py
│   │
│   ├── config/
│   │   ├── settings.py
│   │   └── logging.py
│   │
│   ├── ingestion/
│   │   ├── loader.py
│   │   ├── parser.py
│   │   ├── splitter.py
│   │   └── metadata.py
│   │
│   ├── embeddings/
│   │   ├── embedding_model.py
│   │   └── embedding_service.py
│   │
│   ├── vectorstore/
│   │   ├── qdrant_client.py
│   │   ├── indexing.py
│   │   └── collections.py
│   │
│   ├── retrieval/
│   │   ├── dense.py
│   │   ├── bm25.py
│   │   ├── hybrid.py
│   │   ├── mmr.py
│   │   ├── reranker.py
│   │   └── compression.py
│   │
│   ├── self_rag/
│   │   ├── retrieval_reflection.py
│   │   ├── answer_reflection.py
│   │   ├── confidence.py
│   │   ├── pipeline.py
│   │   └── state.py
│   │
│   ├── generation/
│   │   ├── provider.py
│   │   ├── prompts.py
│   │   └── generator.py
│   │
│   ├── router/
│   ├── query_rewriter/
│   ├── evaluation/
│   ├── observability/
│   ├── utils/
│   └── main.py
│
├── frontend/
│
├── docker/
│
├── tests/
│
├── docs/
│
├── requirements.txt
├── docker-compose.yml
├── README.md
└── LICENSE
```

---

# 📂 Module Overview

## API Layer

Responsible for exposing REST endpoints, request validation, authentication, exception handling and response serialization.

---

## Authentication

Implements JWT authentication and request authorization.

Responsibilities include:

- Login
- Token generation
- Token validation
- Protected endpoints

---

## Ingestion Pipeline

Converts raw medical documents into searchable chunks.

Pipeline:

```
Documents

↓

Cleaning

↓

Chunking

↓

Metadata Extraction

↓

Embedding Generation

↓

Vector Database
```

---

## Embedding Layer

Transforms text into dense vector representations using transformer embedding models.

Current embedding model:

```
BAAI/bge-base-en-v1.5
```

Responsibilities

- Batch embeddings
- Query embeddings
- Document embeddings

---

## Vector Store

Responsible for

- Collection management
- Index creation
- Document storage
- Similarity search

Current implementation:

```
Qdrant
```

---

## Retrieval Layer

Contains all retrieval strategies.

Modules include

- Dense Retrieval
- BM25
- Hybrid Retrieval
- Reciprocal Rank Fusion
- MMR
- Cross Encoder
- Compression

Each module is independently replaceable.

---

## Self-RAG

Implements reflection-based retrieval improvement.

Modules

```
Retrieval Reflection

↓

Corrective Retrieval

↓

Answer Reflection

↓

Confidence Engine
```

---

## Generation

Provider-independent text generation.

Supported providers

- Groq
- HuggingFace
- Ollama
- OpenAI

Adding a new provider only requires implementing a provider adapter.

---

## Observability

LangSmith integration records every execution stage.

Collected metrics include

- Latency
- Tokens
- Retrieval scores
- Reflection outputs
- Confidence
- Errors

---

# 🛠 Technology Stack

| Category | Technology |
|-----------|------------|
| Language | Python 3.12 |
| Backend | FastAPI |
| Frontend | React + Vite |
| Authentication | JWT |
| Vector Database | Qdrant |
| Embeddings | BAAI/bge-base-en-v1.5 |
| Reranker | BAAI/bge-reranker-base |
| Framework | LangChain |
| Observability | LangSmith |
| Database | SQLite / PostgreSQL |
| Containerization | Docker |

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/Dhruv716-stack/-MyMed-Medical-RAG-System-.git

cd -MyMed-Medical-RAG-System-
```

---

## Create Virtual Environment

Windows

```bash
python -m venv venv

venv\Scripts\activate
```

Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🐳 Docker Deployment

Start Qdrant

```bash
docker run -p 6333:6333 qdrant/qdrant
```

or

```bash
docker compose up -d
```

Check status

```bash
docker ps
```

---

# 🚀 Running the Backend

```bash
uvicorn backend.main:app --reload
```

Server

```
http://localhost:8000
```

Swagger

```
http://localhost:8000/docs
```

ReDoc

```
http://localhost:8000/redoc
```

---

# ⚛ Running the Frontend

```bash
cd frontend

npm install

npm run dev
```

Application

```
http://localhost:5173
```

---

# 🔑 Environment Variables

```env
#####################################
# Authentication
#####################################

JWT_SECRET_KEY=

JWT_ALGORITHM=HS256

ACCESS_TOKEN_EXPIRE_MINUTES=30

#####################################
# Qdrant
#####################################

QDRANT_HOST=localhost

QDRANT_PORT=6333

QDRANT_COLLECTION=medical_rag_documents

#####################################
# Embeddings
#####################################

EMBEDDING_MODEL=BAAI/bge-base-en-v1.5

#####################################
# Reranker
#####################################

RERANKER_MODEL=BAAI/bge-reranker-base

#####################################
# LangSmith
#####################################

LANGCHAIN_TRACING_V2=true

LANGCHAIN_API_KEY=

#####################################
# Providers
#####################################

GROQ_API_KEY=

OPENAI_API_KEY=

HUGGINGFACEHUB_API_TOKEN=

OLLAMA_BASE_URL=http://localhost:11434

#####################################
# Database
#####################################

DATABASE_URL=sqlite:///medical.db
```

---

# ⚙ Configuration

The configuration system uses a centralized settings module built on **Pydantic Settings**.

Benefits include:

- Environment-based configuration
- Strong typing
- Validation
- Default values
- Easy deployment

---

# 🔌 API Reference

## Authentication

| Method | Endpoint | Description |
|----------|----------|-------------|
| POST | `/login` | Authenticate user |
| POST | `/register` | Register user |

---

## Medical Query

| Method | Endpoint | Description |
|----------|----------|-------------|
| POST | `/query` | Generate medical response |

---

## Ingestion

| Method | Endpoint | Description |
|----------|----------|-------------|
| POST | `/ingest` | Upload and index documents |

---

## Health

| Method | Endpoint | Description |
|----------|----------|-------------|
| GET | `/health` | Application status |

---

## Vector Database

| Method | Endpoint | Description |
|----------|----------|-------------|
| POST | `/reindex` | Rebuild embeddings |
| GET | `/collections` | List collections |

---

# 📈 Performance Goals

| Metric | Target |
|----------|---------|
| Retrieval Latency | < 300 ms |
| Generation Latency | < 3 s |
| Reflection Latency | < 500 ms |
| End-to-End Response | < 5 s |
| Retrieval Precision | High |
| Confidence Calibration | Production Ready |

---

# 🧪 Testing

Run all tests

```bash
pytest
```

Run with coverage

```bash
pytest --cov=backend
```

---
---

# 📊 Performance & Evaluation

MyMed AI is designed with a strong emphasis on retrieval quality, factual grounding, and production-grade observability rather than raw LLM generation.

The system continuously evaluates every stage of the retrieval pipeline to ensure that responses remain explainable, evidence-backed, and reliable.

## Target Performance

| Metric | Goal |
|----------|------|
| Query Rewriting | < 150 ms |
| Dense Retrieval | < 150 ms |
| BM25 Retrieval | < 100 ms |
| Hybrid Retrieval | < 250 ms |
| Cross Encoder Reranking | < 800 ms |
| Reflection Pipeline | < 500 ms |
| Total Response Time | < 5 sec |

---

# 📈 Evaluation Strategy

The retrieval pipeline is evaluated independently from the language model.

Current evaluation focuses on

- Retrieval Precision
- Recall
- Context Coverage
- Groundedness
- Faithfulness
- Hallucination Reduction
- Confidence Calibration

Future releases will integrate automated benchmarking using:

- RAGAS
- DeepEval
- LangSmith Evaluation
- Custom Medical QA Benchmark

---

# 🤖 AI Models

## Embedding Model

**BAAI/bge-base-en-v1.5**

Purpose

- Semantic embeddings
- Dense retrieval
- Query representation

Advantages

- Strong retrieval performance
- Lightweight
- Open source
- High benchmark scores

---

## Cross Encoder

**BAAI/bge-reranker-base**

Purpose

- Document reranking
- Precision improvement

Advantages

- Better than cosine similarity
- Joint query-document encoding
- Production-grade ranking

---

## Generation Model

Provider independent

Supported providers

- Groq
- OpenAI
- Ollama
- HuggingFace

The architecture separates provider logic from business logic, making the generation layer easily extensible.

---

# 🔒 Security

Security is considered throughout the system design.

Current protections include

- JWT Authentication
- Password hashing
- Request validation
- Input sanitization
- Configuration through environment variables
- Protected API endpoints

Future improvements

- OAuth2
- RBAC
- API rate limiting
- Audit logging
- Secrets management
- Encryption at rest

---

# 📸 Screenshots

> Replace these placeholders with actual project screenshots.

## Dashboard

```text
docs/screenshots/dashboard.png
```

---

## Query Interface

```text
docs/screenshots/query.png
```

---

## Retrieval Pipeline

```text
docs/screenshots/pipeline.png
```

---

## LangSmith Trace

```text
docs/screenshots/langsmith.png
```

---

## Confidence Report

```text
docs/screenshots/confidence.png
```

---

# 🎥 Demo

Demo GIF

```text
docs/demo/demo.gif
```

Demo Video

```text
https://youtube.com/your-demo
```

---

# 💡 Example Query

Input

```
What are the early symptoms of diabetes?
```

Pipeline

```
Query Rewriter

↓

Medical Router

↓

Hybrid Retrieval

↓

MMR

↓

Cross Encoder

↓

Reflection

↓

Generation

↓

Confidence Engine
```

Output

```
Answer

Confidence

Retrieved Sources

Reasoning
```

---

# 🛣 Roadmap

## Retrieval

- [ ] Multi-query retrieval
- [ ] Knowledge Graph integration
- [ ] GraphRAG
- [ ] Multi-vector retrieval
- [ ] Parent-child retrieval
- [ ] Adaptive chunking

---

## Intelligence

- [ ] Agentic RAG
- [ ] Multi-agent reasoning
- [ ] Medical ontology integration
- [ ] Tool calling
- [ ] Function calling
- [ ] Clinical guideline integration

---

## Infrastructure

- [ ] Kubernetes deployment
- [ ] CI/CD
- [ ] Redis caching
- [ ] PostgreSQL production deployment
- [ ] Monitoring dashboard
- [ ] Horizontal scaling

---

## User Experience

- [ ] Streaming responses
- [ ] Conversation memory
- [ ] Voice interface
- [ ] Mobile application
- [ ] Dark mode improvements

---

# ⚠ Limitations

Current limitations include

- English language support only
- General medical information only
- No real-time clinical database integration
- Not a replacement for professional medical advice
- Depends on indexed knowledge quality

These limitations will be addressed in future iterations.

---

# ❓ FAQ

## Why Hybrid Retrieval?

Hybrid retrieval combines semantic understanding with keyword matching, improving recall and precision compared to using either method alone.

---

## Why Self-RAG?

Self-RAG evaluates retrieved evidence before generation, reducing hallucinations and improving factual consistency.

---

## Why Corrective RAG?

Corrective RAG enables the system to recover from poor retrieval by performing additional search iterations instead of generating low-quality responses.

---

## Why Reflection?

Reflection allows the system to assess both retrieved evidence and generated responses, improving overall reliability.

---

## Why Cross Encoder?

Cross Encoders provide significantly more accurate document ranking than vector similarity alone.

---

# 🐛 Troubleshooting

## Qdrant connection refused

Check

```bash
docker ps
```

Ensure the container is running.

---

## No documents retrieved

Verify

- Embeddings generated
- Collection exists
- Documents indexed

---

## Empty LLM response

Check

- Provider API key
- Internet connectivity
- Model availability

---

## LangSmith not recording traces

Verify

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=YOUR_KEY
```

---

# 🤝 Contributing

Contributions are welcome.

To contribute

```bash
Fork Repository

↓

Create Feature Branch

↓

Implement Changes

↓

Run Tests

↓

Submit Pull Request
```

Please open an issue before implementing major architectural changes.

---

# 👥 Contributors

| Contributor |
|--------------|
| **avik106-Avik Sarkar** |-Avik Sarkar
| **Dhruv716-stack-Dhruv Sahu** |-Dhruv Sahu

---

# 📚 References

- Retrieval-Augmented Generation (Lewis et al.)
- Self-RAG
- Corrective RAG
- LangChain
- Qdrant
- Hugging Face
- FastAPI
- LangSmith

---

# 📄 License

This project is licensed under the **MIT License**.

See the LICENSE file for more information.

---

# ⭐ Support the Project

If you found this repository useful,

⭐ Star the repository

🍴 Fork the project

🐛 Report issues

💬 Share feedback

---

<div align="center">

# 🩺 MyMed AI

### *Building Trustworthy Medical Intelligence through Reflection, Retrieval, and Reasoning.*

---

**Hybrid Retrieval • Self-RAG • Corrective RAG • Reflection • Confidence-Aware Generation**

Made with ❤️ by the MyMed AI Team

</div>