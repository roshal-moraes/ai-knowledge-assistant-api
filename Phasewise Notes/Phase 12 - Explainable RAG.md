# 📘 Phase 12 — Explainable Retrieval-Augmented Generation (Explainable RAG)

## 🧭 Overview

In Phase 12, the project moved beyond simply generating AI answers. The focus shifted to understanding **how Retrieval-Augmented Generation (RAG) works internally**.

Instead of treating the AI as a black box, the application now exposes each stage of the retrieval pipeline, making it possible to inspect, debug, and improve the quality of AI responses.

The primary goal of this phase was to make the RAG pipeline **transparent and explainable**.

---

# 🎯 Objectives

At the beginning of this phase, the application could:

* Accept user questions
* Retrieve relevant document chunks
* Generate AI responses

However, it was difficult to determine:

* Why a particular answer was generated
* Which document chunks were retrieved
* Whether poor answers were caused by retrieval or by the language model
* How retrieval parameters affected response quality

Phase 12 addressed these problems by exposing the intermediate stages of the RAG pipeline.

---

# 🏗 Explainable RAG Architecture

The pipeline now looks like:

```text
User Question
        │
        ▼
Generate Query Embedding
        │
        ▼
Semantic Search (ChromaDB)
        │
        ▼
Retrieved Chunks
        │
        ▼
Prompt Construction
        │
        ▼
Large Language Model
        │
        ▼
Final Answer
```

Unlike previous phases, every stage can now be inspected independently.

---

# 📦 RetrievedChunk DTO

A dedicated DTO was introduced:

```java
RetrievedChunk
```

Each retrieved chunk contains:

* filename
* similarity score
* retrieved text

Example:

```json
{
    "filename": "spring.pdf",
    "score": 0.94,
    "text": "Dependency Injection is..."
}
```

Using a DTO instead of LangChain4j classes decouples the REST API from the underlying AI framework and produces cleaner, more maintainable APIs.

---

# 🔍 Retrieval Debugging

A new retrieval debugging feature was implemented.

Instead of immediately asking the LLM, the application first retrieves the most relevant chunks and returns them directly.

Endpoint:

```http
GET /documents/debug/retrieve?q=dependency injection
```

This endpoint allows developers to inspect:

* Which chunks were retrieved
* Which documents they originated from
* Their similarity scores
* The exact text that would be provided to the language model

No AI generation occurs during this step.

---

# 🧠 Why Retrieval Debugging Matters

Retrieval debugging helps determine whether poor answers originate from:

* Incorrect chunking
* Weak embeddings
* Poor similarity search
* Incorrect retrieval thresholds

Without this visibility, every incorrect answer appears to be an AI model problem, even when retrieval is actually responsible.

---

# 📋 Full RAG Trace

A new DTO was introduced:

```java
DebugRagResponse
```

It represents the complete execution of a Retrieval-Augmented Generation request.

Fields:

* question
* retrievedChunks
* prompt
* answer

Example:

```json
{
    "question": "...",
    "retrievedChunks": [...],
    "prompt": "...",
    "answer": "..."
}
```

This provides a complete snapshot of the RAG pipeline for debugging and learning purposes.

---

# 🤖 Prompt Construction

Instead of sending raw document chunks to the LLM, retrieved content is now formatted into a structured prompt.

Example:

```text
Document: spring.pdf

Similarity Score: 0.94

Constructor injection is preferred...

----------------------------

Document: java-time.pdf

Similarity Score: 0.82

LocalDateTime represents...
```

Benefits:

* Clear document boundaries
* Better context organization
* Easier debugging
* Improved response quality

---

# 🌐 AI Debug Endpoint

A dedicated endpoint was introduced:

```http
GET /ai/debug
```

Example:

```http
GET /ai/debug?sessionId=user1&q=What is dependency injection?
```

Response:

```json
{
    "question": "...",
    "retrievedChunks": [...],
    "prompt": "...",
    "answer": "..."
}
```

This endpoint demonstrates every stage of the RAG pipeline in a single response.

---

# 🧪 Retrieval Evaluation

The debugging endpoints make it possible to evaluate retrieval quality.

Questions that can now be answered include:

* Was the correct document retrieved?
* Was the most relevant chunk ranked first?
* Was the similarity score reasonable?
* Was enough context retrieved?
* Did the prompt contain the expected information?

These questions are essential when improving RAG systems.

---

# 📖 Important Concepts Learned

## Retrieval-Augmented Generation (RAG)

RAG combines semantic search with a language model.

Instead of relying solely on pretrained knowledge, the AI first retrieves relevant information from an external knowledge base before generating a response.

---

## Semantic Search

Semantic search retrieves information based on meaning rather than exact keywords.

Instead of matching identical words, embeddings are compared using vector similarity.

This allows queries such as:

```
How does Spring inject objects?
```

to retrieve documents discussing:

```
Dependency Injection
```

even when the wording differs.

---

## Embeddings

Embeddings convert text into high-dimensional numerical vectors.

Similar meanings produce vectors that are close together in vector space.

Both documents and user queries are embedded using the same embedding model, allowing similarity comparisons.

---

## Vector Similarity Search

Instead of searching for keywords, ChromaDB compares embedding vectors.

The closest vectors are returned as the most relevant document chunks.

The similarity score represents how closely the retrieved chunk matches the user's query.

---

## Explainable AI

One of the most valuable additions in this phase is explainability.

Instead of simply accepting an AI answer, developers can inspect:

* Retrieved documents
* Similarity scores
* Constructed prompt
* Final response

This greatly simplifies debugging and increases trust in AI-generated answers.

---

## Separation of Responsibilities

The project architecture was further improved.

### DocumentService

Responsible for:

* Document ingestion
* Chunking
* Embedding generation
* Retrieval
* Retrieval debugging

### AIService

Responsible for:

* Prompt construction
* Chat memory
* AI interaction
* Answer generation

This separation follows the Single Responsibility Principle and improves maintainability.

---

# 🎯 Key Learning Outcomes

By the end of Phase 12, the project demonstrates:

* Explainable Retrieval-Augmented Generation
* Retrieval debugging
* Similarity score inspection
* Prompt inspection
* Structured context construction
* Transparent AI reasoning
* Separation between retrieval and generation
* Cleaner service architecture
* DTO-based API design

---

# 🚀 Phase 12 Outcome

The application has evolved from a simple RAG chatbot into an **Explainable RAG system**.

Developers can now inspect the complete AI pipeline, making it significantly easier to:

* Understand AI behavior
* Improve retrieval quality
* Debug incorrect answers
* Tune embeddings and chunking strategies
* Experiment with prompt engineering

This phase establishes a strong foundation for building production-grade Retrieval-Augmented Generation systems.

---
