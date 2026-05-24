# 🗄️ Phase 6 — Persistent Vector Storage with Chroma

This phase upgrades the application from an **in-memory embedding store** to a **persistent vector database** using Chroma.

---

## 🎯 Goals

- Persist embeddings outside application memory
- Keep uploaded documents searchable after app restart
- Use a real vector database
- Learn dependency/version management
- Move toward production-ready RAG architecture

---

## Why Persistence Matters

Previously, embeddings were stored using:

```java
InMemoryEmbeddingStore<TextSegment>