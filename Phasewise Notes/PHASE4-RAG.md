# 🥇 Phase 4 — RAG (Retrieval-Augmented Generation)

This phase introduces **RAG**, allowing the AI to answer questions using **user-provided documents** instead of relying only on model training data.

---

## 🎯 Goals

- Upload custom documents
- Store document content
- Inject document context into prompts
- Generate grounded answers based on uploaded data

---

## 🧠 What is RAG?

RAG = **Retrieval + Augmentation + Generation**

Flow:

User question  
↓  
Retrieve relevant document content  
↓  
Inject content into prompt  
↓  
LLM generates answer

---

## 🔧 Implementation

### 1. Document Upload

Added a new endpoint:

POST `/documents/upload`

This accepts a text file using Spring’s `MultipartFile`.

---

### 2. Document Storage

Stored uploaded content in memory:

```java
private String documentContext = "";