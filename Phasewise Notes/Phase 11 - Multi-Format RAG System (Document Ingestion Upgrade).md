# 📘 Phase 11 — Multi-Format RAG System (Document Ingestion Upgrade)

## 🧭 Overview

Phase 11 upgrades the AI system from a basic PDF-only ingestion pipeline into a **multi-format, production-style Retrieval-Augmented Generation (RAG) system**.

The system now supports:

- Multiple document types (PDF, DOCX, TXT)
- Pluggable document extraction architecture
- Metadata-aware embeddings
- Persistent vector storage using ChromaDB
- Session-based AI conversations
- Basic document lifecycle management using soft delete

---

# 🏗️ Architecture

## End-to-End Flow

```text
Upload File
    ↓
DocumentController
    ↓
DocumentExtractionService
    ↓
DocumentExtractor (Strategy Pattern)
    ↓
Plain Text
    ↓
DocumentService
    ↓
Recursive Document Chunking
    ↓
Embedding Generation (Ollama)
    ↓
Chroma Vector Database
```

---

# 📂 Multi-Format Document Support

## Supported Formats

### PDF

- Text extracted using Apache PDFBox
- Binary PDF converted into plain text

### TXT

- Plain text extracted directly from uploaded file

### DOCX

- Implemented using Apache POI
- Microsoft Word documents converted into plain text before processing

---

# 🔌 Document Extraction Architecture

Instead of checking file extensions in the controller, document extraction now uses the **Strategy Pattern**.

```
DocumentExtractor
│
├── PdfDocumentExtractor
├── DocxDocumentExtractor
└── TxtDocumentExtractor
```

Each extractor is responsible for:

- Determining whether it supports the uploaded file
- Extracting plain text
- Returning text in a common format for downstream processing

Adding support for a new document type now requires creating a single new extractor class without modifying existing code.

---

# 📤 Document Upload Pipeline

Endpoint:

```http
POST /documents/upload
```

Processing steps:

1. Receive uploaded file
2. Detect supported extractor
3. Extract document text
4. Store document metadata
5. Split document into chunks
6. Generate embeddings
7. Store embeddings in ChromaDB

---

# 🧩 Document Chunking

Implemented using LangChain4j's recursive splitter.

```java
DocumentSplitters.recursive(500, 100);
```

Configuration:

- Chunk size: 500
- Chunk overlap: 100

Advantages:

- Preserves semantic meaning
- Improves retrieval quality
- Maintains context across chunk boundaries

---

# 🧠 Embedding Generation

Embedding model:

```
nomic-embed-text
```

Running through Ollama.

Responsibilities:

- Convert document chunks into vector embeddings
- Enable semantic similarity search

---

# 🗄️ Vector Database

Database:

```
ChromaDB
```

Each chunk is stored together with metadata.

Example metadata:

```json
{
  "documentId": "uuid",
  "filename": "spring-guide.pdf",
  "source": "upload",
  "uploadedAt": "2026-06-27"
}
```

---

# 🧾 Metadata

Each stored chunk includes:

| Field | Purpose |
|--------|----------|
| documentId | Groups all chunks belonging to the same document |
| filename | Display and traceability |
| source | Origin of ingestion |
| uploadedAt | Audit information |

---

# 🤖 Retrieval-Augmented Generation (RAG)

Query flow:

```
User Question
        ↓
Generate Query Embedding
        ↓
Search ChromaDB
        ↓
Retrieve Similar Chunks
        ↓
Construct Prompt
        ↓
Llama 3.1
        ↓
Answer
```

The AI assistant now answers using retrieved document context rather than relying only on the language model's pretrained knowledge.

---

# 💬 Session-Based Chat Memory

Implemented using:

```java
ChatMemoryProvider
```

Features:

- Separate conversation history per session
- Maximum of 20 messages per session
- Stored using a ConcurrentHashMap

---

# 📄 Document Management

A relational table (`UploadedDocument`) stores document metadata.

Current fields include:

- documentId
- filename
- source
- uploadDate
- active

This table acts as the application's source of truth for uploaded documents.

---

# 🗑️ Soft Delete

Because the current LangChain4j Chroma integration does not expose metadata-based deletion APIs, documents are managed using a soft delete strategy.

Instead of deleting vectors:

```
DELETE
    ↓
active = false
```

Benefits:

- Prevents accidental data loss
- Keeps document history
- Simplifies document lifecycle management
- Avoids limitations of the current Chroma abstraction

---

# 📄 Document Listing API

Endpoint:

```http
GET /documents/active
```

Returns:

```json
[
  {
    "documentId": "...",
    "filename": "spring.pdf",
    "source": "upload",
    "active": true
  }
]
```

Purpose:

- View uploaded documents
- Obtain document IDs
- Support deletion
- Debug ingestion

---

# 🧪 Testing Workflow

1. Upload document

```
POST /documents/upload
```

2. View active documents

```
GET /documents/active
```

3. Ask AI questions

```
GET /ai/ask
```

4. Soft delete document

```
DELETE /documents/{documentId}
```

5. Verify document no longer appears in active document list.

---

# ⚠️ Current Limitations

- Chroma vectors are not physically deleted.
- Soft delete currently affects SQL metadata only.
- Application-level filtering still needs to be fully integrated so inactive documents are excluded from retrieval.
- Versioning and hard deletion are planned for future phases.

---

# 🎯 Key Learning Outcomes

During this phase the following AI engineering concepts were implemented:

- Retrieval-Augmented Generation (RAG)
- Multi-format document ingestion
- Strategy Pattern for document loaders
- Recursive document chunking
- Embedding generation
- Metadata-aware vector storage
- Session-based chat memory
- Hybrid SQL + Vector Database architecture
- Soft delete document lifecycle management

---

# ✅ Phase 11 Outcome

At the end of this phase, the application supports:

- PDF ingestion
- DOCX ingestion
- TXT ingestion
- Recursive document chunking
- Embedding generation using Ollama
- Persistent storage in ChromaDB
- Metadata-aware document storage
- Session-aware conversations
- Basic document management
- Soft delete architecture

The project has evolved from a simple chatbot into a modular Retrieval-Augmented Generation system capable of ingesting and querying knowledge from multiple document formats.