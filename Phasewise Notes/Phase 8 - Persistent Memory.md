# Phase 8 – Persistent Memory

## Features Added
- Replaced ArrayList conversation history with Spring Data JPA
- Added ChatMessage entity for storing chat records
- Implemented ChatMessageRepository
- Stored both user and AI responses in H2 database
- Enabled file-based persistence so conversations survive application restarts

## Flow

User Message
↓
Save to Database
↓
Load Conversation History
↓
RAG Retrieval
↓
LLM Response
↓
Save AI Response