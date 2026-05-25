# AI Demo – Phase 7: Tool Calling with LangChain4j + Ollama + ChromaDB

This phase upgrades the AI assistant into a **tool-augmented RAG system** using LangChain4j.  
The system can now retrieve documents (RAG), maintain conversation memory, and call Java tools dynamically.

---

## 🚀 What’s New in Phase 7

### 🧠 1. Tool Calling (AI Agents)
- Integrated LangChain4j `AiServices`
- AI can now call Java methods as tools
- Added `TimeTools` as a system utility tool
- Enables AI to respond with real-time system data (e.g., current time)

---

### 🤖 2. LLM Upgrade (Ollama)
- Switched from `phi3` → `llama3.1`
- Reason: Tool calling support is required
- Model runs locally via Ollama

---

### 📚 3. RAG Pipeline (unchanged but integrated)
- User questions are embedded using `OllamaEmbeddingModel`
- Stored in `ChromaEmbeddingStore`
- Top-k relevant document chunks are retrieved
- Injected into prompt context for better answers

---

### 💬 4. Conversation Memory
- Maintains last N messages in-memory
- Adds conversational continuity across queries

---

## ⚙️ How Tool Calling Works

1. User asks a question  
2. AI decides whether a tool is needed  
3. If required, AI calls Java tool (`TimeTools`)  
4. Tool executes and returns result  
5. AI uses result to form final response  

---

## 🧩 Key Components

### AIService
- Handles:
  - embedding search (RAG)
  - prompt building
  - tool-enabled AI response
  - conversation memory

---

### AIAssistant (LangChain4j)
```java
public interface AIAssistant {
    String chat(String message);
}