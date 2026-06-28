# Phase 10 — Streaming AI Responses with Server-Sent Events (SSE)

---

# Objective

Until Phase 9, the application waited for the Large Language Model (LLM) to finish generating the entire response before returning anything to the user.

Example:

User:

> Tell me about Spring Boot.

The application flow was:

```text
User
 ↓
Controller
 ↓
Service
 ↓
AIAssistant
 ↓
Ollama
 ↓
Complete response generated
 ↓
Return response to user
```

The user only saw the answer after generation was completely finished.

For long answers this introduces unnecessary waiting.

The goal of Phase 10 is to stream responses token-by-token so the user can see the answer appearing immediately.

---

# Why Streaming?

Modern AI applications such as:

- ChatGPT
- Claude
- Gemini
- Perplexity

all stream tokens while the model is generating.

Instead of:

```text
(wait...)
(wait...)
(wait...)

Here is the complete answer.
```

users see:

```text
Here
Here is
Here is the
Here is the answer...
```

Streaming improves:

- perceived speed
- user experience
- responsiveness
- interactivity

---

# Existing Architecture (Phase 9)

```text
Controller
    ↓
AIService
    ↓
AIAssistant
    ↓
ChatModel
    ↓
String response
```

Everything returned a complete `String`.

---

# New Streaming Architecture

A completely separate pipeline was introduced to avoid breaking previous phases.

```text
StreamingController
        ↓
StreamingAIService
        ↓
StreamingAssistant
        ↓
StreamingChatModel
        ↓
TokenStream
        ↓
SseEmitter
        ↓
Browser
```

The original pipeline from Phases 1–9 remains untouched.

---

# Why Create Separate Components?

Instead of modifying:

```text
AIAssistant
AIService
AIController
```

new classes were introduced:

- StreamingAssistant
- StreamingAIService
- StreamingController

Benefits:

- zero impact on existing functionality
- easier debugging
- cleaner architecture
- both approaches can coexist

---

# StreamingChatModel

Added:

```java
@Bean
public StreamingChatModel streamingChatModel()
```

using:

```java
OllamaStreamingChatModel
```

Unlike `OllamaChatModel`, this model produces tokens incrementally.

---

# StreamingAssistant

Created:

```java
public interface StreamingAssistant
```

with:

```java
TokenStream chat(
        @MemoryId String memoryId,
        @UserMessage String message
);
```

## Important Difference

Normal assistant:

```java
String chat(...)
```

Streaming assistant:

```java
TokenStream chat(...)
```

---

# Session Memory Support

The same `ChatMemoryProvider` from Phase 9 is reused.

This means:

```text
sessionId=user1
```

and

```text
sessionId=user2
```

maintain completely separate conversations.

Streaming does not affect memory behavior.

---

# StreamingAIService

Created:

```java
StreamingAIService
```

Responsibilities:

- invoke StreamingAssistant
- receive TokenStream
- convert tokens into SSE events
- complete stream when generation finishes

---

# SseEmitter

Spring provides:

```java
SseEmitter
```

which allows the server to continuously push data to the client.

Instead of:

```java
return String;
```

the controller now returns:

```java
return SseEmitter;
```

---

# TokenStream Callbacks

LangChain4j provides:

### onPartialResponse()

Called every time a token arrives.

Example:

```text
"The"
" capital"
" of"
" France"
```

Each token is sent immediately to the client.

---

### onCompleteResponse()

Called when generation finishes.

Used to:

```java
emitter.complete();
```

---

### onError()

Handles streaming failures.

Used to:

```java
emitter.completeWithError(...)
```

---

# Streaming Endpoint

Created:

```text
GET /stream
```

Example:

```text
http://localhost:8080/stream?sessionId=user1&message=Hello
```

Parameters:

| Parameter | Description |
|------------|-------------|
| sessionId | identifies the conversation |
| message | user prompt |

---

# Important LangChain4j Requirement

The method:

```java
TokenStream chat(
        @MemoryId String memoryId,
        String message
)
```

caused:

```text
IllegalConfigurationException
```

because LangChain4j requires parameters to be annotated.

Correct version:

```java
TokenStream chat(
        @MemoryId String memoryId,
        @UserMessage String message
)
```

Without `@UserMessage`, LangChain4j cannot determine which parameter contains the user's prompt.

---

# Result

Before Phase 10:

```text
(wait...)
(wait...)
(wait...)

Full response appears.
```

After Phase 10:

```text
H
He
Hel
Hello
Hello,
Hello, how
Hello, how can...
```

Responses appear progressively.

---

# Benefits Achieved

✅ Real-time AI responses

✅ Better user experience

✅ No breaking changes to previous phases

✅ Separate streaming pipeline

✅ Session memory still works

✅ Token-level generation

✅ Production-style architecture

---

# Classes Added

```text
assistant/
    StreamingAssistant.java

service/
    StreamingAIService.java

controller/
    StreamingController.java
```

---

# Beans Added

```java
streamingChatModel()
streamingAssistant()
```

---

# Existing Components Reused

```java
ChatMemoryProvider
```

from Phase 9.

---

# Concepts Learned

### StreamingChatModel

Generates tokens incrementally.

---

### TokenStream

Represents a live response stream.

---

### Server-Sent Events (SSE)

Allows the server to continuously push updates to clients.

---

### SseEmitter

Spring abstraction for SSE responses.

---

### Callback-based Programming

Using:

- onPartialResponse()
- onCompleteResponse()
- onError()

to react to streaming events.

---

# Current Project Progress

Completed:

- Phase 1 — Basic Ollama integration
- Phase 2 — REST endpoints
- Phase 3 — AI Services
- Phase 4 — Tool calling
- Phase 5 — RAG fundamentals
- Phase 6 — Embeddings
- Phase 7 — Persistent ChromaDB
- Phase 8 — Document ingestion
- Phase 9 — Session-based memory
- Phase 10 — Streaming responses

---

# Next Phase

Phase 11 will focus on improving Retrieval-Augmented Generation (RAG) quality with advanced retrieval techniques and better context handling.