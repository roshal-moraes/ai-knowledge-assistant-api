# Phase 9 – Conversation Memory & Multi-User Sessions

## Goal

Enable the AI assistant to remember previous messages within a conversation and maintain separate memories for different users.

---

# Problem Before Phase 9

The assistant treated every request as completely independent.

Example:

```text
User: My name is Roshal.
AI: Nice to meet you, Roshal.

User: What is my name?
AI: I don't know.
```

There was no conversation history.

---

# What is Chat Memory?

Chat memory stores previous messages and sends them back to the model with each new request.

Example:

```text
User: My name is Roshal.
AI: Nice to meet you.

User: What is my name?
AI: Your name is Roshal.
```

The model can answer because previous messages are included in the prompt.

---

# MessageWindowChatMemory

LangChain4j provides:

```java
MessageWindowChatMemory
```

This stores only the most recent messages.

Example:

```java
MessageWindowChatMemory.withMaxMessages(20);
```

Behavior:

```text
Message 1
Message 2
...
Message 20

Message 21 arrives

Message 1 is removed
```

This prevents memory from growing indefinitely.

---

# Initial Memory Configuration

Added a ChatMemory bean:

```java
@Bean
public ChatMemory chatMemory() {
    return MessageWindowChatMemory.withMaxMessages(20);
}
```

Configured memory in the AI service:

```java
.chatMemory(chatMemory)
```

Result:

```text
Single shared memory
```

All users shared the same conversation history.

---

# Problem with Shared Memory

Example:

```text
User A:
My name is Alice.

User B:
What is my name?

AI:
Your name is Alice.
```

Memory leakage occurs because all users share the same memory.

This is unacceptable in real-world systems.

---

# Session-Based Memory

Each user should have independent memory.

Example:

```text
Session A
 └─ remembers Alice

Session B
 └─ remembers Bob

Session C
 └─ remembers only its own data
```

---

# ChatMemoryProvider

LangChain4j provides:

```java
ChatMemoryProvider
```

Instead of a single memory object, a memory instance is created for each session.

Example:

```java
@Bean
public ChatMemoryProvider chatMemoryProvider() {

    return memoryId ->
            MessageWindowChatMemory.withMaxMessages(20);
}
```

---

# Using @MemoryId

Updated the assistant interface:

```java
public interface AIAssistant {

    String chat(
        @MemoryId String memoryId,
        String message
    );
}
```

The memory ID determines which conversation history is used.

---

# Configuring AI Service

Before:

```java
String ask(String question)
```

After:

```java
String ask(String sessionId,
           String question)
```

Example:

```java
return aiAssistant.chat(sessionId, question);
```

---

# Controller Changes

Before:

```java
/ask?question=hello
```

After:

```java
/ask?sessionId=user1
    &question=hello
```

The session ID now controls memory retrieval.

---

# Improving Memory Storage

Implemented memory reuse with:

```java
ConcurrentHashMap
```

Example:

```java
@Bean
public ChatMemoryProvider chatMemoryProvider() {

    Map<Object, ChatMemory> memories =
            new ConcurrentHashMap<>();

    return memoryId ->
            memories.computeIfAbsent(
                    memoryId,
                    id -> MessageWindowChatMemory
                            .withMaxMessages(20)
            );
}
```

Benefits:

* Reuses existing memory
* Avoids duplicate memory creation
* Supports concurrent users

---

# Testing

## Session 1

Request:

```text
My name is Roshal.
```

Request:

```text
What is my name?
```

Response:

```text
Your name is Roshal.
```

---

## Session 2

Request:

```text
What is my name?
```

Response:

```text
I don't know your name yet.
```

Memory is isolated per session.

---

# Architecture After Phase 9

```text
User Request
      ↓
Session ID
      ↓
ChatMemoryProvider
      ↓
Session Memory
      ↓
RAG Retrieval
      ↓
Ollama (Phi3)
      ↓
Response
```

---

# Key Concepts Learned

## ChatMemory

Stores conversation history.

---

## MessageWindowChatMemory

Sliding-window memory implementation.

---

## ChatMemoryProvider

Creates memory per user/session.

---

## @MemoryId

Maps a request to a specific memory instance.

---

## Session Isolation

Prevents users from sharing conversation history.

---

## ConcurrentHashMap

Stores memory safely for multiple users.

---

# Current Capabilities

✅ Local LLM (Ollama)

✅ Prompt Engineering

✅ Tool Integration

✅ RAG with ChromaDB

✅ Document Retrieval

✅ Conversation Memory

✅ Multi-User Sessions

---

# Limitations

Current memory is:

```text
In-Memory (RAM)
```

When the application restarts:

```text
All conversations are lost.
```

---

# Next Phase

## Phase 10 – Streaming Responses

Current behavior:

```text
Wait...
Wait...
Full response appears.
```

Target behavior:

```text
The...
The capital...
The capital of France...
The capital of France is Paris.
```

Streaming will provide:

* Real-time token generation
* Faster perceived responses
* ChatGPT-like user experience
* Server-Sent Events (SSE)
* StreamingChatModel integration

---

# Phase 9 Summary

Implemented conversation memory and session-based memory management using LangChain4j.

The assistant can now:

* Remember previous messages
* Maintain context across requests
* Support multiple independent users
* Isolate conversation history per session

This transforms the application from a simple chatbot into a conversational AI assistant.
