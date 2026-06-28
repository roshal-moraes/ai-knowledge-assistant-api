# 🤖 AI Knowledge Assistant (Spring Boot + Ollama)

A simple AI-powered backend application built using **Spring Boot** that integrates with a **local LLM (Llama 3 via Ollama)** to answer user questions through a REST API.

---

## 🚀 Features

* REST API for asking AI questions
* Clean backend architecture (Controller → Service)
* Local AI integration (no API key required)
* JSON-based request/response handling
* Beginner-friendly and extensible design

---

## 🧱 Architecture Overview

```
Client (Browser / Postman)
        ↓
Controller (API Layer)
        ↓
Service (Business Logic)
        ↓
Ollama (Local AI Model)
        ↓
Response → Client
```

---

## 📡 API Endpoint

### Ask a Question

```
GET /ai/ask?q=YourQuestion
```

### Example

```
http://localhost:8080/ai/ask?q=What is Spring Boot?
```

### Response

```json
{
  "answer": "Spring Boot is a Java framework..."
}
```

---

## 🧠 AI Concepts Explained

This project demonstrates core AI concepts used in modern applications:

---

### 1. Large Language Models (LLMs)

LLMs are models trained on massive text data that generate responses by predicting the next word in a sequence.

👉 They do NOT “know” facts like humans
👉 They generate responses based on probability

---

### 2. Prompt → Response

The core interaction:

* **Prompt** = User input
* **Completion** = AI-generated output

Example:

```
Prompt: What is Java?
Response: Java is a programming language...
```

---

### 3. Tokens

LLMs process text as **tokens**, not full words.

* "Spring Boot is great" → split into smaller units
* Token count affects:

    * Performance
    * Cost (in cloud APIs)
    * Response limits

---

### 4. Temperature (Creativity Control)

Controls randomness of output:

* Low (0.1) → precise, factual
* High (0.9) → creative, less predictable

---

### 5. Context Window

LLMs only remember what is included in the current request.

👉 No built-in memory
👉 You must send previous conversation manually

---

### 6. Hallucination

AI can generate **incorrect or made-up information**.

👉 It predicts likely text—not verified truth

---

### 7. Embeddings

Text can be converted into vectors (numbers) to represent meaning.

Example:

```
"dog" ≈ "cat" (similar vectors)
```

Used for:

* Search
* Recommendations
* Document retrieval

---

### 8. RAG (Retrieval-Augmented Generation)

Improves AI accuracy by combining:

1. External data (documents, DB)
2. AI model

Flow:

```
User Question → Retrieve Data → Send to AI → Generate Answer
```

---

### 9. Stateless vs Stateful AI

* **Stateless** → each request is independent
* **Stateful** → remembers previous conversation

This project currently uses a **stateless approach**

---

## 🛠️ Tech Stack

* Java
* Spring Boot
* REST API
* Ollama (Local LLM)
* Llama 3 model

---

## ⚙️ Setup Instructions

### 1. Install Ollama

Download from:
https://ollama.com

---

### 2. Pull and run model

```bash
ollama pull llama3
ollama run llama3
```

---

### 3. Run the application

```bash
mvn spring-boot:run
```

---

### 4. Test the API

```
http://localhost:8080/ai/ask?q=Hello
```

---

## 📌 Future Improvements

* Clean JSON response parsing
* Add chat memory (stateful AI)
* Implement RAG (document-based QA)
* Add frontend UI
* Switch between local and cloud models

---

## 💡 Learning Goals

This project demonstrates:

* Backend API design
* AI integration via HTTP
* Clean architecture (Controller/Service)
* Core AI/LLM concepts

---

## 📄 License

MIT License

---

## 🙌 Acknowledgment

Built as part of a learning journey into AI + backend development.
