package com.self.aidemo.service;


import com.self.aidemo.assistant.AIAssistant;
import com.self.aidemo.dto.AIResponse;
import com.self.aidemo.dto.DebugRagResponse;
import com.self.aidemo.dto.RetrievedChunk;
import com.self.aidemo.entity.ChatMessage;
import com.self.aidemo.entity.UploadedDocument;
import com.self.aidemo.repository.ChatMessageRepository;

import com.self.aidemo.repository.UploadedDocumentRepository;
import dev.langchain4j.model.ollama.OllamaEmbeddingModel;
import dev.langchain4j.rag.content.Content;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.query.Query;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;

import dev.langchain4j.store.embedding.chroma.ChromaEmbeddingStore;

import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.data.segment.TextSegment;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Core AI service responsible for:
 * <ul>
 *     <li>Managing conversation history (short-term memory)</li>
 *     <li>Generating embeddings for semantic search</li>
 *     <li>Retrieving relevant document context from the vector store</li>
 *     <li>Constructing prompts for the language model</li>
 *     <li>Sending prompts either manually to Ollama or via LangChain4j AI Services</li>
 * </ul>
 *
 * <p>This service demonstrates two approaches to LLM integration:</p>
 * <ol>
 *     <li><b>Manual Ollama API invocation</b> using RestTemplate</li>
 *     <li><b>Framework-managed AI interaction</b> using LangChain4j AiServices with tool calling</li>
 * </ol>
 *
 * <p>It also supports Retrieval-Augmented Generation (RAG) by storing and
 * searching document embeddings in a Chroma vector database.</p>
 */

@Service
public class AIService {

    private final OllamaEmbeddingModel embeddingModel;
    private final ChromaEmbeddingStore embeddingStore;
    private final AIAssistant assistant;

    private String documentContext = "";

    //private final List<String> conversationHistory = new ArrayList<>();

    private final ChatMessageRepository chatRepository;

    private final String API_KEY = System.getenv("GEMINI_API_KEY");

    private final ContentRetriever contentRetriever;
    private UploadedDocumentRepository uploadedDocumentRepository;

    private final DocumentService documentService;

    /**
     * Creates the AI service with required dependencies.
     *
     * @param embeddingModel   embedding model used to convert text into vector embeddings
     * @param embeddingStore   persistent Chroma vector database used for semantic retrieval
     * @param assistant        LangChain4j AI assistant capable of tool calling
     * @param contentRetriever Create a dedicated retrieval method
     */
    public AIService(
            AIAssistant assistant,
            ChromaEmbeddingStore embeddingStore,
            OllamaEmbeddingModel embeddingModel,
            ContentRetriever contentRetriever,
            ChatMessageRepository chatRepository,
            UploadedDocumentRepository uploadedDocumentRepository,
            DocumentService documentService) {

        this.assistant = assistant;
        this.embeddingStore = embeddingStore;
        this.embeddingModel = embeddingModel;
        this.contentRetriever = contentRetriever;
        this.chatRepository = chatRepository;
        this.uploadedDocumentRepository = uploadedDocumentRepository;
        this.documentService = documentService;
    }


    /**
     * Answers a user question using manual HTTP communication with the Ollama API.
     *
     * <p>Workflow:</p>
     * <ol>
     *     <li>Add user question to conversation history</li>
     *     <li>Generate embedding for the question</li>
     *     <li>Search vector store for relevant document chunks</li>
     *     <li>Construct a prompt containing retrieved context and chat history</li>
     *     <li>Send the prompt directly to Ollama via REST API</li>
     *     <li>Store AI response in conversation history</li>
     * </ol>
     *
     * <p>This method exists mainly for educational purposes to demonstrate
     * low-level LLM integration without framework abstractions.</p>
     *
     * @param question the user's input question
     * @return AI-generated response, or an error message if the request fails
     */
    public String askWithManualOllama(String question) {

        //String url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=" + API_KEY;
        String url = "http://localhost:11434/api/generate";
        RestTemplate restTemplate = new RestTemplate();

        chatRepository.save(
                new ChatMessage("User", question)
        );
        /*if (conversationHistory.size() > 10) {
            conversationHistory.remove(0);
        }*/

        String conversation = chatRepository.findAll().stream()
                .map(msg -> msg.getSender() + ": " + msg.getMessage())
                .reduce("", (a, b) -> a + "\n" + b);
        var queryEmbedding = embeddingModel.embed(question).content();

        EmbeddingSearchRequest searchRequest = EmbeddingSearchRequest.builder()
                .queryEmbedding(queryEmbedding)
                .maxResults(10)
                .build();

        EmbeddingSearchResult<TextSegment> result =
                embeddingStore.search(searchRequest);

        String retrievedContext = result.matches().stream()
                .map(match -> match.embedded().text())
                .reduce("", (a, b) -> a + "\n" + b);

        String fullPrompt =
                "Use this document to answer the question:\n"
                        + retrievedContext
                        + "\n\nConversation:\n"
                        + conversation
                        + "\nAI:";

        Map<String, Object> ollamaRequest = Map.of(
                "model", "phi3",
                "prompt", "You are a helpful Java backend tutor. Answer clearly:\n" + fullPrompt,
                "stream", false,
                "options", Map.of(
                        "num_predict", 300
                )
        );


        try {

            Map response = restTemplate.postForObject(url, ollamaRequest, Map.class);
            String answer = (String) response.get("response");
            //System.out.println(response.toString());
            chatRepository.save(
                    new ChatMessage("AI", answer)
            );

            /*if (conversationHistory.size() > 10) {
                conversationHistory.remove(0);
            }*/


            return answer;
        } catch (Exception e) {
            return "Error: Unable to get response from AI";
        }
    }

    /**
     * Answers a user question using LangChain4j AI Services.
     *
     * <p>Workflow:</p>
     * <ol>
     *     <li>Add user question to conversation history</li>
     *     <li>Generate embedding for semantic retrieval</li>
     *     <li>Retrieve relevant document chunks from Chroma</li>
     *     <li>Build an augmented prompt with context and memory</li>
     *     <li>Send the prompt through the AI assistant</li>
     *     <li>Allow the model to invoke tools if needed</li>
     *     <li>Store the AI response in conversation history</li>
     * </ol>
     *
     * <p>This is the preferred production-style method because it enables
     * tool calling and cleaner abstraction through LangChain4j.</p>
     *
     * @param question the user's input question
     * @return AI-generated response
     */
    public AIResponse ask(String sessionId, String question) {

        // Optional: Persist conversation for history/auditing
        chatRepository.save(new ChatMessage("User", question));

        // Let LangChain4j handle:
        // - Chat memory
        // - Retrieval
        // - Prompt augmentation
        String answer = assistant.chat(sessionId, question);

        chatRepository.save(new ChatMessage("AI", answer));

        List<String> sources = getSources(question);

        return new AIResponse(answer, sources);
    }

    public List<String> getSources(String question) {

        Query query = Query.from(question);

        List<Content> contents = contentRetriever.retrieve(query)
                .stream()
                .filter(content -> {

                    TextSegment segment = (TextSegment) content.textSegment();

                    String documentId =
                            segment.metadata().getString("documentId");

                    UploadedDocument doc =
                            uploadedDocumentRepository.findByDocumentId(documentId);

                    return doc != null && doc.isActive();

                })
                .toList();

        System.out.println("========== RETRIEVED SEGMENTS ==========");

        contents.forEach(content -> {
            TextSegment segment = (TextSegment) content.textSegment();

            System.out.println("----------------------------------------");
            System.out.println("Filename : " + segment.metadata().getString("filename"));
            System.out.println(segment.text());
        });

        return contents.stream()
                .map(content -> ((TextSegment) content.textSegment())
                        .metadata()
                        .getString("filename"))
                .distinct()
                .toList();
    }

    /**
     * Executes the complete Retrieval-Augmented Generation (RAG) pipeline
     * and returns every intermediate step for debugging purposes.
     *
     * <p>This method is intended for development and learning. It exposes
     * the retrieved document chunks, the constructed prompt, and the final
     * language model response.</p>
     *
     * @param sessionId unique chat session identifier
     * @param question  user's question
     * @return complete RAG execution trace
     */
    public DebugRagResponse debugAsk(String sessionId, String question) {

        // Step 1: Retrieve relevant chunks
        List<RetrievedChunk> retrievedChunks =
                documentService.debugRetrieve(question);

        // Step 2: Build context string
        String context = retrievedChunks.stream()
                .map(chunk ->
                        """
                                Document: %s
                                Similarity Score: %.3f
                                
                                %s
                                -----------------------------
                                """.formatted(
                                chunk.filename(),
                                chunk.score(),
                                chunk.text()
                        ))
                .collect(Collectors.joining("\n"));

        // Step 3: Build prompt
        String prompt =
                """
                        You are a helpful Java backend tutor.
                        
                        Answer ONLY using the retrieved context below.
                        
                        If the answer cannot be found in the context,
                        clearly say so.
                        
                        Context:
                        
                        %s
                        
                        Question:
                        %s
                        """.formatted(context, question);

        // Step 4: Generate answer
        String answer = assistant.chat(sessionId, prompt);

        // Step 5: Return complete pipeline
        return new DebugRagResponse(
                question,
                retrievedChunks,
                prompt,
                answer
        );
    }


}