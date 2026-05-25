package com.self.aidemo.config;

import com.self.aidemo.assistant.AIAssistant;
import com.self.aidemo.tools.TimeTools;
import dev.langchain4j.model.ollama.OllamaChatModel;
import dev.langchain4j.model.ollama.OllamaEmbeddingModel;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.chroma.ChromaApiVersion;
import dev.langchain4j.store.embedding.chroma.ChromaEmbeddingStore;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;


/**
 * Spring configuration class for AI-related beans.
 *
 * <p>This class defines and wires together the core AI infrastructure
 * used by the application, including:</p>
 *
 * <ul>
 *     <li>Embedding model for generating vector embeddings</li>
 *     <li>Persistent Chroma vector store for semantic retrieval</li>
 *     <li>Chat model for natural language responses</li>
 *     <li>AI assistant abstraction with tool-calling support</li>
 * </ul>
 *
 * <p>Using Spring dependency injection allows these components to be
 * shared and reused across the application.</p>
 */
@Configuration
public class AIConfig {
    /**
     * Creates the embedding model used to convert text into vector embeddings.
     *
     * <p>Embeddings are used for semantic search in the RAG pipeline,
     * allowing the application to retrieve relevant document context
     * based on meaning rather than exact keyword matches.</p>
     *
     * @return configured Ollama embedding model
     */
    @Bean
    public OllamaEmbeddingModel embeddingModel() {
        return OllamaEmbeddingModel.builder()
                .baseUrl("http://localhost:11434")
                .modelName("nomic-embed-text")
                .build();
    }

    /**
     * Creates the persistent Chroma vector database connection.
     *
     * <p>This embedding store holds document embeddings and supports
     * similarity search for retrieval-augmented generation (RAG).</p>
     *
     * <p>Unlike in-memory storage, Chroma persists data across
     * application restarts.</p>
     *
     * @return configured Chroma embedding store
     */
    @Bean
    public ChromaEmbeddingStore chromaEmbeddingStore() {
        return ChromaEmbeddingStore.builder()
                .baseUrl("http://localhost:8000")
                .apiVersion(ChromaApiVersion.V2)
                .tenantName("default_tenant")
                .databaseName("default_database")
                .collectionName("aidemo")
                .build();
    }

    /**
     * Creates the chat model used for natural language generation.
     *
     * <p>This model generates responses to user prompts and may
     * invoke registered tools when needed.</p>
     *
     * @return configured Ollama chat model
     */
    @Bean
    public OllamaChatModel chatModel() {
        return OllamaChatModel.builder()
                .baseUrl("http://localhost:11434")
                .modelName("llama3.1")
                .build();
    }

    /**
     * Creates the high-level AI assistant using LangChain4j AiServices.
     *
     * <p>The assistant combines:</p>
     * <ul>
     *     <li>The chat model for language generation</li>
     *     <li>Registered tool classes for function calling</li>
     * </ul>
     *
     * <p>LangChain4j automatically generates an implementation of the
     * {@code AIAssistant} interface and manages tool invocation.</p>
     *
     * @param chatModel configured LLM chat model
     * @param timeTools available AI-callable tools
     * @return configured AI assistant
     */

    @Bean
    public AIAssistant aiAssistant(
            OllamaChatModel chatModel,
            TimeTools timeTools
    ) {
        return AiServices.builder(AIAssistant.class)
                .chatModel(chatModel)
                .tools(timeTools)
                .build();
    }
}