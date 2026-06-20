package com.self.aidemo.config;

import com.self.aidemo.assistant.AIAssistant;
import com.self.aidemo.assistant.StreamingAssistant;
import com.self.aidemo.tools.TimeTools;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.ChatMemoryProvider;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.ollama.OllamaChatModel;
import dev.langchain4j.model.ollama.OllamaEmbeddingModel;
import dev.langchain4j.model.ollama.OllamaStreamingChatModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.router.DefaultQueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.chroma.ChromaApiVersion;
import dev.langchain4j.store.embedding.chroma.ChromaEmbeddingStore;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;


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
                .collectionName("aidemo2")
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
     * Creates the streaming chat model used for natural language generation.
     *
     * <p>This model generates responses to user prompts and may
     * invoke registered tools when needed.</p>
     *
     * @return configured Ollama chat model
     */
    @Bean
    public StreamingChatModel streamingChatModel() {
        return OllamaStreamingChatModel.builder()
                .baseUrl("http://localhost:11434")
                .modelName("llama3.1")
                .build();
    }


    /**
     *
     * ChatMemory holds current chat messages
     *
     *
     * */
    /*@Bean
    public ChatMemory chatMemory() {
        return MessageWindowChatMemory.withMaxMessages(20);
    }*/

    @Bean
    public ChatMemoryProvider chatMemoryProvider() {

        Map<Object, ChatMemory> memories = new ConcurrentHashMap<>();

        return memoryId ->
                memories.computeIfAbsent(
                        memoryId,
                        id -> MessageWindowChatMemory.withMaxMessages(20)
                );
    }



    /**
     * Creates the streaming AI assistant used for real-time responses.
     *
     * <p>This assistant is separate from the regular {@link AIAssistant}
     * used in previous phases. It uses a {@link StreamingChatModel}
     * so responses can be delivered token-by-token instead of waiting
     * for the full answer.</p>
     *
     * <p>The same session-based memory provider is reused so each user
     * maintains conversation context across requests.</p>
     *
     * @param streamingChatModel configured streaming chat model
     * @param chatMemoryProvider session-based memory provider
     * @return configured streaming assistant
     */
    @Bean
    public StreamingAssistant streamingAssistant(
            StreamingChatModel streamingChatModel,
            ChatMemoryProvider chatMemoryProvider
    ) {

        return AiServices.builder(StreamingAssistant.class)
                .streamingChatModel(streamingChatModel)
                .chatMemoryProvider(chatMemoryProvider)
                .build();
    }


    /**
     * Creates the content retriever used by the RAG pipeline.
     *
     * <p>The retriever converts the user query into an embedding,
     * searches Chroma for similar document chunks, and returns
     * the most relevant segments.</p>
     *
     * @param embeddingStore Chroma vector database
     * @param embeddingModel embedding model used for query vectors
     * @return configured content retriever
     */
    @Bean
    public ContentRetriever contentRetriever(
            ChromaEmbeddingStore embeddingStore,
            OllamaEmbeddingModel embeddingModel
    ) {

        return EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(3)
                .minScore(0.1)
                .build();
    }

    /**
     * Creates the RetrievalAugmentor used by RAG.
     *
     * <p>The augmentor retrieves relevant document chunks from the
     * vector database and injects them into the prompt before sending
     * the request to the LLM.</p>
     *
     * @param contentRetriever retriever responsible for semantic search
     * @return configured retrieval augmentor
     */
    @Bean
    public RetrievalAugmentor retrievalAugmentor(
            ContentRetriever contentRetriever
    ) {

        return DefaultRetrievalAugmentor.builder()
                .queryRouter(new DefaultQueryRouter(contentRetriever))
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
     * @return configured AI assistant
     */

    /*@Bean
    public AIAssistant aiAssistant(
            OllamaChatModel chatModel,
//            // @param timeTools available AI-callable tools
            TimeTools timeTools
    ) {
        return AiServices.builder(AIAssistant.class)
                .chatModel(chatModel)
                .tools(timeTools)
                .build();
    }*/
    @Bean
    public AIAssistant aiAssistant(
            ChatModel chatModel,
            ChatMemoryProvider chatMemoryProvider,
            RetrievalAugmentor retrievalAugmentor
    ) {
        return AiServices.builder(AIAssistant.class)
                .chatModel(chatModel)
                .chatMemoryProvider(chatMemoryProvider)
                .retrievalAugmentor(retrievalAugmentor)
                .build();
    }

}