package com.self.aidemo.assistant;

import dev.langchain4j.service.MemoryId;
import dev.langchain4j.service.UserMessage;

/**
 * High-level AI assistant abstraction used by LangChain4j AiServices.
 *
 * <p>This interface defines the contract for interacting with the language model.
 * At runtime, LangChain4j automatically generates an implementation that connects
 * the configured chat model and any registered tools.</p>
 *
 * <p>This abstraction separates application business logic from direct model
 * communication, making the AI interaction cleaner and easier to maintain.</p>
 */
public interface AIAssistant {

    /**
     * Sends a prompt to the AI assistant and returns the generated response.
     *
     * <p>The assistant may answer directly or invoke available tools
     * before generating the final response.</p>
     *
     * @param message the prompt or user message to send to the AI
     * @return the AI-generated response
     */
    String chat(@MemoryId String sessionId,
                @UserMessage String message);
    //String chat(String message);

}