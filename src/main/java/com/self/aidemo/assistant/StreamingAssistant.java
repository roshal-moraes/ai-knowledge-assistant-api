package com.self.aidemo.assistant;

import dev.langchain4j.service.MemoryId;
import dev.langchain4j.service.TokenStream;
import dev.langchain4j.service.UserMessage;

/**
 * Marker interface for the streaming AI pipeline.
 *
 * <p>This assistant will be used in Phase 10 to support
 * token-by-token streaming responses using Server-Sent Events (SSE).</p>
 *
 * <p>Keeping a separate interface avoids affecting the
 * existing non-streaming assistant introduced in earlier phases.</p>
 */
public interface StreamingAssistant {

    /**
     * Streams an AI response for the specified user session.
     *
     * <p>The session identifier allows each user to maintain
     * independent conversation memory.</p>
     *
     * @param memoryId unique session identifier
     * @param message user message
     * @return token stream containing the generated response
     */
    TokenStream chat(
            @MemoryId String memoryId,
            @UserMessage String message
    );
}
