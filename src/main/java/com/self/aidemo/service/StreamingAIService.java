package com.self.aidemo.service;

import org.springframework.stereotype.Service;

import com.self.aidemo.assistant.StreamingAssistant;
import dev.langchain4j.service.TokenStream;
import org.springframework.stereotype.Service;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

/**
 * Service responsible for streaming AI responses to clients.
 *
 * <p>This service bridges LangChain4j's {@link TokenStream}
 * with Spring's {@link SseEmitter}.</p>
 *
 * <p>Responses are delivered token-by-token, allowing clients
 * to display text progressively instead of waiting for the
 * entire answer.</p>
 */
@Service
public class StreamingAIService {

    private final StreamingAssistant streamingAssistant;

    /**
     * Creates the streaming AI service.
     *
     * @param streamingAssistant streaming AI assistant
     */
    public StreamingAIService(StreamingAssistant streamingAssistant) {
        this.streamingAssistant = streamingAssistant;
    }

    /**
     * Starts streaming an AI response for a session.
     *
     * @param sessionId unique session identifier
     * @param message user message
     * @return SSE emitter used to push tokens to the client
     */
    public SseEmitter stream(String sessionId, String message) {

        SseEmitter emitter = new SseEmitter();

        TokenStream tokenStream =
                streamingAssistant.chat(sessionId, message);

        tokenStream
                .onPartialResponse(token -> {
                    try {
                        emitter.send(token);
                    } catch (Exception e) {
                        emitter.completeWithError(e);
                    }
                })
                .onCompleteResponse(response -> emitter.complete())
                .onError(emitter::completeWithError)
                .start();

        return emitter;
    }
}