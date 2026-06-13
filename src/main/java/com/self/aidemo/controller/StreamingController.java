package com.self.aidemo.controller;

import com.self.aidemo.service.StreamingAIService;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

/**
 * REST controller exposing Server-Sent Event (SSE) endpoints
 * for token-by-token AI responses.
 *
 * <p>Unlike the regular controller which returns a complete
 * response, this controller streams generated tokens to the
 * client as soon as they are produced by the model.</p>
 */
@RestController
public class StreamingController {

    private final StreamingAIService streamingAIService;

    /**
     * Creates the streaming controller.
     *
     * @param streamingAIService service responsible for AI streaming
     */
    public StreamingController(StreamingAIService streamingAIService) {
        this.streamingAIService = streamingAIService;
    }

    /**
     * Streams an AI response for a user session.
     *
     * Example:
     *
     * <pre>
     * GET /stream?sessionId=user1&message=Hello
     * </pre>
     *
     * @param sessionId unique session identifier
     * @param message user prompt
     * @return SSE emitter streaming the response
     */
    @GetMapping("/stream")
    public SseEmitter stream(
            @RequestParam String sessionId,
            @RequestParam String message
    ) {

        return streamingAIService.stream(sessionId, message);
    }
}