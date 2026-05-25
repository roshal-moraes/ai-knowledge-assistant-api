package com.self.aidemo.controller;

import com.self.aidemo.dto.AIResponse;
import com.self.aidemo.service.AIService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

/**
 * REST controller exposing AI chat endpoints.
 *
 * <p>This controller receives user questions and delegates processing
 * to {@code AIService}, which handles retrieval-augmented generation (RAG),
 * conversation memory, and AI response generation.</p>
 *
 * <p>It serves as the main entry point for interacting with the AI assistant.</p>
 */
@RestController
@RequestMapping("/ai")
public class AIController {

    private final AIService aiService;

    @Autowired
    public AIController(AIService aiService) {

        this.aiService = aiService;
    }


    /**
     * Processes a user question and returns an AI-generated response.
     *
     * <p>The request is forwarded to the AI service, which may:</p>
     * <ul>
     *     <li>Retrieve relevant document context from the vector store</li>
     *     <li>Include conversation history</li>
     *     <li>Generate an answer using the language model</li>
     *     <li>Invoke tools if needed</li>
     * </ul>
     *
     * @param q the user's question
     * @return AI-generated answer
     */
    @GetMapping("/ask")
    public AIResponse ask(@RequestParam String q) {

        String result = aiService.ask(q);
        return new AIResponse(result);
    }
}