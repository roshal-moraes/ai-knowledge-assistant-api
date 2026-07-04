package com.self.aidemo.controller;

import com.self.aidemo.dto.AIResponse;
import com.self.aidemo.dto.BenchmarkResult;
import com.self.aidemo.dto.DebugRagResponse;
import com.self.aidemo.service.AIService;
import com.self.aidemo.service.BenchmarkService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;

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

    private final BenchmarkService benchmarkService;
    @Autowired
    public AIController(AIService aiService,  BenchmarkService benchmarkService) {

        this.aiService = aiService;
        this.benchmarkService = benchmarkService;
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
    public AIResponse ask(@RequestParam String sessionId,@RequestParam String q) {

        AIResponse response =  aiService.ask(sessionId, q);
        System.out.println(response);
        return response;

    }

    /**
     * Executes the complete Retrieval-Augmented Generation (RAG)
     * pipeline and returns every intermediate step.
     *
     * <p>This endpoint is intended for debugging and educational
     * purposes. It exposes:
     *
     * <ul>
     *     <li>User question</li>
     *     <li>Retrieved document chunks</li>
     *     <li>Constructed prompt</li>
     *     <li>Final AI response</li>
     * </ul>
     *
     * This endpoint makes the RAG pipeline fully explainable.
     *
     * @param sessionId unique chat session
     * @param question user question
     * @return complete RAG execution trace
     */
    @GetMapping("/debug")
    public DebugRagResponse debugAsk(
            @RequestParam String sessionId,
            @RequestParam("q") String question
    ) {
        return aiService.debugAsk(sessionId, question);
    }

    /**
     * Executes the predefined RAG benchmark suite.
     *
     * @return benchmark results
     */
    @GetMapping("/debug/benchmark")
    public List<BenchmarkResult> benchmark() {
        return benchmarkService.runBenchmark();
    }
}