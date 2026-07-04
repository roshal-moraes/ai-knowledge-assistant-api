package com.self.aidemo.service;

import com.self.aidemo.dto.AIResponse;
import com.self.aidemo.dto.BenchmarkResult;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

/**
 * Executes a predefined set of benchmark questions against the RAG system.
 *
 * <p>This service provides a repeatable way to evaluate retrieval quality,
 * answer quality, and response time after making changes to the RAG pipeline.</p>
 */
@Service
public class BenchmarkService {

    private final AIService aiService;

    public BenchmarkService(AIService aiService) {
        this.aiService = aiService;
    }

    /**
     * Executes all benchmark questions.
     *
     * @return benchmark results
     */
    public List<BenchmarkResult> runBenchmark() {

        List<String> questions = List.of(
                "What is Dependency Injection?",
                "What is Constructor Injection?",
                "What is LocalDateTime?"
        );

        List<BenchmarkResult> results = new ArrayList<>();

        String sessionId = "benchmark";

        for (String question : questions) {

            long start = System.currentTimeMillis();

            AIResponse response =
                    aiService.ask(sessionId, question);

            long elapsed = System.currentTimeMillis() - start;

            results.add(new BenchmarkResult(
                    question,
                    response.getSources(),
                    response.getAnswer(),
                    elapsed
            ));
        }

        return results;
    }
}