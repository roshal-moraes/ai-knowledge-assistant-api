package com.self.aidemo.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

/**
 * Represents the outcome of a single benchmark query.
 *
 * <p>This DTO is used to evaluate retrieval quality and
 * response characteristics across a fixed set of questions.</p>
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class BenchmarkResult {

    /**
     * Benchmark question.
     */
    private String question;

    /**
     * Source documents retrieved for the answer.
     */
    private List<String> sources;

    /**
     * AI-generated answer.
     */
    private String answer;

    /**
     * Time taken to answer in milliseconds.
     */
    private long responseTimeMs;
}