package com.self.aidemo.dto;

import java.util.List;

/**
 * Represents the complete Retrieval-Augmented Generation (RAG)
 * pipeline for a single user query.
 *
 * <p>This DTO is intended for debugging and educational purposes.
 * It exposes each stage of the RAG workflow, allowing developers
 * to inspect what information was retrieved, the prompt sent to
 * the language model, and the final generated answer.</p>
 *
 * @param question original user question
 * @param retrievedChunks document chunks retrieved from Chroma
 * @param prompt final prompt sent to the language model
 * @param answer generated AI response
 */
public record DebugRagResponse(

        String question,

        List<RetrievedChunk> retrievedChunks,

        String prompt,

        String answer

) {
}