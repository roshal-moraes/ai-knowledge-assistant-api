package com.self.aidemo.dto;


/**
 * Represents a single document chunk retrieved from the vector database.
 *
 * <p>This DTO is used by the Explainable RAG API to expose
 * which document fragments were retrieved for a user query.
 * It helps developers understand why the AI produced a
 * particular answer.</p>
 *
 * @param filename name of the source document
 * @param score similarity score returned by the vector search
 * @param text retrieved chunk text
 */
public record RetrievedChunk(
        String filename,
        double score,
        String text
) {
}