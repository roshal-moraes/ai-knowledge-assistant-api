package com.self.aidemo.service;

import dev.langchain4j.model.chat.ChatModel;
import org.springframework.stereotype.Service;

/**
 * Expands search queries to improve retrieval recall.
 *
 * <p>This helps when queries are short, ambiguous, or acronym-based
 * (e.g. "DI", "IOC", "spring mvc").</p>
 */
@Service
public class QueryExpander {

    private final ChatModel chatModel;

    public QueryExpander(ChatModel chatModel) {
        this.chatModel = chatModel;
    }

    /**
     * Expands a standalone query into a richer search query.
     *
     * @param rewrittenQuestion output from QuestionRewriter
     * @return expanded query for embedding search
     */
    public String expand(String rewrittenQuestion) {

        String prompt = """
                You are a search query expansion engine.

                Expand the query into a more detailed version
                that improves semantic search.

                Rules:
                - Do NOT answer the question
                - Add relevant technical terms
                - Include synonyms and related concepts
                - Keep it concise (1 sentence)

                Query:
                %s
                """.formatted(rewrittenQuestion);

        return chatModel.chat(prompt);
    }
}