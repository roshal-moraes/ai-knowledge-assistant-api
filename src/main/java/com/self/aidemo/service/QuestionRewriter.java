package com.self.aidemo.service;

import dev.langchain4j.model.chat.ChatModel;
import org.springframework.stereotype.Service;

/**
 * Rewrites user questions into standalone search queries
 * using conversation context.
 */
@Service
public class QuestionRewriter {

    private final ChatModel chatModel;

    public QuestionRewriter(ChatModel chatModel) {
        this.chatModel = chatModel;
    }

    /**
     * Rewrite question using conversation context.
     *
     * @param conversation last N messages from session
     * @param question latest user question
     * @return standalone search query
     */
    public String rewrite(String conversation, String question) {

        String prompt = """
                You are an expert search query rewriter.

                Use the conversation to understand context.

                Rewrite ONLY the latest question into a standalone search query.

                Do NOT answer it.

                Return ONLY the rewritten question.

                Conversation:
                %s

                Question:
                %s
                """.formatted(conversation, question);

        return chatModel.chat(prompt);
    }
}