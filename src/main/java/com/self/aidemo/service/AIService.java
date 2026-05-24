package com.self.aidemo.service;


import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.model.ollama.OllamaEmbeddingModel;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;

import dev.langchain4j.store.embedding.chroma.ChromaEmbeddingStore;

import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.data.segment.TextSegment;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;


@Service
public class AIService {

    private final OllamaEmbeddingModel embeddingModel;
    private final ChromaEmbeddingStore embeddingStore;

    private String documentContext = "";

    private final List<String> conversationHistory = new ArrayList<>();

    private final String API_KEY = System.getenv("GEMINI_API_KEY");

    public AIService(
            OllamaEmbeddingModel embeddingModel,
            ChromaEmbeddingStore embeddingStore
    ) {
        this.embeddingModel = embeddingModel;
        this.embeddingStore = embeddingStore;
    }




    public String ask(String question) {

        //String url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=" + API_KEY;
        String url = "http://localhost:11434/api/generate";
        RestTemplate restTemplate = new RestTemplate();

        conversationHistory.add("User: " + question);
        if (conversationHistory.size() > 10) {
            conversationHistory.remove(0);
        }
        var queryEmbedding = embeddingModel.embed(question).content();

        EmbeddingSearchRequest searchRequest = EmbeddingSearchRequest.builder()
                .queryEmbedding(queryEmbedding)
                .maxResults(10)
                .build();

        EmbeddingSearchResult<TextSegment> result =
                embeddingStore.search(searchRequest);

        String retrievedContext = result.matches().stream()
                .map(match -> match.embedded().text())
                .reduce("", (a, b) -> a + "\n" + b);

        String fullPrompt =
                "Use this document to answer the question:\n"
                        + retrievedContext
                        + "\n\nConversation:\n"
                        + String.join("\n", conversationHistory)
                        + "\nAI:";

        Map<String, Object> ollamaRequest = Map.of(
                "model", "phi3",
                "prompt", "You are a helpful Java backend tutor. Answer clearly:\n" + fullPrompt,
                "stream", false,
                "options", Map.of(
                        "num_predict", 300
                )
        );


        try {

            Map response = restTemplate.postForObject(url, ollamaRequest, Map.class);
            String answer = (String) response.get("response");
            //System.out.println(response.toString());
            conversationHistory.add("AI: " + answer);

            if (conversationHistory.size() > 10) {
                conversationHistory.remove(0);
            }


            return answer;
        } catch (Exception e) {
            return "Error: Unable to get response from AI";
        }
    }

    /*public void storeDocument(String content) {
        this.documentContext = content;
    }*/

    public void storeDocument(String content) {

        List<String> chunks = List.of(content.split("\\."));

        for (String chunk : chunks) {

            chunk = chunk.trim();

            if (chunk.isBlank()) {
                continue;
            }

            TextSegment segment = TextSegment.from(chunk);

            var embedding = embeddingModel.embed(segment).content();

            embeddingStore.add(embedding, segment);
        }
    }


}