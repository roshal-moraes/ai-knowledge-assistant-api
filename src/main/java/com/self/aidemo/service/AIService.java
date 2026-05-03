package com.self.aidemo.service;


import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.Map;


@Service
public class AIService {

    private final String API_KEY = System.getenv("GEMINI_API_KEY");

    public String ask(String question) {

        //String url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=" + API_KEY;
        String url = "http://localhost:11434/api/generate";
        RestTemplate restTemplate = new RestTemplate();

        Map<String, Object> request = Map.of(
                "model", "llama3",
                "prompt", "You are a helpful Java backend tutor. Answer clearly:\n" + question,
                "stream", false
        );


        try {
            Map response = restTemplate.postForObject(url, request, Map.class);
            return (String) response.get("response");
        } catch (Exception e) {
            return "Error: Unable to get response from AI";
        }
    }
}