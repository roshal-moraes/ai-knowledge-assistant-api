package com.self.aidemo.service;


import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;


@Service
public class AIService {
    private final List<String> conversationHistory = new ArrayList<>();

    private final String API_KEY = System.getenv("GEMINI_API_KEY");

    public String ask(String question) {

        //String url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=" + API_KEY;
        String url = "http://localhost:11434/api/generate";
        RestTemplate restTemplate = new RestTemplate();

        conversationHistory.add("User: " + question);
        if (conversationHistory.size() > 10) {
            conversationHistory.remove(0);
        }
        String fullPrompt = String.join("\n", conversationHistory) + "\nAI:";

        Map<String, Object> request = Map.of(
                "model", "llama3",
                "prompt", "You are a helpful Java backend tutor. Answer clearly:\n" + fullPrompt,
                "stream", false
        );


        try {

            Map response = restTemplate.postForObject(url, request, Map.class);
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
}