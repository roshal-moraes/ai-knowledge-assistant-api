package com.self.aidemo.controller;

import com.self.aidemo.dto.AIResponse;
import com.self.aidemo.service.AIService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/ai")
public class AIController {

    private final AIService aiService;

    @Autowired
    public AIController(AIService aiService) {

        this.aiService = aiService;
    }

    @GetMapping("/ask")
    public AIResponse ask(@RequestParam String q) {

        String result = aiService.ask(q);
        return new AIResponse(result);
    }
}