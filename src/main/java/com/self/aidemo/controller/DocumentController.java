package com.self.aidemo.controller;

import com.self.aidemo.service.AIService;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

@RestController
@RequestMapping("/documents")
public class DocumentController {

    private final AIService aiService;

    public DocumentController(AIService aiService) {
        this.aiService = aiService;
    }

    @PostMapping("/upload")
    public String upload(@RequestParam("file") MultipartFile file) throws IOException {
        String content = new String(file.getBytes());
        aiService.storeDocument(content);
        return "Document uploaded successfully.";
    }
}