package com.self.aidemo.controller;

import com.self.aidemo.service.AIService;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
/**
 * REST controller responsible for document ingestion.
 *
 * <p>This controller accepts uploaded documents and passes their
 * content to the AI service for chunking, embedding generation,
 * and storage in the vector database.</p>
 *
 * <p>Uploaded documents become searchable context for future
 * retrieval-augmented AI responses.</p>
 */
@RestController
@RequestMapping("/documents")
public class DocumentController {

    private final AIService aiService;


    public DocumentController(AIService aiService) {
        this.aiService = aiService;
    }

    /**
     * Uploads a document and stores its contents for semantic retrieval.
     *
     * <p>The document text is processed into chunks, converted into
     * vector embeddings, and saved to the persistent vector store.</p>
     *
     * @param file the uploaded document file
     * @return confirmation message
     */
    @PostMapping("/upload")
    public String upload(@RequestParam("file") MultipartFile file) throws IOException {
        String content = new String(file.getBytes());
        aiService.storeDocument(content);
        return "Document uploaded successfully.";
    }
}