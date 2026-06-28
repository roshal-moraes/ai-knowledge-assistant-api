package com.self.aidemo.controller;

import com.self.aidemo.document.DocumentExtractionService;
import com.self.aidemo.dto.DocumentInfo;
import com.self.aidemo.dto.RetrievedChunk;
import com.self.aidemo.entity.UploadedDocument;
import com.self.aidemo.service.DocumentService;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.List;
import java.util.UUID;

/**
 * REST controller responsible for document ingestion.
 *
 * <p>This controller accepts uploaded documents and delegates text
 * extraction to the {@link DocumentExtractionService}. Once the document
 * has been converted into plain text, it is passed to the AI pipeline
 * for chunking, embedding generation, and storage in the vector database.</p>
 *
 * <p>The controller is intentionally unaware of individual document
 * formats (PDF, TXT, DOCX, etc.). New document types can be supported
 * without modifying this controller.</p>
 */
@RestController
@RequestMapping("/documents")
public class DocumentController {

    /**
     * Service responsible for storing processed documents.
     */
    private final DocumentService documentService;

    /**
     * Service responsible for converting uploaded documents into plain text.
     */
    private final DocumentExtractionService extractionService;

    /**
     * Creates the document controller.
     *
     * @param documentService   service responsible for document ingestion
     * @param extractionService service responsible for document text extraction
     */
    public DocumentController(
            DocumentService documentService,
            DocumentExtractionService extractionService
    ) {
        this.documentService = documentService;
        this.extractionService = extractionService;
    }

    /**
     * Uploads a document into the AI knowledge base.
     *
     * <p>The uploaded file is first converted into plain text by the
     * {@link DocumentExtractionService}. The resulting text is then
     * chunked, embedded, and stored in the vector database.</p>
     *
     * @param file uploaded document
     * @return confirmation message
     * @throws IOException if the document cannot be read
     */
    @PostMapping("/upload")
    public String upload(@RequestParam("file") MultipartFile file)
            throws IOException {

        String content = extractionService.extract(file);
        String documentId = UUID.randomUUID().toString();

        documentService.storeDocument(
                content,
                file.getOriginalFilename(),
                "upload", documentId
        );

        return "Document uploaded successfully.";
    }

    /**
     * Returns all uploaded documents recorded by the application.
     *
     * @return list of uploaded documents
     */
    @GetMapping
    public List<UploadedDocument> getDocuments() {
        return documentService.findAll();
    }


    /**
     * Returns all active documents available in the system.
     *
     * <p>This endpoint is useful for:
     * <ul>
     *     <li>Viewing uploaded documents</li>
     *     <li>Selecting documents for deletion</li>
     *     <li>Debugging RAG ingestion</li>
     * </ul>
     * </p>
     *
     * @return list of active document metadata
     */
    @GetMapping("/active")
    public List<DocumentInfo> getActiveDocuments() {
        return documentService.getActiveDocuments();
    }

    /**
     * Soft deletes a document from the system.
     *
     * <p>This endpoint does NOT remove embeddings from Chroma.
     * Instead, it marks the document as inactive in the database,
     * so it is excluded from future AI retrieval results.</p>
     *
     * @param documentId unique identifier of the document to delete
     * @return confirmation message
     */
    @DeleteMapping("/{documentId}")
    public String deleteDocument(@PathVariable String documentId) {

        documentService.deleteDocument(documentId);

        return "Document soft deleted successfully: " + documentId;
    }

    /**
     * Retrieves the document chunks that are most relevant to a query.
     *
     * <p>This endpoint is intended for debugging the Retrieval-Augmented
     * Generation (RAG) pipeline. It performs semantic search against the
     * vector database and returns the retrieved chunks without generating
     * an AI response.</p>
     *
     * <p>This helps developers understand:
     * <ul>
     *     <li>Which documents were retrieved</li>
     *     <li>The similarity score of each match</li>
     *     <li>The exact text sent to the language model</li>
     * </ul>
     * </p>
     *
     * @param question the search query
     * @return list of retrieved document chunks
     */
    @GetMapping("/debug/retrieve")
    public List<RetrievedChunk> debugRetrieve(
            @RequestParam("q") String question
    ) {
        return documentService.debugRetrieve(question);
    }
}