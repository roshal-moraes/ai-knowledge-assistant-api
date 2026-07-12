package com.self.aidemo.tools;


import com.self.aidemo.entity.UploadedDocument;
import com.self.aidemo.service.DocumentService;
import dev.langchain4j.agent.tool.P;
import dev.langchain4j.agent.tool.Tool;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.stream.Collectors;

/**
 * AI tools for interacting with uploaded documents.
 *
 * <p>These methods expose business functionality that the LLM
 * can invoke automatically when answering user requests.</p>
 */
@Component
public class DocumentTools {

    private final DocumentService documentService;

    public DocumentTools(DocumentService documentService) {
        this.documentService = documentService;
    }

    /**
     * Lists all active uploaded documents.
     *
     * @return formatted document list
     */
    @Tool("List all uploaded documents")
    public String listDocuments() {

        List<UploadedDocument> documents =
                documentService.findAll();

        if (documents.isEmpty()) {
            return "No uploaded documents were found.";
        }

        return documents.stream()
                .map(doc ->
                        String.format(
                                "ID=%d, Filename=%s",
                                doc.getDocumentId(),
                                doc.getFilename()))
                .collect(Collectors.joining("\n"));
    }

    /**
     * Returns the number of uploaded documents.
     *
     * @return document count
     */
    @Tool("Count uploaded documents")
    public int countDocuments() {
        return documentService.findAll().size();
    }

    /**
     * Returns details for a document.
     *
     * @param documentId document identifier
     * @return document information
     */
    @Tool("Get document details by ID")
    public String getDocumentById(
            @P("Document ID") long documentId
    ) {

        return documentService.findAll().stream()
                .filter(d -> d.getDocumentId().equals(documentId))
                .findFirst()
                .map(doc ->
                        """
                        Document
                        --------
                        ID: %d
                        Filename: %s
                        Source: %s
                        Uploaded: %s
                        Active: %s
                        """.formatted(
                                doc.getDocumentId(),
                                doc.getFilename(),
                                doc.getSource(),
                                doc.getUploadDate(),
                                doc.isActive()))
                .orElse("Document not found.");
    }
}