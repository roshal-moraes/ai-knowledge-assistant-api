package com.self.aidemo.service;

import com.self.aidemo.entity.UploadedDocument;
import com.self.aidemo.repository.UploadedDocumentRepository;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

/**
 * Performs simple keyword-based document retrieval.
 *
 * <p>This complements semantic vector search by matching
 * exact terms such as class names, annotations,
 * exception names, or method names.</p>
 */
@Service
public class KeywordRetriever {

    private final UploadedDocumentRepository repository;

    public KeywordRetriever(UploadedDocumentRepository repository) {
        this.repository = repository;
    }

    /**
     * Finds documents whose filename contains the query.
     *
     * <p>This is the first version. Later we'll search document
     * contents and rank matches.</p>
     *
     * @param query user search query
     * @return matching documents
     */
    public List<UploadedDocument> search(String query) {

        List<UploadedDocument> matches = new ArrayList<>();

        String lower = query.toLowerCase();

        for (UploadedDocument doc : repository.findByActiveTrue()) {

            if (doc.getFilename().toLowerCase().contains(lower)) {
                matches.add(doc);
            }
        }

        return matches;
    }
}