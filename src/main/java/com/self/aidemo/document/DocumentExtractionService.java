package com.self.aidemo.document;

import com.self.aidemo.document.DocumentExtractor;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.List;

/**
 * Coordinates document text extraction by delegating to the appropriate
 * {@link DocumentExtractor} implementation.
 *
 * <p>This service is responsible for selecting an extractor based on the
 * uploaded file's type. It does not perform AI-related tasks such as
 * chunking, embedding generation, or vector storage. Its sole responsibility
 * is converting supported document formats into plain text.</p>
 *
 * <p>Adding support for a new document type only requires creating another
 * {@code DocumentExtractor} implementation. This service automatically
 * discovers all extractor beans through Spring dependency injection.</p>
 */
@Service
public class DocumentExtractionService {

    private final List<DocumentExtractor> extractors;

    /**
     * Creates the document extraction service with all available extractors.
     *
     * @param extractors all {@link DocumentExtractor} implementations
     *                   discovered by Spring
     */
    public DocumentExtractionService(List<DocumentExtractor> extractors) {
        this.extractors = extractors;
    }

    /**
     * Extracts plain text from the uploaded document.
     *
     * <p>The service iterates through all registered extractors until it finds
     * one that supports the uploaded file. The extracted text is then returned
     * for further processing by the AI pipeline.</p>
     *
     * @param file the uploaded document
     * @return the extracted plain text
     * @throws IOException if the document cannot be read
     * @throws IllegalArgumentException if no extractor supports the file type
     */
    public String extract(MultipartFile file) throws IOException {

        String filename = file.getOriginalFilename();

        for (DocumentExtractor extractor : extractors) {

            if (extractor.supports(filename)) {
                return extractor.extract(file);
            }
        }

        throw new IllegalArgumentException(
                "Unsupported document type: " + filename
        );
    }
}