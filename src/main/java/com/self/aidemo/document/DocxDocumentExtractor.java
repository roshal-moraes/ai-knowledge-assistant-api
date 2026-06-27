package com.self.aidemo.document;

import org.apache.poi.xwpf.usermodel.XWPFDocument;
import org.apache.poi.xwpf.extractor.XWPFWordExtractor;
import org.springframework.stereotype.Component;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

/**
 * Extracts plain text from Microsoft Word (.docx) documents.
 *
 * <p>This extractor uses Apache POI to read the contents of a DOCX file
 * and convert it into plain text. The extracted text is then returned
 * to the AI ingestion pipeline, where it will be chunked, embedded,
 * and stored in the vector database.</p>
 *
 * <p>This class is responsible only for document text extraction.
 * It deliberately does not perform chunking, embedding generation,
 * metadata creation, or vector storage.</p>
 */
@Component
public class DocxDocumentExtractor implements DocumentExtractor {

    /**
     * Determines whether this extractor supports the uploaded file.
     *
     * @param filename uploaded file name
     * @return {@code true} if the file has a .docx extension
     */
    @Override
    public boolean supports(String filename) {

        return filename != null
                && filename.toLowerCase().endsWith(".docx");
    }

    /**
     * Extracts plain text from a Microsoft Word document.
     *
     * @param file uploaded DOCX file
     * @return extracted document text
     * @throws IOException if the document cannot be read
     */
    @Override
    public String extract(MultipartFile file) throws IOException {

        try (XWPFDocument document =
                     new XWPFDocument(file.getInputStream());

             XWPFWordExtractor extractor =
                     new XWPFWordExtractor(document)) {

            return extractor.getText();
        }
    }
}