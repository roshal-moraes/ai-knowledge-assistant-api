package com.self.aidemo.document;

import org.apache.pdfbox.Loader;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.text.PDFTextStripper;
import org.springframework.stereotype.Component;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

/**
 * Extracts plain text from PDF documents.
 *
 * <p>This extractor uses Apache PDFBox to read the uploaded PDF and
 * convert its contents into plain text. The extracted text is then
 * returned to the AI ingestion pipeline for chunking, embedding, and
 * storage in the vector database.</p>
 *
 * <p>This class is responsible only for text extraction. It does not
 * perform any AI-related processing such as chunking, embedding
 * generation, or vector storage.</p>
 */
@Component
public class PdfDocumentExtractor implements DocumentExtractor {

    /**
     * Determines whether this extractor supports the uploaded file.
     *
     * @param filename uploaded file name
     * @return {@code true} if the file is a PDF; otherwise {@code false}
     */
    @Override
    public boolean supports(String filename) {

        return filename != null
                && filename.toLowerCase().endsWith(".pdf");
    }

    /**
     * Extracts plain text from a PDF document.
     *
     * @param file uploaded PDF document
     * @return extracted text
     * @throws IOException if the PDF cannot be read
     */
    @Override
    public String extract(MultipartFile file) throws IOException {

        try (PDDocument document = Loader.loadPDF(file.getBytes())) {

            PDFTextStripper stripper = new PDFTextStripper();

            return stripper.getText(document);
        }
    }
}