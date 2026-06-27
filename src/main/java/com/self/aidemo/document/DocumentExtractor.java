package com.self.aidemo.document;


import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

public interface DocumentExtractor {

    /**
     * Returns true if this extractor can process the file.
     */
    boolean supports(String filename);

    /**
     * Extracts plain text from the document.
     */
    String extract(MultipartFile file) throws IOException;

}