package com.self.aidemo.document;


import org.springframework.stereotype.Component;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.nio.charset.StandardCharsets;

@Component
public class TxtDocumentExtractor implements DocumentExtractor {

    @Override
    public boolean supports(String filename) {

        return filename != null &&
                filename.toLowerCase().endsWith(".txt");
    }

    @Override
    public String extract(MultipartFile file) throws IOException {

        return new String(file.getBytes(), StandardCharsets.UTF_8);

    }
}