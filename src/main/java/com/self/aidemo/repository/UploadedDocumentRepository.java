package com.self.aidemo.repository;

import com.self.aidemo.entity.UploadedDocument;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface UploadedDocumentRepository
        extends JpaRepository<UploadedDocument, Long> {
    void deleteByDocumentId(String documentId);

    UploadedDocument findByDocumentId(String documentId);

    List<UploadedDocument> findByActiveTrue();
}