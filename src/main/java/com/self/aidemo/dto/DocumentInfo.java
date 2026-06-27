package com.self.aidemo.dto;


/**
 * Lightweight response used to expose document metadata
 * without leaking internal database structure.
 */
public record DocumentInfo(
        String documentId,
        String filename,
        String source,
        boolean active
) {}