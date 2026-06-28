package com.self.aidemo.service;

import com.self.aidemo.assistant.AIAssistant;
import com.self.aidemo.dto.DocumentInfo;
import com.self.aidemo.dto.RetrievedChunk;
import com.self.aidemo.entity.UploadedDocument;
import com.self.aidemo.repository.ChatMessageRepository;
import com.self.aidemo.repository.UploadedDocumentRepository;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.ollama.OllamaEmbeddingModel;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import dev.langchain4j.store.embedding.chroma.ChromaEmbeddingStore;
import dev.langchain4j.store.embedding.filter.Filter;
import dev.langchain4j.store.embedding.filter.MetadataFilterBuilder;
import org.springframework.stereotype.Service;

import java.time.LocalDate;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

@Service
public class DocumentService {


    private final OllamaEmbeddingModel embeddingModel;
    private final ChromaEmbeddingStore embeddingStore;


    private String documentContext = "";

    //private final List<String> conversationHistory = new ArrayList<>();


    private final UploadedDocumentRepository uploadedDocumentRepository;



    /**
     * Creates the AI service with required dependencies.
     *
     * @param embeddingModel embedding model used to convert text into vector embeddings
     * @param embeddingStore persistent Chroma vector database used for semantic retrieval

     */
    public DocumentService(
            ChromaEmbeddingStore embeddingStore,
            OllamaEmbeddingModel embeddingModel,
            UploadedDocumentRepository uploadedDocumentRepository) {


        this.embeddingStore = embeddingStore;
        this.embeddingModel = embeddingModel;
        this.uploadedDocumentRepository = uploadedDocumentRepository;
    }
    /**
     * Stores a document in Chroma after splitting it into smaller chunks.
     *
     * @param content raw document text
     */
    public void storeDocument(String content,
                              String filename,
                              String source,
                              String documentId) {

        Document document = Document.from(content);

        UploadedDocument doc = new UploadedDocument();

        doc.setFilename(filename);
        doc.setSource(source);
        doc.setUploadDate(LocalDate.now());
        doc.setActive(true);
        doc.setDocumentId(documentId);
        uploadedDocumentRepository.save(doc);

        List<TextSegment> segments =
                DocumentSplitters.recursive(
                                500,   // chunk size
                                100    // overlap
                        )
                        .split(document);

        Metadata metadata = new Metadata();
        metadata.put("documentId", documentId);
        metadata.put("filename", filename);
        metadata.put("source", source);
        metadata.put("uploadedAt", LocalDate.now().toString());
        for (TextSegment segment : segments) {



            TextSegment segmentWithMetadata =
                    TextSegment.from(segment.text(), metadata);

            var embedding = embeddingModel.embed(segmentWithMetadata).content();



            embeddingStore.add(embedding,segmentWithMetadata);
        }
    }

    public List<UploadedDocument> findAll() {

        return uploadedDocumentRepository.findAll();

    }

    /**
     * Deletes a document and all its associated embeddings.
     *
     * <p>This operation ensures consistency between:
     * <ul>
     *     <li>Relational database (document metadata)</li>
     *     <li>Vector database (Chroma embeddings)</li>
     * </ul>
     *
     * <p>All chunks are filtered using the {@code documentId} metadata field.</p>
     *
     * @param documentId unique document identifier
     */
    public void deleteDocument(String documentId) {

        UploadedDocument doc =
                uploadedDocumentRepository.findByDocumentId(documentId);

        if (doc == null) {
            throw new IllegalArgumentException("Document not found: " + documentId);
        }

        doc.setActive(false);

        uploadedDocumentRepository.save(doc);
    }

    /**
     * Returns all active documents available for RAG retrieval.
     *
     * <p>This is used for UI selection, debugging, and manual deletion workflows.</p>
     *
     * @return list of active document metadata
     */
    public List<DocumentInfo> getActiveDocuments() {

        return uploadedDocumentRepository.findByActiveTrue()
                .stream()
                .map(doc -> new DocumentInfo(
                        doc.getDocumentId(),
                        doc.getFilename(),
                        doc.getSource(),
                        true
                ))
                .toList();
    }


    /**
     * Retrieves the most relevant document chunks for a user query.
     *
     * <p>This method performs semantic search against the vector database
     * without invoking the language model. It is primarily intended for
     * debugging and understanding the Retrieval-Augmented Generation (RAG)
     * pipeline.</p>
     *
     * <p>Each returned chunk contains:
     * <ul>
     *     <li>The source filename</li>
     *     <li>The similarity score</li>
     *     <li>The retrieved text</li>
     * </ul>
     * </p>
     *
     * @param question the user's search query
     * @return list of retrieved document chunks ordered by similarity
     */
    public List<RetrievedChunk> debugRetrieve(String question) {

        var queryEmbedding = embeddingModel.embed(question).content();

        EmbeddingSearchRequest request = EmbeddingSearchRequest.builder()
                .queryEmbedding(queryEmbedding)
                .maxResults(5)
                .minScore(0.7)
                .build();

        EmbeddingSearchResult<TextSegment> result =
                embeddingStore.search(request);

        return result.matches()
                .stream()
                .map(match -> {

                    TextSegment segment = match.embedded();

                    String filename =
                            segment.metadata().getString("filename");

                    return new RetrievedChunk(
                            filename,
                            match.score(),
                            segment.text()
                    );

                })
                .toList();
    }

}
