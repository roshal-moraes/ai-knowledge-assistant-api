package com.self.aidemo.memory;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import org.springframework.stereotype.Component;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * Stores chat memory for each user session.
 *
 * <p>This component centralizes session memory management.
 * It is used by both the AI assistant and other services
 * (such as the QuestionRewriter) that need access to the
 * recent conversation history.</p>
 *
 * <p>Each session receives its own {@link ChatMemory}
 * instance backed by a fixed-size message window.</p>
 */
@Component
public class SessionMemoryStore {

    private final Map<Object, ChatMemory> memories =
            new ConcurrentHashMap<>();

    /**
     * Returns the chat memory associated with a session.
     *
     * <p>If no memory exists yet, one is created automatically.</p>
     *
     * @param sessionId unique session identifier
     * @return session chat memory
     */
    public ChatMemory get(Object sessionId) {

        return memories.computeIfAbsent(
                sessionId,
                id -> MessageWindowChatMemory.withMaxMessages(20)
        );
    }


    /**
     * Returns recent conversation formatted as text for LLM prompts.
     */
    public String getConversationContext(
            Object sessionId,
            int maxMessages
    ) {

        ChatMemory memory = get(sessionId);

        return memory.messages().stream()
                .skip(Math.max(0, memory.messages().size() - maxMessages))
                .map(message -> {

                    if (message instanceof UserMessage m) {
                        return "User: " + m.singleText();
                    }

                    if (message instanceof AiMessage m) {
                        return "Assistant: " + m.text();
                    }

                    if (message instanceof SystemMessage m) {
                        return "System: " + m.text();
                    }

                    return "Unknown: " + message.toString();
                })
                .collect(Collectors.joining("\n"));
    }

}