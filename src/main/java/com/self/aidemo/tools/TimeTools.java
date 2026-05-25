package com.self.aidemo.tools;

import dev.langchain4j.agent.tool.Tool;
import org.springframework.stereotype.Component;

import java.time.LocalDateTime;

/**
 * Collection of utility tools exposed to the AI model.
 *
 * <p>Methods annotated with {@code @Tool} can be invoked dynamically
 * by the language model through LangChain4j's tool-calling mechanism.</p>
 *
 * <p>This enables the AI assistant to perform actions or retrieve
 * real-time information instead of relying only on its built-in knowledge.</p>
 */
@Component
public class TimeTools {

    /**
     * Returns the current system date and time.
     *
     * <p>This method can be called by the AI when a user asks
     * time-related questions such as "What time is it?"</p>
     *
     * @return the current system timestamp as a string
     */
    @Tool("Get the current system time")
    public String getCurrentTime() {
        return LocalDateTime.now().toString();
    }
}