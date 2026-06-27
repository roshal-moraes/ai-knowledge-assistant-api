package com.self.aidemo.dto;

import java.util.List;
import java.util.Objects;

public class AIResponse {

    private String answer;
    private List<String> sources;

    public AIResponse(String answer, List<String> sources) {
        this.answer = answer;
        this.sources = sources;
    }

    @Override
    public String toString() {
        return "AIResponse{" +
                "answer='" + answer + '\'' +
                ", sources=" + sources +
                '}';
    }

    @Override
    public boolean equals(Object o) {
        if (o == null || getClass() != o.getClass()) return false;
        AIResponse that = (AIResponse) o;
        return Objects.equals(answer, that.answer) && Objects.equals(sources, that.sources);
    }

    @Override
    public int hashCode() {
        return Objects.hash(answer, sources);
    }

    public String getAnswer() {
        return answer;
    }

    public void setAnswer(String answer) {
        this.answer = answer;
    }

    public List<String> getSources() {
        return sources;
    }

    public void setSources(List<String> sources) {
        this.sources = sources;
    }

    // getters/setters
}