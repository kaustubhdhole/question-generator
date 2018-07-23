package net.kaustubh.questiongenerator.transducer;

/**
 * A pair of question and answer (FAQ).
 *
 * @author kaustubhdholé.
 */
public class QaPair implements IQaPair{

    private String question;

    private String answer;

    public QaPair(String question, String answer) {
        this.question = question;
        this.answer = answer;
    }

    public String question() {
        return question;
    }

    public void question(String question) {
        this.question = question;
    }

    public String answer() {
        return answer;
    }

    public void answer(String answer) {
        this.answer = answer;
    }

}
