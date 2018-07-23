package net.kaustubh.questiongenerator.transducer;

/**
 * Accessors for question and answer,
 *
 * @author kaustubhdholé.
 */
public interface IQaPair {

    String question();

    void question(String question);

    String answer();

    void answer(String answer);
}
