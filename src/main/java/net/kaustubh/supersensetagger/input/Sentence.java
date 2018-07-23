package net.kaustubh.supersensetagger.input;

import java.util.ArrayList;
import java.util.List;

import lombok.Data;
import lombok.experimental.Accessors;

/**
 * @author kaustubhdholé.
 */
@Data
@Accessors(fluent = true)
public class Sentence {

    private List<Word> words;

    public Sentence() {
        words = new ArrayList<Word>();
    }

    public Sentence addWord(Word word){
        words.add(word);
        return this;
    }
}
