package net.kaustubh.supersensetagger.input;

import lombok.Data;
import lombok.experimental.Accessors;

/**
 * @author kaustubhdholé.
 */
@Data
@Accessors(fluent = true)
public class Word {

    String word;

    String pos;

    String stem;

    public Word(String word, String pos) {
        this.word = word;
        this.stem = stem;
    }

    public boolean hasStem() {
        return stem != null;
    }
}
