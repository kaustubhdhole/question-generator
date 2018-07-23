package net.kaustubh.questiongenerator.transducer;

import java.util.List;

/**
 * FaqTransducer for generating question answer pairs for a particular fact.
 *
 * @author kaustubhdholé.
 */
public interface IFaqTransducer {

    /**
     * Generating question answer pairs for a particular fact.
     */
    List<QaPair> transduceFact(String sentence);

}
