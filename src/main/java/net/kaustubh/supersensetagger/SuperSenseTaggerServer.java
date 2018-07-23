package net.kaustubh.supersensetagger;

import net.kaustubh.supersensetagger.input.Sentence;
import net.kaustubh.supersensetagger.input.Word;

/**
 * @author kaustubhdholé.
 */
public class SuperSenseTaggerServer {

    private DiscriminativeTagger tagger;

    public SuperSenseTaggerServer(String propertiesFile, String modelFile) {
        DiscriminativeTagger.loadProperties(propertiesFile);
        tagger = DiscriminativeTagger.loadModel(modelFile);
    }

    public static void main(String[] args) {
        SuperSenseTaggerServer
                superSenseTaggerServer = new SuperSenseTaggerServer("tagger.properties", "models/superSenseModelAllSemcor.ser.gz");
        Sentence sample = getSampleSentence();
        LabeledSentence labeledSentence = superSenseTaggerServer.tag(sample);
        System.out.println();
    }

    private static Sentence getSampleSentence() {
        return new Sentence()
                .addWord(new Word("The", "DT"))
                .addWord(new Word("man", "NN"))
                .addWord(new Word("is", "VBZ"))
                .addWord(new Word("in", "IN"))
                .addWord(new Word("the", "DT"))
                .addWord(new Word("jungle", "NN"))
                .addWord(new Word(".", "."));
    }

    public LabeledSentence tag(Sentence sentence) {
        LabeledSentence labeledSentence = new LabeledSentence();
        for (Word word : sentence.words()) {
            if (word.hasStem()) {
                labeledSentence.addToken(word.word(), word.stem(), word.pos(), "0");
            } else {
                String stem = SuperSenseFeatureExtractor.getInstance().getStem(word.word(), word.pos());
                labeledSentence.addToken(word.word(), stem, word.pos(), "0");
            }
        }
        tagger.findBestLabelSequenceViterbi(labeledSentence, tagger.getFinalWeights());
        return labeledSentence;
    }

}
