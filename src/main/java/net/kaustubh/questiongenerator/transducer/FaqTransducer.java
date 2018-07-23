package net.kaustubh.questiongenerator.transducer;

import net.kaustubh.questiongenerator.ark.AnalysisUtilities;
import net.kaustubh.questiongenerator.ark.GlobalProperties;
import net.kaustubh.questiongenerator.ark.InitialTransformationStep;
import net.kaustubh.questiongenerator.ark.Question;
import net.kaustubh.questiongenerator.ark.QuestionTransducer;

import java.util.ArrayList;
import java.util.List;

import edu.stanford.nlp.trees.Tree;

/**
 * FaqTransducer implementation based on
 * "Question Generation via Overgenerating
 * Transformations and Ranking" by Michael Heilman and Noah A. Smith
 * for generating question answer pairs for a particular fact.
 *
 * @author kaustubhdholé.
 */
public class FaqTransducer implements IFaqTransducer {

    static final boolean justWH = false;
    private QuestionTransducer questionTransducer;
    private InitialTransformationStep transformer;
    private ConstituencyParser constituencyParser;

    public FaqTransducer() {
        constituencyParser = new ConstituencyParser();
        questionTransducer = new QuestionTransducer();
        transformer = new InitialTransformationStep();
        questionTransducer.setAvoidPronounsAndDemonstratives(false);
        String propertiesPath = FaqTransducer.class.getClassLoader().getResource("config/QuestionTransducer.properties").getPath();
        GlobalProperties.loadProperties(propertiesPath);
        transformer.setDoPronounNPC(true);
        transformer.setDoNonPronounNPC(true);
    }

    public static void main(String[] args) {
        IFaqTransducer faqTransducer = new FaqTransducer();
        List<QaPair> qaPairs = faqTransducer.transduceFact("The man ate the mango.");
        System.out.println();
    }

    /**
     * The input sentence should be tokenised.
     */
    public List<QaPair> transduceFact(String sentence) {
        List<Question> questionList = new ArrayList<Question>();
        Integer maxLength = 30;

        List<Tree> inputTrees = new ArrayList<Tree>();

        Tree parsed = constituencyParser.parse(sentence);
        inputTrees.add(parsed);

        List<Question> transformationOutput = transformer.transform(inputTrees);

        for (Question question : transformationOutput) {
            questionTransducer.generateQuestionsFromParse(question);
            questionList.addAll(questionTransducer.getQuestions());
        }

        QuestionTransducer.removeDuplicateQuestions(questionList);
        List<String> quests = new ArrayList<String>();
        String booleanQuestion = "";
        for (Question question : questionList) {

            if (question.getTree().getLeaves().size() > maxLength) {
                continue;
            }

            if (justWH && question.getFeatureValue("whQuestion") != 1.0) {
                continue;
            }

            Tree ansTree = question.getAnswerPhraseTree();
            if (ansTree != null) {
                quests.add(question.yield() + "\t" + AnalysisUtilities.getCleanedUpYield(question.getAnswerPhraseTree()));
            } else {
                booleanQuestion = question.yield();
                quests.add(booleanQuestion);
            }
        }

        //return quests;
        return new ArrayList<QaPair>();
    }
}
