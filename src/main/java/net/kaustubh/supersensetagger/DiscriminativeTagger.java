package net.kaustubh.supersensetagger;


import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Random;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

import lombok.Getter;
import lombok.Setter;


public class DiscriminativeTagger implements Serializable{

    @Getter
    @Setter
    private int maxIters = 5;
    private List<LabeledSentence> trainingData;
    private List<LabeledSentence> testData;

    @Getter
    private double [] finalWeights;

    /*
     * feature weights are stored in an array of size equal to the number
     * of features times the number of labels
     *
     *  this map goes from feature names (keys) to feature indexes WITHOUT offsets.
     *  the offsets are equal to the label index times the number of features
     *
     */
    private Map<String, Integer> featureIndexes;
    @Getter
    @Setter
    private List<String> labels;
    private int lastFeatureIndex = 0;
    @Getter
    @Setter
    private String savePrefix = null;

    private double [][] dpValues;
    private int [][] dpBackPointers;
    private Random rgen;
    @Setter
    private boolean developmentMode;
    private static Properties properties;
    private boolean perceptron = false;
    private boolean useBIconstraintInDecoding = true;

    private static final long serialVersionUID = 7096385301991299782L;

    public DiscriminativeTagger(){
        featureIndexes = new HashMap<String, Integer>();
        trainingData = null;
        labels = new ArrayList<String>();
        rgen = new Random(1234567);
    }

    /**
     * Load a list of the possible labels.  This must be done before training
     * so that the feature vector has the appropriate dimensions
     *
     * @param labelFile
     * @return
     */
    public static List<String> loadLabelList(String labelFile){
        List<String> res = new ArrayList<String>();
        try {
            BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(labelFile)));
            String buf;
            while((buf = br.readLine())!= null){
                res.add(buf);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return res;
    }


    /**
     * Load the BIO tagged supersense data from Semcor, as provided in the SuperSenseTagger release (SEM_07.BI).
     * We also use their POS labels, which presumably were what their paper used.
     * One difference is that this method expects the data to be converted into a 3-column format
     * with an extra newline between each sentence (as in CoNLL data), which can be created
     * from the SST data with a short perl script.
     *
     * @param path
     * @param labels
     * @return
     */
    public static List<LabeledSentence> loadSuperSenseData(String path, List<String> labels){
        List<LabeledSentence> res = new ArrayList<LabeledSentence>();

        try {
            BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(path)));
            String buf;
            String [] parts;
            LabeledSentence sent = new LabeledSentence();
            while((buf = br.readLine())!= null){
                if(buf.length()==0){
                    if(sent.length()>0){
                        res.add(sent);
                        sent = new LabeledSentence();
                    }
                    continue;
                }
                parts = buf.split("\\t");
                String label = removeExtraLabels(parts[2], labels);
                sent.addToken(parts[0], SuperSenseFeatureExtractor.getInstance().getStem(parts[0], parts[1]), parts[1], label);
                if(parts.length > 3 && !parts[3].equals("")){
                    sent.setArticleID(parts[3]);
                }
            }

            if(sent.length()>0) res.add(sent);

        } catch (IOException e) {
            e.printStackTrace();
        }


        return res;
    }


    /**
     * remove labels for adjectives and adverbs, which the SST does not address
     * because they are lumped together in wordnet
     *
     * @param label
     * @param labels
     * @return
     */
    public static String removeExtraLabels(String label, List<String> labels) {
		/*if(label.contains("-adj.") || label.contains("-adv.") || label.endsWith(".other")){
			return "0";
		}*/
        if(!labels.contains(label)){
            return "0";
        }
        return label;
    }

    /**
     * serialize model, clearing out unneeded data first
     * (and then resetting it so it can be used in subsequent iterations if necessary)
     *
     * @param savePath
     */
    private void saveModel(String savePath) {
        List<LabeledSentence> tmpTrainingData = trainingData;
        List<LabeledSentence> tmpTestData = testData;
        String tmpSavePrefix = savePrefix;
        try {
            ObjectOutputStream out = new ObjectOutputStream(new GZIPOutputStream(new FileOutputStream(savePath)));
            trainingData = null;
            testData = null;
            savePrefix = null;
            out.writeObject(this);
            out.flush();
            out.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        trainingData = tmpTrainingData;
        testData = tmpTestData;
        savePrefix = tmpSavePrefix;
    }


    /**
     *
     * load a serialized model
     *
     * @param loadPath
     * @return
     */
    public static DiscriminativeTagger loadModel(String loadPath){
       DiscriminativeTagger res = null;
        try {
            ObjectInputStream in = new ObjectInputStream(new GZIPInputStream(new FileInputStream(loadPath)));
            res = (DiscriminativeTagger) in.readObject();

        } catch (Exception e) {
            e.printStackTrace();
        }

        return res;
    }

    /**
     * train the model using the averaged perceptron
     * (or perhaps MIRA in the future, but that doesn't currently work)
     * See Collins paper on Discriminative HMMs.
     *
     */
    public void train(){
        if(trainingData == null){
            System.err.println("training data not set.");
            return;
        }
        if(perceptron) System.err.println("training with the perceptron.");
        else System.err.println("training with 1-best MIRA.");

        createDPTables();
        createFeatures();

        double[] intermediateWeights = new double[labels.size()*featureIndexes.size()];

        long numWordsProcessed = 0;
        long numWordsIncorrect=0;
        long totalInstancesProcessed = 0;

        long trainingDataSize = trainingData.size();

        int numIters = 0;
        for(numIters=0;numIters<maxIters;numIters++){
            System.err.println("iter="+numIters);
            Collections.shuffle(trainingData, rgen);
            for(int i=0; i<trainingData.size(); i++){
                LabeledSentence sent = trainingData.get(i);


                if(perceptron){
                    findBestLabelSequenceViterbi(sent, intermediateWeights);
                    perceptronUpdate(sent, intermediateWeights);
                }else{
                    findBestLabelSequenceViterbi(sent, intermediateWeights, true);
                    MIRAUpdate(sent, intermediateWeights);
                }

                for(int j=0; j<sent.length(); j++){
                    if(!sent.getTrueLabels().get(j).equals(sent.getPredictions().get(j))){
                        numWordsIncorrect++;
                    }
                }
                numWordsProcessed+=sent.length();
                totalInstancesProcessed++;

                for(int f=0;f<finalWeights.length;f++){
                    finalWeights[f] += intermediateWeights[f];
                }

                if(totalInstancesProcessed % 500 == 0){
                    System.err.println("totalInstancesProcessed="+totalInstancesProcessed);
                    System.err.println("pct. correct words in last 500 inst.:"+NumberFormat.getInstance().format((double)(numWordsProcessed-numWordsIncorrect)/numWordsProcessed));
                    numWordsIncorrect=0; numWordsProcessed=0;
                }


            }

        }

        //average the weights for the "averaged" part of the averaged perceptron
        double normalizer = (double)maxIters * trainingDataSize;
        multiplyByScalar(finalWeights, 1.0/normalizer);
        if(savePrefix != null) saveModel(savePrefix);

    }

    private void multiplyByScalar(double[] weights, double scalar) {
        for(int i=0;i<weights.length; i++){
            weights[i] *= scalar;
        }
    }



    /**
     * helper method for perceptron training.
     * basically, update weights by adding the feature vector for the correct label
     * and subtracting the feature vector for the predicted label
     *
     * this method breaks down the process so that the changes are made token by token
     *
     * @param sent
     * @param intermediateWeights
     */
    private void perceptronUpdate(LabeledSentence sent, double[] intermediateWeights) {
        if(sent.predictionsAreCorrect()){
            return;
        }

        for(int j=0; j<sent.length(); j++){
            int predictedLabelIndex = labels.indexOf(sent.getPredictions().get(j));
            int correctLabelIndex = labels.indexOf(sent.getTrueLabels().get(j));
            if(correctLabelIndex == predictedLabelIndex){
                continue;
            }

            Map<String, Double> featureValuesPredicted = SuperSenseFeatureExtractor.getInstance().extractFeatureValues(sent, j);
            Map<String, Double> featureValuesGold = SuperSenseFeatureExtractor.getInstance().extractFeatureValues(sent, j, false);

            int indexOffsetForCorrectLabel = correctLabelIndex*featureIndexes.size();
            int indexOffsetForPredictedLabel = predictedLabelIndex*featureIndexes.size();
            int featureIndex;

            for(String key: featureValuesGold.keySet()){
                featureIndex = featureIndexes.get(key);
                intermediateWeights[featureIndex+indexOffsetForCorrectLabel] += featureValuesGold.get(key);
            }
            for(String key: featureValuesPredicted.keySet()){
                featureIndex = featureIndexes.get(key);
                intermediateWeights[featureIndex+indexOffsetForPredictedLabel] -= featureValuesPredicted.get(key);
            }

        }
    }



    private void addToMap(Map<String, Double> mapToAddTo, Map<String, Double> map, boolean doSubtraction){
        for(String key: map.keySet()){
            Double val1 = mapToAddTo.get(key);
            Double val2 = map.get(key);
            if(val1 == null){
                val1 = 0.0;
            }

            if(doSubtraction){
                val1 += val2;
            }else{
                val1 -= val2;
            }
            mapToAddTo.put(key, val1);
        }
    }

    private void addToMap(Map<String, Double> mapToAddTo, Map<String, Double> map){
        addToMap(mapToAddTo, map, false);
    }


    private void subtractFromMap(Map<String, Double> mapToAddTo, Map<String, Double> map){
        addToMap(mapToAddTo, map, true);
    }


    /**
     * 1-best MIRA, using Andre and Kevin's paper
     *
     * @param sent
     * @param intermediateWeights
     */
    private void MIRAUpdate(LabeledSentence sent, double[] intermediateWeights) {
        if(sent.predictionsAreCorrect()){
            return;
        }

        //compute x L2 norm for denominator in MIRA update
        //compute number of incorrect labels (hamming loss)
        double C = 1.0;
        double x2 = 0.0;
        Map<String, Double> featureValuesPredicted;
        Map<String, Double> featureValuesGold;
        Map<String, Double> featureValuesDifferences = new HashMap<String,Double>();
        Map<String,Double> tmpFeatureValues;

        //compute the step size from looking at the whole sentence
        double numWrong=0;
        double scoreGold = 0.0;
        double scorePredicted = 0.0;
        int predictedLabelIndex;
        int correctLabelIndex;
        for(int j=0; j<sent.length(); j++){
            predictedLabelIndex = labels.indexOf(sent.getPredictions().get(j));
            tmpFeatureValues = SuperSenseFeatureExtractor.getInstance().extractFeatureValues(sent, j);
            scorePredicted += computeScore(tmpFeatureValues,intermediateWeights, predictedLabelIndex);
            addToMap(featureValuesDifferences, tmpFeatureValues);

            correctLabelIndex = labels.indexOf(sent.getTrueLabels().get(j));
            tmpFeatureValues = SuperSenseFeatureExtractor.getInstance().extractFeatureValues(sent, j, false);
            scoreGold += computeScore(tmpFeatureValues,intermediateWeights,  correctLabelIndex);
            subtractFromMap(featureValuesDifferences, tmpFeatureValues);

            if(predictedLabelIndex != correctLabelIndex){
                numWrong+=1.0;
            }
        }

        for(Double val: featureValuesDifferences.values()){
            x2 += val*val;
        }

        double scoreDifference = scorePredicted - scoreGold;
        double update = Math.min(1.0/C, (scoreDifference + numWrong ) /x2);

        //Now update the features for each word.
        //It is done this way, rather than a single update for the sentence,
        //which should be equivalent,
        //due to implementation tricks used for extracting features.
        for(int j=0; j<sent.length(); j++){
            predictedLabelIndex = labels.indexOf(sent.getPredictions().get(j));
            correctLabelIndex = labels.indexOf(sent.getTrueLabels().get(j));
            int indexOffsetForCorrectLabel = correctLabelIndex*featureIndexes.size();
            int indexOffsetForPredictedLabel = predictedLabelIndex*featureIndexes.size();
            int featureIndex;

            featureValuesPredicted = SuperSenseFeatureExtractor.getInstance().extractFeatureValues(sent, j);
            featureValuesGold = SuperSenseFeatureExtractor.getInstance().extractFeatureValues(sent, j, false);

            for(String key: featureValuesGold.keySet()){
                featureIndex = featureIndexes.get(key);
                intermediateWeights[featureIndex+indexOffsetForCorrectLabel] += update * featureValuesGold.get(key);
            }
            for(String key: featureValuesPredicted.keySet()){
                featureIndex = featureIndexes.get(key);
                intermediateWeights[featureIndex+indexOffsetForPredictedLabel] -= update * featureValuesPredicted.get(key);
            }

        }


    }

    /**
     * initialize dynamic programming tables
     * used by the viterbi algorithm
     *
     */
    private void createDPTables() {

        int maxNumTokens = 0;

		/*
		for(LabeledSentence sent: trainingData){
			if(sent.length()>maxNumTokens){
				maxNumTokens = sent.length();
			}
		}
		*/

        maxNumTokens = 200;
        dpValues = new double[maxNumTokens][labels.size()];
        dpBackPointers = new int[maxNumTokens][labels.size()];
    }


    /**
     * compute a dot product of a set of feature values and the corresponding weights.
     * This involves looking up the appropriate indexes into the weight vector.
     *
     * @param featureValues
     * @param weights
     * @return
     */
    private double computeScore(Map<String, Double> featureValues, double[] weights, int labelIndex) {
        double res = 0.0;


        if(labelIndex==-1){
            return 0.0;
        }

        double featureValue;
        double featureWeight;
        int indexOffsetForLabel = labelIndex*featureIndexes.size();


        for(String key: featureValues.keySet()){
            featureValue = featureValues.get(key);
            Integer index = featureIndexes.get(key);
            if(index != null){ //test set features may not have been instantiated from the training data
                featureWeight = weights[index+indexOffsetForLabel];
                res += featureWeight*featureValue;
            }
        }

        return res;
    }


    /**
     * This is an efficiency trick so that the whole feature vector need not be
     * computed in each inner loop of the viterbi algorith.
     * Since only one feature of a token depends on the previous label, we can compute
     * all the others first, and then add this one when needed.
     * This can reduce computation by a factor equal to the number of labels, roughly (83 for supersenses!)
     *
     *
     * @param prevLabel
     * @param curLabelIndex
     * @param weights
     * @return
     */
    private double previousLabelFeatureScore(String prevLabel, int curLabelIndex, double[] weights) {
        int indexOffsetForLabel = curLabelIndex*featureIndexes.size();
        return weights[featureIndexes.get("prevLabel="+prevLabel)+indexOffsetForLabel]; // * 1.0, ommitted for brevity
    }



    /**
     * before training, loop through the training data
     * once to instantiate all the possible features,
     * so we don't have to worry about null in the HashMaps
     *
     */
    private void createFeatures(){
        lastFeatureIndex = 0;

        //instantiate previous label features for all possible previous labels
        for(String label: labels){
            String key = "prevLabel="+label;
            if(!featureIndexes.containsKey(key)){
                featureIndexes.put(key, lastFeatureIndex++);
            }
        }

        //instantiate the rest of the features
        for(LabeledSentence sent: trainingData){
            for(int i=0; i<sent.length(); i++){
                if(i>0) sent.getPredictions().set(i-1, sent.getTrueLabels().get(i-1));
                Map<String, Double> values = SuperSenseFeatureExtractor.getInstance().extractFeatureValues(sent, i, false);

                for(String key: values.keySet()){
                    if(!featureIndexes.containsKey(key)){
                        featureIndexes.put(key, lastFeatureIndex++);
                    }
                }
            }
        }

        //now create the array of feature weights
        finalWeights = new double[labels.size()*featureIndexes.size()];

    }

    public static Properties getProperties(){
        if(properties == null){
            loadProperties("tagger.properties");
        }
        return properties;
    }

    public static Properties loadProperties(String propertiesFile){
        properties = new Properties();
        try{
            properties.load(new FileInputStream(propertiesFile));
        }catch(Exception e){
            e.printStackTrace();
            System.exit(0);
        }

        return properties;
    }


    public void findBestLabelSequenceViterbi(LabeledSentence sent, double [] weights){
        findBestLabelSequenceViterbi(sent, weights, false);
    }


    /**
     * uses the Viterbi algorithm to find the current best sequence
     * of labels for a sentence, given the weight vector.
     * used in both training and testing
     *
     * @param sent
     * @param weights
     * @param includeLossTerm whether to perform loss augmented inference (e.g., with MIRA)
     */
    public void findBestLabelSequenceViterbi(LabeledSentence sent, double [] weights, boolean includeLossTerm){
        int numTokens = sent.length();

        if(dpValues.length < numTokens){ //expand the size of the dynamic programming tables if necessary
            dpValues = new double[(int)(numTokens*1.5)][labels.size()];
            dpBackPointers = new int[(int)(numTokens*1.5)][labels.size()];
        }

        double maxScore;
        double score;
        int maxIndex = -1;
        String label;
        Map<String, Double> featureValues;
        double tmpScore;
        String prevLabel;

        for(int i=0;i<numTokens; i++){
            sent.getPredictions().set(i, null);
        }

        //for each token
        for(int i=0;i<numTokens; i++){
            featureValues = SuperSenseFeatureExtractor.getInstance().extractFeatureValues(sent, i);
            //String stem = sent.getStems().get(i);
            //String tok = sent.getTokens().get(i);
            //String pos  = sent.getPosLabels().get(i);

            //for each current label
            for(int j=0;j<labels.size();j++){
                maxIndex = -1;
                maxScore = Double.NEGATIVE_INFINITY;
                tmpScore = computeScore(featureValues, weights, j);
                label = labels.get(j);
                if(includeLossTerm && !label.equals(sent.getTrueLabels().get(i))){
                    tmpScore += 1.0;
                }

                //skip if this sense is not possible according to wordnet
                //if no possible senses are listed, allow any (e.g., for names)
				/*
				Set<String> possibleSenses = SuperSenseFeatureExtractor.getInstance().getPossibleSenses(stem);
				Set<String> tmp = SuperSenseFeatureExtractor.getInstance().getPossibleSenses(tok);
				if(tmp != null && possibleSenses != null) possibleSenses.addAll(tmp);
				if(possibleSenses!=null && !label.equals("0")
						&& !possibleSenses.contains(label)
						&& !pos.matches("^NNPS?")){
					dpValues[i][j] = maxScore;
					dpBackPointers[i][j] = maxIndex;
					if(sent.getTrueLabels().get(i).equals(label)){
						continue;
					}
					continue;
				}
				*/

                //consider each possible previous label
                for(int k=0; k<labels.size();k++){
                    prevLabel = labels.get(k);
                    if(useBIconstraintInDecoding && label.charAt(0) == 'I'){
                        if(prevLabel.equals("0") || i==0){
                            continue;
                        }
                        if(!prevLabel.substring(2).equals(label.substring(2))){//assume tags are formatted B-class1, I-class1, etc.
                            continue;
                        }
                    }

                    //	compute current score based on previous scores
                    score = 0.0;
                    if(i>0){
                        sent.getPredictions().set(i-1, labels.get(k));
                        score = dpValues[i-1][k];
                    }

                    //the score for the previous label is added on separately here,
                    //in order to avoid computing the whole score, which only depends
                    //on the previous label for one feature,
                    //a large number of times: O(labels*labels).
                    //TODO plus versus times doesn't matter here, right?  Use plus because of numeric overflow
                    score += tmpScore + previousLabelFeatureScore(labels.get(k), j, weights);


                    //find the max of the combined score
                    //use that to choose what to output for the previous label
                    if(score>maxScore){
                        maxScore = score;
                        maxIndex = k;
                    }

                    //if this is the first token, we don't need to iterate over all possible previous labels,
                    //because there is only one possibility (i.e., null)
                    if(i==0){
                        break;
                    }
                }

                dpValues[i][j] = maxScore;
                dpBackPointers[i][j] = maxIndex;
            }
        }

        //extract predictions from backpointers
        maxIndex = -1;
        maxScore = Double.NEGATIVE_INFINITY;
        //first, find the best label for the last token
        for(int j=0; j<labels.size(); j++){
            score = dpValues[numTokens-1][j];
            if(score > maxScore){
                maxScore = score;
                maxIndex = j;
            }
        }
        //now iterate backwards by following backpointers
        for(int i=numTokens-1;i>=0;i--){
            sent.getPredictions().set(i,labels.get(maxIndex));
            maxIndex = dpBackPointers[i][maxIndex];
        }

    }

}
