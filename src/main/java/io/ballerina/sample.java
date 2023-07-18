//package io.ballerina;
//
//import ai.djl.huggingface.tokenizers.Encoding;
//import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
//import ai.djl.modality.nlp.preprocess.Tokenizer;
//import ai.onnxruntime.*;
//import ai.onnxruntime.OnnxTensor;
//import com.genesys.roberta.tokenizer.RobertaTokenizer;
//import com.genesys.roberta.tokenizer.RobertaTokenizerResources;
//
//import java.io.IOException;
//import java.nio.file.Paths;
//import java.util.HashMap;
//import java.util.List;
//import java.util.Map;
//import java.util.Optional;
//
//public class sample {
//
//    public static void main(String[] args){
//        String sentence = "string <mask> = getCountries();";
//        sample predictor = new sample();
//        String predictedToken = predictor.getPredictedToken(sentence);
//        System.out.println(predictedToken);
//    }
//
//    public static String getPredictedToken(String sentence){
//        try {
//            HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.newInstance(Paths.get("/home/vinoth/IdeaProjects/prediction_by_java/artifacts/finetuned_albert/tokenizer.json"));
//
//            RobertaTokenizerResources robertaResources = new RobertaTokenizerResources("/home/vinoth/IdeaProjects/prediction_by_java/artifacts/finetuned_albert/finetuned_albert5");
//            Tokenizer robertaTokenizer = (Tokenizer) new RobertaTokenizer(robertaResources);
//
//            List<String> tokenizedSentence = robertaTokenizer.tokenize(sentence);
//            System.out.println(tokenizedSentence);
//
//            Encoding encoding = tokenizer.encode(sentence);
//
//            int maskTokenIndex = -1;  // Initialize the mask token index
//
//            String[] tokens = encoding.getTokens();
//            for (int j = 0; j < tokens.length; j++) {
//                tokens[j] = tokens[j].replace(" ", "").replace("Ġ", "");
//                System.out.print(tokens[j] + ", ");
//                if (tokens[j].equals("<mask>")) {
//                    maskTokenIndex = j;
//                }
//            }
//
//            if (maskTokenIndex == -1) {
//                // No masked token found in the sentence.
//                return null;
//            }
//
//            long[] input_ids = encoding.getIds();
//            long[] attention_mask = encoding.getAttentionMask();
//
//            OrtEnvironment environment = OrtEnvironment.getEnvironment();
//            OrtSession session = environment.createSession("/home/vinoth/IdeaProjects/prediction_by_java/artifacts/finetuned_albert/fine_tuned_model.onnx");
//
//            OnnxTensor inputIds = OnnxTensor.createTensor(environment, new long[][]{input_ids});
//            OnnxTensor attentionMask = OnnxTensor.createTensor(environment, new long[][]{attention_mask});
//
//            Map<String, OnnxTensor> inputs = new HashMap<>();
//            inputs.put("input_ids", inputIds);
//            inputs.put("attention_mask", attentionMask);
//
//            // Run the model
//            OrtSession.Result outputs = session.run(inputs);
//
//            // Get the predictions for the masked token
//            Optional<OnnxValue> optionalValue = outputs.get("output");
//            OnnxTensor predictionsTensor = (OnnxTensor) optionalValue.get();
//            float[][][] predictions = (float[][][]) predictionsTensor.getValue();
//            int[] predictedTokenIndices = getTopKIndices(predictions[0][maskTokenIndex], 5); // Helper function to get top K indices
//
//            // Get the top predicted tokens
//            String[] topPredictedTokens = new String[predictedTokenIndices.length];
//            for (int i = 0; i < predictedTokenIndices.length; i++) {
//                long predictedTokenId = predictedTokenIndices[i];
//                String predictedToken = tokenizer.decode(new long[]{predictedTokenId});
//                topPredictedTokens[i] = predictedToken;
//            }
//
//            // Return the top predicted token
//            return topPredictedTokens[0];
//        } catch (IOException e) {
//            throw new RuntimeException(e);
//        } catch (OrtException e) {
//            throw new RuntimeException(e);
//        }
//    }
//
//    private static int[] getTopKIndices(float[] array, int k) {
//        int[] indices = new int[k];
//        for (int i = 0; i < k; i++) {
//            int maxIndex = -1;
//            float maxValue = Float.NEGATIVE_INFINITY;
//            for (int j = 0; j < array.length; j++) {
//                if (array[j] > maxValue) {
//                    maxValue = array[j];
//                    maxIndex = j;
//                }
//            }
//            indices[i] = maxIndex;
//            array[maxIndex] = Float.NEGATIVE_INFINITY;
//        }
//        return indices;
//    }
//}


package io.ballerina;
public class sample {
    public static void main(String[] args){




    }
}




















