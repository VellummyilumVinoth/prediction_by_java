package io.ballerina;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.onnxruntime.*;
import ai.onnxruntime.OnnxTensor;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

public class MaskedTokenPrediction {

    public static void main(String[] args){
//        String sentence = "\nint [MASK] = getCount();\n";
//        MaskedTokenPrediction predictor = new MaskedTokenPrediction();
//        String predictedToken = predictor.getPredictedToken(sentence);
//        System.out.println(predictedToken);
    }

    public static String getPredictedToken(String sentence){
        try {
            HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.newInstance(Paths.get("/home/vinoth/IdeaProjects/language-server-identifier-generator/predictor/artifacts/albert_base/albert_tokenizer.json"));

            Encoding encoding = tokenizer.encode(sentence);

            int maskTokenIndex = -1;  // Initialize the mask token index

            String[] tokens = encoding.getTokens();
            for (int j = 0; j < tokens.length; j++) {
                tokens[j] = tokens[j].replace(" ", "").replace("_", "");
//                System.out.print(tokens[j] + ", ");
                if (tokens[j].equals("[MASK]")) {
                    maskTokenIndex = j;
                }
            }

            if (maskTokenIndex == -1) {
//                System.out.println("No masked token found in the sentence.");
                return null;  // Exit if no masked token is found
            }

            long[] input_ids = encoding.getIds();
            long[] attention_mask = encoding.getAttentionMask();
            long[] token_type_ids = encoding.getTypeIds();

            OrtEnvironment environment = OrtEnvironment.getEnvironment();

            OrtSession session = environment.createSession("/home/vinoth/IdeaProjects/language-server-identifier-generator/predictor/artifacts/albert_base/model.onnx");
            OnnxTensor inputIds = OnnxTensor.createTensor(environment, new long[][]{input_ids});
            OnnxTensor attentionMask = OnnxTensor.createTensor(environment, new long[][]{attention_mask});
            OnnxTensor tokenTypeIds = OnnxTensor.createTensor(environment, new long[][]{token_type_ids});

            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put("input_ids", inputIds);
            inputs.put("attention_mask", attentionMask);
            inputs.put("token_type_ids", tokenTypeIds);

            // Run the model
            OrtSession.Result outputs = session.run(inputs);

            // Get the predictions for the masked token
            Optional<OnnxValue> optionalValue = outputs.get("logits");
            OnnxTensor predictionsTensor = (OnnxTensor) optionalValue.get();
            float[][][] predictions = (float[][][]) predictionsTensor.getValue();
            int[] predictedTokenIndices = getTopKIndices(predictions[0][maskTokenIndex], 5); // Helper function to get top K indices

            // Get the top predicted tokens
            String[] topPredictedTokens = new String[predictedTokenIndices.length];
            for (int i = 0; i < predictedTokenIndices.length; i++) {
                long predictedTokenId = predictedTokenIndices[i];
                String predictedToken = tokenizer.decode(new long[]{predictedTokenId});
                topPredictedTokens[i] = predictedToken;
            }

            // Return the top predicted token
            return topPredictedTokens[2];
        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (OrtException e) {
            throw new RuntimeException(e);
        }
    }

    private static int[] getTopKIndices(float[] array, int k) {
        int[] indices = new int[k];
        for (int i = 0; i < k; i++) {
            int maxIndex = -1;
            float maxValue = Float.NEGATIVE_INFINITY;
            for (int j = 0; j < array.length; j++) {
                if (array[j] > maxValue) {
                    maxValue = array[j];
                    maxIndex = j;
                }
            }
            indices[i] = maxIndex;
            array[maxIndex] = Float.NEGATIVE_INFINITY;
        }
        return indices;
    }
}
