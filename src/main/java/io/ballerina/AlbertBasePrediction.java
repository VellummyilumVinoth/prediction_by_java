package io.ballerina;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

public class AlbertBasePrediction {

    public static void main(String[] args) {
//        String sentence = "string [MASK];";
//        AlbertBasePrediction predictor = new AlbertBasePrediction();
//        String[] predictedTokens = predictor.getBasePredictedToken(sentence);
//        for (String predictedToken : predictedTokens) {
//            System.out.println(predictedToken);
//        }
    }

    public static String[] getBasePredictedToken(String sentence) {
        try {
            HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.newInstance(Paths.get("/home/vinoth/IdeaProjects/prediction_by_java/artifacts/albert_base/albert_tokenizer.json"));

            Encoding encoding = tokenizer.encode(sentence);

            int maskTokenIndex = -1;  // Initialize the mask token index

            String[] tokens = encoding.getTokens();
            for (int j = 0; j < tokens.length; j++) {
                tokens[j] = tokens[j].replace(" ", "").replace("_", "");
                if (tokens[j].equals("[MASK]")) {
                    maskTokenIndex = j;
                }
            }

            if (maskTokenIndex == -1) {
                return null;  // Exit if no masked token is found
            }

            long[] inputIds = encoding.getIds();
            long[] attentionMask = encoding.getAttentionMask();
            long[] token_type_ids = encoding.getTypeIds();

            OrtEnvironment environment = OrtEnvironment.getEnvironment();

            OrtSession session = environment.createSession("/home/vinoth/IdeaProjects/prediction_by_java/artifacts/albert_base/model.onnx");
            OnnxTensor inputIdsTensor = OnnxTensor.createTensor(environment, new long[][]{inputIds});
            OnnxTensor attentionMaskTensor = OnnxTensor.createTensor(environment, new long[][]{attentionMask});
            OnnxTensor tokenTypeIds = OnnxTensor.createTensor(environment, new long[][]{token_type_ids});

            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put("input_ids", inputIdsTensor);
            inputs.put("attention_mask", attentionMaskTensor);
            inputs.put("token_type_ids", tokenTypeIds);

            // Run the model
            OrtSession.Result outputs = session.run(inputs);

            // Get the predictions for the masked token
            Optional<OnnxValue> optionalValue = outputs.get("logits");
            OnnxTensor predictionsTensor = (OnnxTensor) optionalValue.get();
            float[][][] predictions = (float[][][]) predictionsTensor.getValue();
            int[] predictedTokenIndices = getTopKIndices(predictions[0][maskTokenIndex], 5); // Helper function to get top K indices

            // Get the top predicted tokens
            String[] topPredictedTokens = new String[Math.max(predictedTokenIndices.length, 5)];
            for (int i = 0; i < Math.min(predictedTokenIndices.length, 5); i++) {
                long predictedTokenId = predictedTokenIndices[i];
                String predictedToken = tokenizer.decode(new long[]{predictedTokenId});

                // Clean the predicted token
                StringBuilder cleanTokenBuilder = new StringBuilder(predictedToken.length());
                for (char c : predictedToken.toCharArray()) {
                    if (Character.isLetterOrDigit(c) || Character.isWhitespace(c)) {
                        cleanTokenBuilder.append(c);
                    }
                }
                String cleanedToken = cleanTokenBuilder.toString().trim();

                topPredictedTokens[i] = cleanedToken;
            }

            // Return the top predicted tokens
            return topPredictedTokens;
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
