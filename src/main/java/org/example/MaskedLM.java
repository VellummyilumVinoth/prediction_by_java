package org.example;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.onnxruntime.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.*;

public class MaskedLM {

    private static final Logger logger = LoggerFactory.getLogger(MaskedLM.class);

    public static void main(String[] args) throws IOException, OrtException {

        String maskedStatement = "int <mask> = getCount();";

        // Load the model and tokenizer
        HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.newInstance(Paths.get("artifacts/tokenizer/tokenizer.json"));
        Encoding encodings = tokenizer.encode(maskedStatement);
        String[] tokenizedStatement = encodings.getTokens();
        System.out.println(tokenizedStatement);
        for (int i = 0; i < tokenizedStatement.length; i++) {
            tokenizedStatement[i] = tokenizedStatement[i].replace(" ", "").replace("Ġ", "").toLowerCase();
            System.out.print(tokenizedStatement[i] + ", ");
        }

        // Get the index of the mask token
        int maskedTokenIndex = -1;
        for (int i = 0; i < tokenizedStatement.length; i++) {
            if (tokenizedStatement[i].equals("<mask>")) {
                maskedTokenIndex = i;
                break;
            }
        }

        long[][] input_ids0 =new long[][]{encodings.getIds()};
        long[][] attentionMask0 = new long[][]{encodings.getAttentionMask()};
        long[][] typeIds = new long[][]{encodings.getTypeIds()};

//        // Convert tokenized statement to tensor of input IDs
//        for (int i = 0; i < tokenizedStatement.length; i++) {
//            System.out.print(input_ids0[i] + ", ");
//        }
//        System.out.println();

//        // Convert tokenized statement to tensor of attention masks
//        for (int i = 0; i < tokenizedStatement.length; i++) {
//            System.out.print(attentionMask0[i] + ", ");
//        }
//        System.out.println();

//        // Convert tokenized statement to tensor of token type IDs
//        for (int i = 0; i < tokenizedStatement.length; i++) {
//            System.out.print(tokenizedStatement[i] + ", ");
//        }
//        System.out.println();

        OrtEnvironment environment = OrtEnvironment.getEnvironment();
        OrtSession session = environment.createSession("artifacts/model1.onnx");
        System.out.println(session.getInputInfo());
        System.out.println(session.getOutputInfo());


        OnnxTensor inputIds = OnnxTensor.createTensor(environment,input_ids0);
        OnnxTensor attentionmask = OnnxTensor.createTensor(environment, attentionMask0);
        OnnxTensor tokenizedstatement = OnnxTensor.createTensor(environment, typeIds);



        Map<String, OnnxTensor> inputs = new LinkedHashMap<>();
        inputs.put("attention_mask",attentionmask);

        inputs.put("input_ids", inputIds);
        inputs.put("token_type_ids",tokenizedstatement);


        System.out.println(inputIds);
        System.out.println(attentionmask);
        System.out.println(tokenizedStatement);
        // Run the inference
        OrtSession.Result outputValues = session.run(inputs);

        // Get the logits for the masked token
        OnnxTensor logits = (OnnxTensor) outputValues.get(0);

        // Get the predicted token index
        long[] predictedIndices = logits.getLongBuffer().array();
        long predictedIndex = predictedIndices[maskedTokenIndex];

        // Convert predicted index to token
        String predictedToken = tokenizer.decode(new long[]{predictedIndex});

        System.out.println("Predicted Token: " + predictedToken);
    }
}
