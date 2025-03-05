**Still working on it. Will upload notebook and report later. Please let me know if there is any problem with the access.**

### **Model Details:**
- **Base Model:** `textattack/bert-base-uncased-imdb`
- **Architecture:** A BERT-based model fine-tuned for sentiment classification on the IMDB dataset.
- **Purpose:** The goal is to apply quantization to the model to improve inference speed while maintaining acceptable accuracy.

### **Methodology for Quantization:**

1. **Loading the Pre-trained Model & Tokenizer:**
   - The model (`textattack/bert-base-uncased-imdb`) and its corresponding tokenizer are loaded.

2. **Dataset Preparation:**
   - The IMDB dataset is loaded using `datasets.load_dataset("imdb")`.
   - A subset of 2000 test samples is randomly selected for evaluation.
   - Tokenization is applied using the tokenizer with truncation and padding.

3. **Quantization Process:**
   - The `bitsandbytes` library is used to perform **4-bit quantization** on the model.
   - The `bnb.QuantizedLinear` layers replace standard linear layers in the model.
   - The quantized model is then evaluated for performance improvements.

### **Evaluation Metrics:**

**1. Model Size (MB)**  
Computes the total storage required for the model in megabytes.

$$ \text{Model Size} = \frac{\sum \text{parameter size} + \sum \text{buffer size}}{1024^2} $$

**2. Inference Latency (seconds per sample)**  
Measures the average time taken to process a single input.

$$ \text{Avg Latency} = \frac{\text{Total Inference Time}}{\text{Number of Samples}} $$

**3. Accuracy**  
Computes classification accuracy by comparing predicted labels to ground truth labels.

$$ \text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Samples}} $$

**4. Size Reduction (%)**  
Measures how much the quantized model reduces in size compared to the original model.

$$ \text{Size Reduction} = 1 - \frac{\text{Quantized Model Size}}{\text{Original Model Size}} $$

**5. Speedup Factor**  
Compares the latency of the quantized model against the original model.

$$ \text{Speedup} = \frac{\text{Original Model Latency}}{\text{Quantized Model Latency}} $$

**6. Accuracy Retention (%)**  
Measures how much accuracy is retained after quantization.

$$ \text{Accuracy Retention} = \frac{\text{Quantized Model Accuracy}}{\text{Original Model Accuracy}} $$

These metrics are computed and compared between the **original** and **quantized** models.
