**Please let me know if there is any problem with the access.**

`Report_of_LLM-Optimization-task.md`: Report of the task

`quantization.ipynb`: Notebook of quantization approach (run on T4 GPU)

`knowledge_distillation.ipynb`: Notebook of knowledge distillation approach (run on T4 GPU)

### **Model Details:**
- **Base Model:** `textattack/bert-base-uncased-imdb`
- **Architecture:** A BERT-based model fine-tuned for sentiment classification on the IMDB dataset.

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
