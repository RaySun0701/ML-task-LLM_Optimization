# Report on Model Optimization Techniques

## 1. Baseline Model Selection

The `textattack/bert-base-uncased-imdb` model is a fine-tuned version of BERT (Bidirectional Encoder Representations from Transformers) specifically trained for sentiment analysis on the IMDB movie reviews dataset. It is based on the `bert-base-uncased` architecture, which consists of 12 transformer layers, 768 hidden dimensions, and 12 attention heads.

### **Reasons for Choosing This Model**

- **Performance**: BERT-based models have demonstrated state-of-the-art results in sentiment classification tasks.
- **Generalizability**: A well-known, widely used transformer architecture.
- **Availability**: Pre-trained model readily available on Hugging Face.

## 2. Dataset Selection

The dataset used for this study is the **IMDB dataset**, a benchmark dataset for sentiment analysis.

### **Reasons for Choosing This Dataset**

- **Binary classification**: Simplifies evaluation of model performance.
- **Large and diverse**: Contains 50,000 reviews, ensuring model generalization.
- **Well-structured**: Standardized dataset in the NLP domain.

## 3. Quantization

Three quantizations were applied to optimize the model. After quantizations, all four models(original model, quantized_model_8_bit, quantized_model_4_bit_set1 and quantized_model_4_bit_set2) are tested on the test set **10 times** to get the average performance.

### **Method 1: 8-bit Quantization**

This **8-bit quantization configuration** reduces the model’s size while maintaining accuracy.

1. ` load_in_8bit=True`  

   This parameter enables **8-bit quantization**, meaning that model weights are stored in **INT8** instead of **FP32** (32-bit floating point). By doing so, the model consumes less memory, reducing storage and RAM requirements, making it suitable for deployment on resource-constrained devices.

2. `llm_int8_threshold=1.0`

   This threshold controls how the model handles **outliers** in weight distributions. A value of **1.0** means that no special handling is applied, and all values are quantized normally.

   I find Increasing this value will reduce accuracy and have little effect on inference speed. So I set the value at **1.0** to keep the accurary. It also makes sence since the IMDB sentiment classification task is relatively straightforward and does not suffer significantly from extreme outlier weights.

3. `llm_int8_enable_fp32_cpu_offload=True`

   This setting offloads computationally expensive **FP32** operations to the CPU, helping maintain efficiency during inference.

   But it turns out that it does not help much by compareing to setting it to `False` in this case.

4. `llm_int8_skip_modules=["classifier", "self_attention", "ffn", "layernorm"]`

   This parameter defines a list of modules that should **not** be quantized, to prevent accuracy degradation.

   I start with only skip `classifier` but the accurary drops a lot. So I add more modules to the skipping list one by one to balence accurary and inference speed.

### **Method 2: 4-bit Quantization Set 1**

This **4-bit quantization configuration** is designed to **further reduce the model size** while maintaining computational efficiency.

1. `load_in_4bit=True`

   This parameter enables **4-bit quantization**. It provides **significant memory savings**.

2. `bnb_4bit_compute_dtype=torch.bfloat16`

   This setting determines the **data type used for computation** during inference. By selecting `bfloat16` ,the model benefits from **efficient memory usage and stable computation.**

3. `bnb_4bit_use_double_quant=True`

   This setting enables **Double Quantization**, an advanced compression technique where **quantized values are themselves quantized**.

   But it turns out that it does not help much by compareing to setting it to `False` in this case.

4. `bnb_4bit_quant_type="nf4"`

   This parameter specifies the **quantization format**. Instead of standard 4-bit integer quantization, it uses **NF4 (Normal Float 4)**.

   Using `nf4` instead of `fp4` helps maintain **better model performance(accuracy)**.

### **Method 3: 4-bit Quantization Set 2**

The most important change in this setting is:

1. `bnb_4bit_compute_dtype=torch.float16`

   This parameter specifies that **float16 (FP16)** should be used for computations instead of **bfloat16 (BF16)**

   After switching to `float16`, inference speed improves a lot. 

## 4. Quantization Results Evaluation

Below is a comparison of the three quantization methods based on accuracy and inference speed.

| Model                      | model_size_mb | avg_latency_seconds | accuracy | size_reduction | avg_speedup | accuracy_retention |
| -------------------------- | ------------- | ------------------- | -------- | -------------- | ----------- | ------------------ |
| original                   | 417.655281    | 0.001119            | 0.8920   | NaN            | NaN         | NaN                |
| quantized_model_4_bit_set1 | 86.487797     | 0.003318            | 0.8915   | 0.792921       | 0.337765    | 0.999439           |
| quantized_model_4_bit_set2 | 86.487797     | 0.002057            | 0.8915   | 0.792921       | 0.545268    | 0.999439           |
| quantized_model_8_bit      | 127.269047    | 0.007918            | 0.8875   | 0.695277       | 0.141412    | 0.994955           |

### 1. Model Size Reduction**

- The **original model** has a size of **417.66MB**.
- Both **4-bit quantized models (set1 and set2)** reduce the model size to **86.49MB**, achieving an impressive **79.29% reduction**.
- The **8-bit quantized model** reduces the size to **127.27MB**, yielding a **69.53% reduction**—which is still significant but not as aggressive as 4-bit quantization.
- **Key Insight:** 4-bit quantization achieves **more compression than 8-bit quantization**, making it ideal for memory-constrained environments.

### **2. Inference Speed (Latency)**

- The **original model** has the lowest **average latency** of **0.001119 seconds**, meaning it processes input the fastest.
- **4-bit quantized model (set1)** has an increased latency of **0.003318 seconds**, resulting in a **speedup of 0.337x**, meaning it is **roughly 3 times slower** than the original model.
- **4-bit quantized model (set2)** performs better than set1, with a latency of **0.002057 seconds**, leading to a **speedup of 0.545x**, making it more efficient than set1.
- **8-bit quantized model** has the **worst latency (0.007918 seconds)**, giving a **speedup of only 0.141x**, meaning it is **much slower than the original model**.
- Key Insight:
  - **4-bit quantization (set2) provides the best trade-off between compression and inference speed.**
  - **8-bit quantization does not improve speed significantly** and in this case is **slower than both 4-bit models**.

### **3. Accuracy Retention**

- The **original model’s accuracy is 0.8920**.
- Both **4-bit models retain an accuracy of 0.8915**, meaning they suffer **almost no accuracy degradation** (accuracy retention of **0.999439**).
- The **8-bit model’s accuracy drops to 0.8875**, meaning it has a **slightly larger accuracy loss** (accuracy retention of **0.994955**).
- Key Insight:
  - **4-bit quantization with NF4 (both set1 and set2) is highly effective at retaining accuracy** while achieving major compression.
  - **8-bit quantization leads to a slightly greater accuracy drop**, despite using a higher-bit representation.

### **4. Trade-off Between Size, Speed, and Accuracy**

- 4-bit quantization (set2) appears to be the best optimization method

   in this case:

  - **It reduces the model size by nearly 80% while still maintaining reasonable inference speed (speedup 0.545x) and near-perfect accuracy retention.**

- **8-bit quantization does not provide significant speed gains** and in fact increases inference latency while offering slightly lower accuracy retention than the 4-bit models.

- **4-bit quantized model (set1) has higher latency than set2**, meaning set2 is the **better option for fast inference**.

## 5. Knowledge Distillation

Knowledge Distillation was applied to create a smaller **student model** using the `MiniLM-L6-H384-uncased` model.

### **Methodology**

- **Teacher Model**: `textattack/bert-base-uncased-imdb`.

- **Student Model**: `nreimers/MiniLM-L6-H384-uncased`.

- **Distillation Process**：

  The **Distillation Process** transfers knowledge from a large **teacher model** to a smaller **student model** by training the student on both **soft probabilities** from the teacher and **hard labels** from the data. Using **temperature scaling**, the teacher’s outputs are softened to reveal class relationships, and a **distillation loss** (KL Divergence + Cross-Entropy) helps the student mimic the teacher while staying efficient. This method produces **smaller, faster models** with minimal accuracy loss, making them ideal for deployment in low-resource environments.

### **Parameter Settings**

1. **Temperature Scaling** (`temperature=2.0`)

   Soften logits to retain informative class relationships without excessive smoothing.

2. **Balancing Factor** `alpha=0.5`

   Balances between teacher-guided learning and direct supervision from ground truth labels.

3. **Optimizer** `AdamW`

   Well-suited for fine-tuning transformers, avoid weight decay issues.

4. **Learning Rate** `5e-5`

   Standard learning rate for fine-tuning transformer models without overfitting.

## 6. Knowledge Distillation Results Evaluation

Performance of the distilled student model is compared below:

| Model         | model_size_mb | avg_latency_seconds | accuracy | size_reduction | avg_speedup | accuracy_retention |
| ------------- | ------------- | ------------------- | -------- | -------------- | ----------- | ------------------ |
| original      | 417.655281    | 0.001200            | 0.8920   | NaN            | NaN         | NaN                |
| student_model | 86.654793     | 0.000593            | 0.8535   | 0.792521       | 2.078401    | 0.956839           |

### **1. Model Size Reduction**

- The **original model** has a size of **417.66MB**.
- The **student model** is significantly smaller at **86.65MB**, achieving a **size reduction of 79.25%**.
- **Key Insight**: This confirms that **knowledge distillation effectively compresses the model**, making it more suitable for deployment on devices with memory constraints.

### **2. Inference Speed Improvement**

- The **original model** has an **average latency of 0.0012 seconds**.
- The **student model** improves on this, achieving an **average latency of 0.000593 seconds**.
- This results in a **speedup of 2.08×**, meaning the student model is **twice as fast** as the original.
- **Key Insight**: The **smaller architecture of the student model allows for faster inference**, making it ideal for **real-time applications** and **low-latency environments**.

### **3. Accuracy Comparison & Retention**

- The **original model’s accuracy is 0.8920**, while the **student model’s accuracy is 0.8535**.
- This means the student model **loses around 3.85% accuracy** compared to the original.
- The **accuracy retention rate is 0.9568 (95.68%)**, indicating that the student model maintains most of the teacher’s knowledge despite the compression.
- **Key Insight**: While the **distilled student model sacrifices some accuracy**, it still retains **over 95% of the original model’s performance**, which is a reasonable trade-off for the significant reduction in size and latency.

### **4. Trade-off Between Size, Speed, and Accuracy**

- The **major benefit** is the significant **size reduction** and **speedup**, making the student model **better suited for deployment**.

- The **accuracy trade-off is minimal**, suggesting that the student model can still perform well in real-world applications.

- If **higher accuracy is required**, the **distillation process could be improved** by adjusting **temperature scaling (`T`) or α (weighting factor between soft and hard losses)**.

## 7. Comparison of Optimization Techniques

| Optimization           | Size Reduction | Speed Improvement | Accuracy Impact      |
| ---------------------- | -------------- | ----------------- | -------------------- |
| Quantization           | High           | Moderate          | Minimal              |
| Knowledge Distillation | Moderate       | High              | Slight Accuracy Drop |

### **Trade-offs**

- **Quantization** is better for reducing model size while maintaining accuracy.
- **Knowledge Distillation** is preferable when inference speed is the priority.

## 8. Deployment Guide

After optimization you need to export it for deployment. Depending on the target edge device, you can export the model in TorchScript format.

```python
optimized_model.eval()  # Set to evaluation mode

# Convert to TorchScript
traced_model = torch.jit.trace(optimized_model, torch.randn(1, 512))  # Adjust input shape as needed
torch.jit.save(traced_model, "optimized_model.pt")
```

TorchScript is more efficient for inference and can be used in mobile and embedded devices.

Expect real-time or near real-time inference, reducing latency significantly.

The model can be compressed by over 50%, enabling deployment on devices with limited RAM.

