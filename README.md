# LoRA Adaptation of GPT-2 for Sentiment Classification on SST-2

## 📌 Overview
This project investigates the efficiency of adapting the GPT-2 language model for sentiment classification on the **SST-2 (Stanford Sentiment Treebank)** dataset.  
We compared three approaches:  
- **Full Fine-Tuning** – Updating all parameters of GPT-2.  
- **LoRA (Low-Rank Adaptation)** – Updating only a small set of low-rank matrices in the attention layers.  
- **PreLayer** – Training only the token and positional embedding layers before the Transformer blocks.  

Results show that **LoRA** not only reduces GPU memory usage and training time but also **outperforms** full fine-tuning in accuracy and F1-score.  

---

## 🎯 Objectives
- Compare full fine-tuning and parameter-efficient fine-tuning techniques (LoRA and PreLayer).
- Evaluate the trade-offs between **model performance** and **computational cost**.
- Demonstrate the practicality of LoRA in **resource-constrained environments**.

---

## 📊 Results Summary
| Method              | Accuracy | F1-Score | Trainable Params | % of Total | Training Time (s) | GPU Memory Allocated |
|--------------------|----------|----------|------------------|------------|--------------------|----------------------|
| No Fine-Tuning     | 50.46%   | 21.45%   | –                | –          | –                  | –                    |
| Full Fine-Tuning   | 83.26%   | 83.41%   | 124M             | 100%       | 3492.40            | 1.89 GB              |
| LoRA               | **86.93%** | **87.42%** | 148K             | 0.12%      | 2464.95            | 0.49 GB              |
| PreLayer           | 83.14%   | 83.72%   | 46K              | 0.037%     | 2662.89            | 0.49 GB              |

---

## 🛠️ Methodology
1. **Dataset** – SST-2 from GLUE benchmark (`datasets.load_dataset("glue", "sst2")`).  
2. **Tokenization** – `GPT2Tokenizer` with padding token set to `eos_token`, max length 128, truncation enabled.  
3. **Models** – `GPT2ForSequenceClassification` with three training configurations:
   - Full Fine-Tuning
   - LoRA (PEFT)
   - PreLayer  
4. **Metrics** – Accuracy, F1-Score, training time, trainable parameters, GPU memory usage.  
5. **Training Setup** – 2 epochs, batch size 16, learning rate `2e-4`, weight decay `0.01`.

---

💡 **Key Insights**
- LoRA achieved better accuracy and F1 than full fine-tuning with only 0.12% of trainable parameters.
- LoRA reduced GPU memory usage by ~74% and training time by ~29%.
- PreLayer offers near full fine-tuning performance with an even smaller parameter footprint.

📚 **References**
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021.
- Socher et al., *Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank*, 2013.
- Vaswani et al., *Attention is All You Need*, 2017.

🧠 **Skills & Technologies**
Python, PyTorch, Hugging Face Transformers, PEFT (LoRA), NLP, Sentiment Analysis, Fine-Tuning, GPU Computing, SST-2 Dataset, Model Evaluation.
