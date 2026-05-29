# Large-Scale Sentiment Classification: Classical ML vs. Deep Transformers

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=black)

## 📌 Project Overview
This repository contains a structural comparative analysis between traditional natural language processing techniques and deep learning architectures. The project maps out the exact performance-to-compute boundaries between a highly optimized, low-latency classical baseline and a computationally heavy, context-aware transformer network. 

**Author:** Yash (M.Tech. Scholar, Department of Computer Science & Engineering)

## 🎯 Core Objectives
* **Benchmark Classical vs. Deep Learning:** Compare the sparse feature extraction of TF-IDF against the bidirectional semantic context of BERT (`bert-base-uncased`).
* **Handle Linguistic Volatility:** Deploy aggressive regular expression filtering to clean heavy noise, slang, and unstructured grammar from microblogging text.
* **Evaluate Operational Trade-offs:** Map the marginal accuracy gains of multi-head self-attention against the extreme computational/memory footprint of deep neural networks.

## 🗄️ Dataset Details
* **Source:** [Sentiment140 (via Kaggle)](https://www.kaggle.com/datasets/kazanova/sentiment140)
* **Scale:** 1.6 Million annotated tweets (Balanced: 800k Positive, 800k Negative).
* **Ingestion:** Automated programmatic pull using the `kagglehub` API.

## 🛠️ Technology Stack
* **Language:** Python
* **Classical Pipeline:** `scikit-learn` (TF-IDF Vectorizer, Logistic Regression with `lbfgs` solver)
* **Deep Learning Pipeline:** `transformers` (Hugging Face), `torch` (PyTorch)
* **Hardware Environment:** Designed for CPU baseline processing and NVIDIA GPU (CUDA) acceleration for transformer fine-tuning.

## ⚙️ Installation & Setup

1. **Clone the repository:**
```bash
   git clone [https://github.com/your-username/sentiment-classification-benchmark.git](https://github.com/your-username/sentiment-classification-benchmark.git)
   cd sentiment-classification-benchmark
