
# 🧠 Sentence Analysis using DistilBERT and PyTorch

This project leverages the power of **DistilBERT**, a lightweight and faster version of BERT, to perform **sentence-level analysis** such as sentiment classification or sentence classification using **PyTorch** and the **HuggingFace Transformers** library.

---

## 📌 Overview

* **Model**: DistilBERT (`distilbert-base-uncased`)
* **Frameworks**: PyTorch + Hugging Face Transformers
* **Task**: Sentiment analysis

---

## 🧑‍💻 Tech Stack

* Python 🐍
* PyTorch 🔥
* HuggingFace Transformers 🤗
* scikit-learn 📊
* pandas, numpy, matplotlib

---

## 🎯 Objective

Build a robust sentence classification pipeline that:

* Preprocesses raw text
* Tokenizes and encodes using DistilBERT tokenizer
* Trains a classifier on sentence embeddings
* Evaluates with accuracy, precision, recall, F1-score


---

## 🔧 Installation & Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/Sentence-analysis-using-distilBERT-and-PyTorch.git
   cd Sentence-analysis-using-distilBERT-and-PyTorch
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

   Or manually:

   ```bash
   pip install torch transformers scikit-learn pandas numpy matplotlib
   ```

---

## 🚀 How It Works

1. **Data Loading**

   * Load a dataset with labeled sentences (CSV or text file).

2. **Preprocessing**

   * Lowercasing, cleaning (optional)
   * Tokenization using `DistilBertTokenizerFast`
   * Encoding using `DistilBertModel`

3. **Model Architecture**

   * DistilBERT base model
   * A custom classification head (Linear + ReLU + Dropout + Softmax)

4. **Training**

   * CrossEntropyLoss + AdamW optimizer
   * Epoch-wise training loop with batch updates


---

## 📌 Requirements

* Python 
* PyTorch
* transformers
* scikit-learn
* numpy

