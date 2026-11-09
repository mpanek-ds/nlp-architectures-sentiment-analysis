# NLP Sentiment Analysis: Architectural Evolution from RNN to Transformers

## 📌 **Project Overview**
This project presents an empirical, end-to-end comparative study of Natural Language Processing (NLP) sequence modeling architectures applied to binary sentiment classification. Benchmarked on the standardized Stanford IMDB dataset (highly polarized movie reviews), the repository chronicles the evolution of modern NLP paradigms. The experiment traces the progression from recurrence-based neural networks (Vanilla RNN, Bidirectional LSTM) to attention-based mechanisms (custom Mini-Transformer from scratch) and modern transfer learning via fine-tuned Transformer foundations (DistilBERT).

## 🎯 **Engineering & Research Objective**
Sequence classification in NLP requires capturing long-range semantic dependencies while maintaining computational efficiency and gradient stability. The objective was to implement, train, and benchmark four foundational NLP architectures under standardized evaluation protocols using PyTorch:
- Assess the susceptibility of recurrence to vanishing and exploding gradients over long token sequences.
- Measure the impact of bidirectional context gates in recurrent units (LSTMs).
- Evaluate a custom Multi-Head Self-Attention encoder implemented without recurrence.
- Contrast scratch-trained architectures against contextualized pre-trained representations (Transfer Learning via DistilBERT) using architecture-tailored optimization hyperparameters (e.g., distinct learning rates and epoch budgets suitable for scratch training vs. fine-tuning).

## 🧠 **Architectural Evolution & Model Pipeline**
The pipeline evaluates four sequential generations of NLP models:

1. **Vanilla RNN (Simple RNN):**
   - Single recurrent layer processing sequential token embeddings step-by-step.
   - Highlights the inductive bias of sequential processing and inherent limitations regarding vanishing gradients over longer review contexts.
2. **Bidirectional LSTM (Bi-LSTM):**
   - Recurrent architecture equipped with input, forget, and output gates to retain long-term dependencies.
   - Processes input sequences forward and backward simultaneously, concatenating directional hidden states for sentiment context aggregation.
3. **Mini-Transformer (Custom Transformer Encoder from Scratch):**
   - Recurrence-free architecture relying entirely on Multi-Head Self-Attention and Scaled Dot-Product Attention mechanisms.
   - Incorporates Positional Encodings, Feed-Forward sublayers, Layer Normalization (`LayerNorm`), and residual connections.
4. **DistilBERT (Pre-trained Transformer Fine-Tuning):**
   - Leverages `DistilBertForSequenceClassification` loaded with pre-trained weights (`distilbert-base-uncased`).
   - Retains 97% of BERT's language understanding capabilities while operating with 40% fewer parameters through knowledge distillation.
   - Fine-tuned using `DistilBertTokenizerFast` subword tokenization with attention masks and dynamic padding.

## ⚙️ **Data Pipeline & Training Strategy**
- **Automated Data Ingestion:** Integrated directly via Hugging Face `datasets` (`stanfordnlp/imdb`), ensuring a strictly balanced 50/50 class distribution across stratified train, validation, and test splits without class skew.
- **PyTorch DataLoader & Batching:** Encapsulated tokenized sequences and attention masks into custom PyTorch `Dataset` wrappers, utilizing GPU device management (`torch.cuda`).
- **Custom PyTorch Training Loop:** Fully controlled manual training loop implementing `torch.optim.Adam` (for scratch-trained models) and `AdamW` (for DistilBERT fine-tuning), tracking running Cross-Entropy Loss (`nn.CrossEntropyLoss`) and epoch-level validation metrics with early stopping.
- **Regularization:** Applied Dropout across layers, learning rate scheduling (`ReduceLROnPlateau`), and validation loss monitoring.

## 📊 **Comparative Benchmark Summary**

| Generation | Architecture | Core Mechanism | Primary Trade-off / Characteristic |
| :--- | :--- | :--- | :--- |
| **Gen 1** | Vanilla RNN | Simple Recurrent Cell | Fast per-step execution, high vanishing gradient risk |
| **Gen 2** | Bidirectional LSTM | Gated Memory Cells | Captures bidirectional context; sequential bottleneck |
| **Gen 3** | Mini-Transformer | Multi-Head Self-Attention | Fully parallelized training, sensitive to hyperparameter tuning |
| **Gen 4** | DistilBERT | Pre-trained Transformer | Superior transfer learning accuracy and rapid convergence |

## 🛠️ **Technology Stack**
- **Language:** Python
- **Deep Learning Framework:** PyTorch (`torch`, `torch.nn`, `torch.optim`, `torch.utils.data`)
- **NLP & Ecosystem:** Hugging Face (`transformers`, `datasets`, `DistilBertTokenizerFast`)
- **Data Manipulation & Metrics:** NumPy, Pandas, Scikit-learn (`classification_report`, `confusion_matrix`)
- **Visualization:** Matplotlib, Seaborn

## 🚀 **Repository Structure & How to Run**
- `nlp_sentiment_analysis.ipynb` – Complete Jupyter Notebook containing EDA, data pipelines, model definitions, training loops, learning curves, and final evaluation.

To reproduce the experiments:
1. Clone or download this repository.
2. Open and run `nlp_sentiment_analysis.ipynb` in Google Colab (with GPU acceleration enabled) or a local PyTorch environment.
3. The Stanford IMDB dataset is automatically loaded and cached via the Hugging Face `datasets` library (no manual dataset download required).

*(Note: The Jupyter Notebook containing the code, theoretical background, and EDA is written in Polish as it was part of a university submission, but the code structure and logic follow universal data science standards.)*

## 👥 **Contributors**
- Mateusz Panek
- Piotr Jasiak
