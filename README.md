# Persian Sentiment Analysis: Logistic Regression vs. LSTM

A comparison of a traditional machine learning baseline and a recurrent neural network for **binary sentiment classification of Persian user reviews**. The project is implemented in Python in a Kaggle/Jupyter notebook and focuses on preprocessing informal Persian text, training two models, and comparing their classification performance.

## Project overview

Persian reviews often contain inconsistent spacing, repeated letters, and excessive punctuation. This notebook first normalizes and cleans the comments, then explores two approaches:

1. **TF-IDF + Logistic Regression**, a fast and interpretable baseline.
2. **Embedding + LSTM**, a PyTorch sequence model that processes token sequences.

The models predict a binary sentiment label based on the `label_id` field in the dataset.

## Dataset

The notebook reads the **training split** of the [ParsiAI Snappfood Sentiment Analysis dataset](https://huggingface.co/datasets/ParsiAI/snappfood-sentiment-analysis) using the Hugging Face dataset URL:

```python
pd.read_csv("hf://datasets/ParsiAI/snappfood-sentiment-analysis/train.csv")
```

It then creates its own stratified **80/20 train/test split** from that loaded data. The separate published `validation.csv` and `test.csv` splits are not used in the current implementation.

> **Naming note:** Some notebook markdown refers to Digikala reviews, but the actual input path points to the **Snappfood** dataset. This README describes the dataset used by the code.

## Preprocessing

The notebook uses `hazm.Normalizer` and custom regular expressions to process informal Persian text:

- Normalize Persian characters and spacing and remove diacritics.
- Reduce repeated characters and punctuation (for example, elongated words and repeated exclamation marks).
- Remove missing and empty comments, and comments shorter than three words.
- Exclude comments that appear with conflicting labels.
- Remove duplicate comments before splitting the data.
- Split the cleaned data with stratification to preserve label proportions.

## Models

### 1. TF-IDF + Logistic Regression

Text features are created with `TfidfVectorizer`:

| Setting | Value |
| --- | --- |
| N-grams | Unigrams and bigrams (`1, 2`) |
| Minimum document frequency | `2` |
| Maximum document frequency | `0.9` |
| Sublinear TF | Enabled |
| Class weights | `balanced` |
| Solver | `liblinear` |
| Maximum iterations | `2000` |

The notebook reports a classification report and confusion matrix and includes a `predict_sentiment(text)` function for trying individual comments with this baseline.

### 2. PyTorch LSTM

The LSTM branch tokenizes reviews by whitespace, builds a vocabulary using only training comments, encodes tokens into IDs, and pads or truncates sequences to **200 tokens**. Its architecture includes:

- A trainable embedding layer (**200 dimensions**)
- A single LSTM layer (**128 hidden units**)
- Dropout (**0.4**)
- A linear layer producing one sentiment logit

Padding is handled with `pack_padded_sequence` so that padded tokens do not contribute to the sequence representation.

| Training setting | Value |
| --- | --- |
| Batch size | `32` |
| Epochs | `20` |
| Optimizer | Adam |
| Learning rate | `0.0005` |
| Loss | `BCEWithLogitsLoss` |
| Gradient clipping | Maximum norm `1.0` |
| Prediction threshold | `0.5` |

The model runs on CUDA when available and otherwise uses the CPU.

## Results

The notebook **reports** the following approximate performance:

| Model | Reported accuracy |
| --- | ---: |
| TF-IDF + Logistic Regression | **86.0%** |
| LSTM (best observed epoch) | **85.09%** |

The accompanying analysis identifies the LSTM's best observed performance at **epoch 3** and notes signs of overfitting in later epochs as training loss continues to decrease. The baseline performs slightly better in the reported comparison, while requiring much less training effort.

**Important:** These figures come from the notebook's commentary and manually specified comparison plot; they have not been independently reproduced for this README. The training-loss plot also uses a manually entered series. Run the notebook to produce fresh classification reports for your own execution.

## Running the notebook

1. Open the `.ipynb` file in **Kaggle** or a Jupyter environment.
2. Make sure the environment can access the Hugging Face `hf://` dataset URL (internet and compatible dataset I/O support may be needed).
3. Install the main dependencies if they are missing:

   ```bash
   pip install pandas numpy scikit-learn torch matplotlib hazm huggingface_hub hf_xet
   ```

4. Run the cells in order. The notebook itself also includes a `!pip install hazm` cell.
5. Inspect the Logistic Regression classification report, the LSTM reports across epochs, and the final comparison plot.

**Main libraries:** Python, pandas, NumPy, scikit-learn, PyTorch, Hazm, and Matplotlib.

## Repository contents

```text
.
├── notebook1258282c8a.ipynb   # Data cleaning, model training, and comparison
└── README.md                  # Project documentation
```

If the notebook has a different filename in your repository, update the entry above accordingly.

## Limitations and possible improvements

- **Evaluation protocol:** The LSTM is evaluated on the test split after every epoch, and the notebook discusses choosing the best epoch based on those results. A separate validation split should be used for model selection; the test split should be reserved for a single final evaluation.
- **Early stopping:** The analysis recommends retaining the best epoch, but the current training loop does not save or restore the best LSTM checkpoint automatically.
- **Reproducibility:** Random seeds are not consistently fixed for the split and training, so results can vary across runs.
- **Plots:** Log training losses and evaluation results programmatically rather than entering numbers manually.
- **Model comparisons:** Try pretrained Persian-language models such as ParsBERT, or evaluate character-aware tokenization for noisy informal writing.
- **Inference:** `predict_sentiment` currently uses the fitted TF-IDF/Logistic Regression pipeline. A separate LSTM inference helper would be needed to compare both models on the same new input.

## Takeaway

This project shows that **simple TF-IDF features with Logistic Regression remain a strong baseline** for Persian sentiment analysis. The LSTM offers a useful introduction to sequence modeling, but the current experiment also illustrates the importance of validation, overfitting control, and reproducible evaluation.
