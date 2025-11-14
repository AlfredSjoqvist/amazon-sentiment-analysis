# Amazon Review Sentiment Analysis

This project implements a sentiment analysis pipeline for Amazon product reviews using classical machine learning methods on top of a custom text preprocessing pipeline. The code loads raw CSV review data, cleans and normalizes the text, and trains multiple classifiers to predict review scores.

The focus is on building a transparent, end to end workflow for text classification, rather than using opaque end to end black box models.

## Skills Demonstrated

- Natural language processing for real world review data
- Text preprocessing: HTML stripping, punctuation removal, tokenization, stop word removal, stemming
- Feature engineering with bag of words representations
- Supervised learning with scikit learn
- Implementation of multiple baseline models for comparison
- Evaluation via train test splits, confusion matrices, and accuracy
- Handling large datasets with pickled intermediate artifacts
- Modular Python code structure and reproducible pipelines

---

## Project Structure

### Root and `src/`

The main training scripts live in the root or `src/` folder and focus on model training and evaluation.

Example scripts:

- `train_logistic_regression.py`  
  Trains a logistic regression classifier on the processed Amazon review text. Uses `CountVectorizer` to build a bag of words representation, fits the model, and reports accuracy and a confusion matrix over the five star rating scale.

- `train_naive_bayes.py`  
  Trains a multinomial Naive Bayes classifier as a fast and strong baseline. Logs accuracy, confusion matrix, execution time, and memory usage.

- `train_decision_tree.py`  
  Uses a decision tree classifier to model the review score as a function of bag of words features. Useful for interpretability and as a non linear baseline.

- `train_gradient_boosting.py`  
  Trains a gradient boosting classifier on the bag of words features. Demonstrates the tradeoff between model complexity, training time, and predictive performance.

Each script:

- Loads preprocessed reviews and labels from pickled artifacts
- Splits data into training and test sets
- Builds a `CountVectorizer` vocabulary on the training data
- Trains the chosen classifier
- Evaluates on held out test data and prints metrics

---

### `preprocessing/`

The preprocessing folder contains a stepwise text cleaning and transformation pipeline applied to the raw Amazon review CSV file.

Typical steps:

- `01_load_raw_csv.py`  
  Loads the original `Reviews.csv` dataset into a Pandas DataFrame and serializes it to a pickled file for faster subsequent runs.

- `02_clean_columns.py`  
  Drops columns that are not used for the sentiment task, such as product id and profile information, and removes duplicate rows.

- `03_remove_html_tags.py`  
  Strips HTML tags from the review text using regular expressions.

- `04_normalize_and_strip_punctuation.py`  
  Removes punctuation, normalizes whitespace, and lowercases the text.

- `05_tokenize_and_clean_whitespace.py`  
  Splits each review into a list of tokens and removes empty tokens.

- `06_remove_stop_words.py`  
  Applies a custom stop word list to remove high frequency, low information words from the token lists.

- `07_stem_tokens.py`  
  Uses NLTK`s SnowballStemmer to reduce words to stems, shrinking the vocabulary size and grouping word variants.

- `08_join_tokens.py`  
  Converts token lists back into space joined strings ready for use by scikit learn`s `CountVectorizer`.

- `09_build_scikit_datasets.py`  
  Extracts final review texts and corresponding labels into Python lists and stores them as `reviews.pkl` and `labels.pkl` for consumption by the training scripts.

Intermediate results are stored as pickled DataFrames and lists to make the pipeline easy to resume without rerunning all steps from scratch.

---

## Data

The data originates from a public Amazon product reviews corpus. The key fields used are:

- `Text`  
  The free form review text.

- `Score`  
  The user assigned rating on a one to five scale.

The project treats the problem as a five class classification task, predicting the score from the text.

The raw CSV file is kept in `preprocessing/` (or in `data/raw/` if further separated) and is not modified in place. All cleaning and transformations operate on copies stored as pickled DataFrames.

---

## Usage

A typical workflow:

1. Run the preprocessing pipeline scripts in order.  
2. Run one of the training scripts, for example:

   ```bash
   python src/train_logistic_regression.py
