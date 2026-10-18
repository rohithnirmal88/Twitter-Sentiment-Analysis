# Twitter Sentiment Analysis

A machine learning project that classifies the sentiment of tweets into **Positive**, **Negative**, **Neutral**, or **Irrelevant** using a Scikit-learn pipeline (TF-IDF + Decision Tree Classifier).

## 📌 Overview

This project uses a labeled dataset of tweets (each tagged with the game/brand they mention and a sentiment label) to train a text classification model. The pipeline handles text vectorization and categorical encoding internally, and the trained model is exported for reuse.

## 📂 Dataset

**File:** `twitter_training.csv`

| Column   | Description                                      |
|----------|---------------------------------------------------|
| `id`     | Numeric identifier                                 |
| `city`   | Entity/topic the tweet refers to (e.g. Borderlands)|
| `review` | Sentiment label — `Positive`, `Negative`, `Neutral`, `Irrelevant` |
| `text`   | Raw tweet text                                     |

- Original size: **74,682 rows**
- Contained **686 null values** and **2,701 duplicate rows**
- After cleaning (dropping nulls & duplicates): **71,655 rows**

## 🧹 Data Preprocessing

1. Loaded the dataset with `pandas`
2. Checked for and removed duplicate rows
3. Checked for and removed null values
4. Dropped the `id` column (not useful as a feature)

## 🛠️ Feature Engineering

Built with a Scikit-learn `ColumnTransformer`:

- **`text`** → `TfidfVectorizer()` — converts tweet text into TF-IDF weighted numerical features
- **`city`** → `OneHotEncoder(handle_unknown='ignore')` — encodes the topic/entity column

## 🤖 Model

- **Algorithm:** `DecisionTreeClassifier` (Scikit-learn), wrapped together with preprocessing in a single `Pipeline`
- **Train/Test split:** 80% / 20% (`random_state=42`)
- Other classifiers imported for experimentation: `LogisticRegression`, `SVC`, `MultinomialNB`

### 📊 Results

| Metric              | Train Set | Test Set |
|---------------------|-----------|----------|
| Accuracy             | 0.99      | 0.81     |
| Precision (weighted) | 0.99      | 0.81     |
| Recall (weighted)    | 0.99      | 0.81     |
| F1-score (weighted)  | 0.99      | 0.81     |

> The gap between train (99%) and test (81%) accuracy indicates the model is **overfitting** on the training data — a good next step would be pruning the tree (`max_depth`, `min_samples_leaf`) or trying an ensemble method like Random Forest.

Per-class performance on the test set:

| Class        | Precision | Recall | F1-score |
|--------------|-----------|--------|----------|
| Irrelevant   | 0.77      | 0.78   | 0.77     |
| Negative     | 0.83      | 0.82   | 0.83     |
| Neutral      | 0.82      | 0.82   | 0.82     |
| Positive     | 0.80      | 0.81   | 0.81     |

## 💾 Model Export

The trained pipeline is saved using `joblib` for later inference without retraining:

```python
import joblib
joblib.dump(main_pipeline, "model.pkl")
```

## 🧰 Tech Stack

- Python
- pandas
- scikit-learn (`TfidfVectorizer`, `OneHotEncoder`, `ColumnTransformer`, `Pipeline`, `DecisionTreeClassifier`, `train_test_split`, `classification_report`, `GridSearchCV`)
- joblib

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas scikit-learn joblib jupyter
```

### Run the notebook

```bash
jupyter notebook twitter.ipynb
```

### Use the saved model

```python
import joblib
import pandas as pd

model = joblib.load("model.pkl")

sample = pd.DataFrame({
    "city": ["Borderlands"],
    "text": ["I love playing this game, it's amazing!"]
})

prediction = model.predict(sample)
print(prediction)
```

## 📁 Project Structure

```
├── twitter.ipynb           # Main notebook: EDA, preprocessing, model training & evaluation
├── twitter_training.csv    # Training dataset
├── model.pkl                # Saved trained pipeline (generated after running the notebook)
└── README.md
```

## 🔮 Future Improvements

- Address overfitting with hyperparameter tuning (`GridSearchCV` is already imported) or a different model (Random Forest, SVM, Logistic Regression)
- Add text cleaning steps (lowercasing, removing URLs/mentions/punctuation, stopword removal)
- Try richer embeddings (Word2Vec, GloVe, or transformer-based embeddings like BERT)
- Deploy the model as a simple web app (Flask/Streamlit) for live predictions

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
