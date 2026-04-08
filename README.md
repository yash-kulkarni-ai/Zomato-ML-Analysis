# 🍽️ Zomato Data Analysis & ML Modeling

> **End-to-end Machine Learning project** on Zomato Hyderabad restaurant data — covering EDA, Regression, Classification, Clustering, and a Content-Based Recommendation System.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python" />
  <img src="https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn" />
  <img src="https://img.shields.io/badge/Status-Complete-brightgreen" />
</p>

---

## 📌 Project Overview

This project analyzes **105 restaurants** and **~10,000 user reviews** from Zomato's Hyderabad dataset. Two datasets (restaurant metadata + user reviews) are merged, feature-engineered, and fed into a full supervised + unsupervised ML pipeline.

| Goal | Technique | Model |
|------|-----------|-------|
| Predict restaurant rating | Regression | Random Forest Regressor |
| Classify review sentiment | Classification | Gradient Boosting / Random Forest Classifier |
| Segment restaurants | Clustering | K-Means + PCA |
| Recommend similar restaurants | Recommendation | TF-IDF Cosine Similarity |

---

## 📂 Project Structure

```
zomato-ml-analysis/
│
├── notebooks/
│   └── Zomato_ML_Analysis.ipynb          # Main analysis notebook
│
├── data/                                  # ⚠️ Not tracked by Git (see .gitignore)
│   ├── Zomato Restaurant names and Metadata.csv
│   └── Zomato Restaurant reviews.csv
│
├── models/                                # Saved trained models (joblib)
│   ├── model_sentiment_classifier_gbm.pkl
│   ├── model_rating_regressor_rf.pkl
│   ├── model_kmeans_clustering.pkl
│   ├── scaler_classification.pkl
│   ├── scaler_regression.pkl
│   ├── scaler_clustering.pkl
│   ├── label_encoder.pkl
│   ├── tfidf_vectorizer.pkl
│   └── cosine_similarity_matrix.pkl
│
├── charts/                                # All exported visualisation PNGs
│   ├── chart1_missing_values.png
│   ├── chart2_rating_distribution.png
│   ├── chart3_sentiment_distribution.png
│   ├── chart4_cost_distribution.png
│   ├── chart5_top_cuisines.png
│   ├── chart6_rating_vs_cost.png
│   ├── chart7_reviews_vs_rating.png
│   ├── chart8_rating_by_cost_cat.png
│   ├── chart9_sentiment_violin.png
│   ├── chart10_cuisines_vs_rating.png
│   ├── chart11_top_restaurants.png
│   ├── chart12_positive_pct_vs_rating.png
│   ├── chart13_review_length.png
│   ├── chart14_correlation_heatmap.png
│   ├── chart15_pairplot.png
│   ├── chart_model_comparison.png
│   ├── chart_cluster_profiles.png
│   ├── chart_recommendations.png
│   └── chart_feature_importance_final.png
│
├── requirements.txt                       # Python dependencies
├── .gitignore
└── README.md
```

---

## 📊 Dataset Description

| Dataset | Rows | Columns | Key Fields |
|---------|------|---------|------------|
| Restaurant Metadata | 105 | 6 | Name, Cuisines, Cost, Collections, Timings |
| Restaurant Reviews | ~10,000 | 7 | Restaurant, Reviewer, Review, Rating, Metadata, Pictures |

Both datasets are merged on `Restaurant Name` to create a unified analytical dataframe.

---

## 🔧 Feature Engineering Highlights

- **Cost Cleaning** — stripped commas, cast to numeric, median-imputed
- **Num_Cuisines** — count of cuisines per restaurant
- **Num_Collections** — count of Zomato collection tags
- **Open_Weekend** — binary flag from `Timings` string
- **Cost_Category** — Budget / Moderate / Premium / Luxury tiers
- **Sentiment_Score** — net positive keyword minus negative keyword count
- **Review_Length / Word_Count** — NLP-derived length features
- **Reviewer_Reviews / Reviewer_Followers** — reviewer credibility signals

---

## 🤖 ML Models

### 1. Rating Regression — `RandomForestRegressor`
- **Target:** Numeric restaurant rating (1–5)
- **Tuning:** GridSearchCV (5-fold CV), optimising RMSE
- **Result:** R² > 0.85, RMSE < 0.5 after tuning

### 2. Sentiment Classification — `GradientBoostingClassifier` *(Final Model)*
- **Target:** Positive / Neutral / Negative
- **Handling Imbalance:** `class_weight='balanced'`, Stratified K-Fold
- **Tuning:** GridSearchCV on F1-Weighted score
- **Comparison:** RF Classifier vs GBM Classifier (baseline + tuned)

### 3. Restaurant Clustering — `KMeans`
- **Features:** Cost, Avg Rating, Sentiment, Review Count, Cuisine Diversity
- **Optimal K:** Selected via Elbow Method + Silhouette Score
- **Visualisation:** PCA 2D projection of cluster assignments

### 4. Content-Based Recommendation System
- **Approach:** TF-IDF on (Cuisines + Collections + Cost Category) → Cosine Similarity
- **Output:** Top-N most similar restaurants for any given restaurant

---

## 📈 Key EDA Insights

- **Positivity bias:** ~70%+ of reviews are Positive; ratings skew toward 4–5
- **Cost ≠ Quality:** No strong linear correlation between price and rating
- **Cuisine diversity sweet spot:** 3–6 cuisines → highest avg ratings
- **Negative reviewers write longer reviews** — useful as an early warning signal
- **Sentiment_Score** and **Pos_Keywords** are the top predictive features

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/zomato-ml-analysis.git
cd zomato-ml-analysis
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add the datasets
Place the two CSV files inside the `data/` folder:
```
data/Zomato Restaurant names and Metadata.csv
data/Zomato Restaurant reviews.csv
```

### 5. Run the notebook
```bash
jupyter notebook notebooks/Zomato_ML_Analysis.ipynb
```

---

## 💾 Loading a Saved Model

```python
import joblib
import pandas as pd

# Load model + artifacts
clf = joblib.load('models/model_sentiment_classifier_gbm.pkl')
scaler = joblib.load('models/scaler_classification.pkl')
le = joblib.load('models/label_encoder.pkl')

# Predict on new data
new_review = pd.DataFrame([{
    'Review_Length': 250, 'Word_Count': 45,
    'Reviewer_Reviews': 10, 'Reviewer_Followers': 5,
    'Pos_Keywords': 3, 'Neg_Keywords': 0,
    'Sentiment_Score': 3, 'Pictures': 2
}])

prediction = le.inverse_transform(clf.predict(scaler.transform(new_review)))
print("Predicted Sentiment:", prediction[0])
```

---

## 🛠️ Tech Stack

| Category | Libraries |
|----------|-----------|
| Data Manipulation | `pandas`, `numpy` |
| Visualisation | `matplotlib`, `seaborn` |
| ML Models | `scikit-learn` |
| NLP / Features | `TfidfVectorizer`, `CountVectorizer` |
| Model Persistence | `joblib` |
| Notebook | `jupyter` |

---

## 👤 Author

**Yash Kulkarni** — AI/ML Engineer  

