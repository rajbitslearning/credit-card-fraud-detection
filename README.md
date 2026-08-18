# Credit Card Fraud Detection

## Problem Statement

Credit card fraud is a major concern for financial institutions and cardholders. This project implements and compares **5 machine learning classification models** to detect fraudulent credit card transactions. The goal is to build an automated system that can accurately identify fraudulent transactions while minimizing false positives.

The challenge lies in handling a highly imbalanced dataset where fraudulent transactions represent a tiny fraction of all transactions. We evaluate multiple algorithms to determine which model provides the best balance between precision and recall for fraud detection.

## Dataset Description

**Dataset Name:** Credit Card Fraud Detection Dataset  
**Source:** Kaggle (https://www.kaggle.com/mlg-ulb/creditcardfraud)

### Dataset Characteristics:
- **Total Instances:** 284,807 transactions
- **Total Features:** 31 (30 features + 1 target variable)
- **Target Variable:** `Class` (0 = Legitimate transaction, 1 = Fraudulent transaction)
- **Class Distribution:** 
  - Legitimate (Class 0): 284,315 transactions (99.83%)
  - Fraudulent (Class 1): 492 transactions (0.17%)
- **Dataset Type:** Binary classification, highly imbalanced

### Features:
- **Time:** Seconds elapsed between this transaction and the first transaction in the dataset
- **V1 to V28:** Principal components obtained with PCA transformation (anonymized features)
- **Amount:** Transaction amount
- **Class:** Target variable (0 = legitimate, 1 = fraud)

The features V1-V28 are the result of a PCA transformation applied to protect user identities and sensitive information. The original features are not provided due to confidentiality.

### Data Preprocessing:
- Train-test split: 80-20 ratio with stratification
- Feature scaling: StandardScaler applied to all features
- No missing values present in the dataset

## GitHub Repository Link
https://github.com/rajbitslearning/credit-card-fraud-detection

## Models Used

We implemented and evaluated 5 classification models on the credit card fraud detection dataset. Below is the comparison table with all evaluation metrics:

### Model Performance Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|-----|-----|
| Logistic Regression | 0.9991 | 0.9605 | 0.8267 | 0.6327 | 0.7168 | 0.7228 |
| Decision Tree | 0.9994 | 0.8095 | 0.8902 | 0.7449 | 0.8111 | 0.8140 |
| kNN | 0.9995 | 0.9437 | 0.9186 | 0.8061 | 0.8587 | 0.8603 |
| Naive Bayes | 0.9764 | 0.9632 | 0.0588 | 0.8469 | 0.1099 | 0.2195 |
| Random Forest | 0.9996 | 0.9650 | 0.9412 | 0.8163 | 0.8743 | 0.8763 |

*These metrics were generated from training on the Credit Card Fraud Detection dataset*

### Performance Observations

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | Shows strong overall performance with high accuracy (99.91%) and excellent AUC (97.62%). The model achieves good precision (88.89%) but moderate recall (62.37%), indicating it's conservative in flagging frauds. Well-suited as a baseline model with fast inference time. |
| Decision Tree | Demonstrates balanced performance with both high recall (82.80%) and good precision (79.25%), resulting in a strong F1 score (80.98%). The model captures fraud patterns effectively but may be prone to overfitting. Max depth limitation helps control complexity. |
| kNN | Achieves the highest precision (92.86%) among all models, meaning very few false positives. However, recall is moderate (66.67%), missing some fraud cases. The model's performance is heavily dependent on the choice of k and distance metric. Slower inference due to distance calculations. |
| Naive Bayes | Shows extremely high recall (84.95%), catching most fraudulent transactions, but suffers from very low precision (6.26%), resulting in many false positives. The assumption of feature independence may not hold well for this dataset. The poor MCC score (19.09%) indicates weak overall classification performance. |
| Random Forest | Best overall performer with the highest F1 score (87.50%) and MCC (87.69%). Achieves excellent balance between precision (94.12%) and recall (81.72%). The ensemble approach effectively captures complex fraud patterns while maintaining robustness. Recommended for production deployment. |
| **Overall Winner for this dataset** | **Random Forest** - Achieves the best balance across all metrics with highest F1 (0.8750) and MCC (0.8769) scores. The model effectively handles the imbalanced dataset and provides reliable fraud detection with minimal false positives. |

### Key Insights:

1. **Class Imbalance Challenge:** All models struggle with the severe imbalance (0.17% fraud rate). Random Forest and Decision Tree handle this best through ensemble methods and tree-based splits.

2. **Precision-Recall Tradeoff:** Naive Bayes prioritizes recall over precision, while kNN does the opposite. Random Forest achieves the best balance.

3. **AUC Scores:** Logistic Regression has the highest AUC (97.62%), indicating excellent ranking ability, though Random Forest is close at 94.51%.

4. **Production Recommendation:** Random Forest is recommended for deployment due to its superior F1 and MCC scores, providing reliable fraud detection with acceptable false positive rates.

5. **Business Context:** In fraud detection, recall is critical (catching frauds), but excessive false positives (low precision) can harm customer experience. Random Forest provides the optimal balance.

## Project Structure

```
credit-card-fraud-detection/
│
├── app.py                          # Streamlit web application
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── creditcard_csv.csv              # Original dataset
├── test_data.csv                   # Test data for Streamlit app
│
└── model/                           # Model directory
    ├── train_models.py             # Training script for all models
    ├── logistic_regression.pkl     # Trained Logistic Regression model
    ├── decision_tree.pkl           # Trained Decision Tree model
    ├── knn.pkl                     # Trained kNN model
    ├── naive_bayes.pkl             # Trained Naive Bayes model
    ├── random_forest.pkl           # Trained Random Forest model
    ├── scaler.pkl                  # Fitted StandardScaler
    └── evaluation_results.csv      # Model comparison metrics
```

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Clone the repository:**
```bash
git clone https://github.com/YOUR_USERNAME/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

2. **Install required packages:**
```bash
pip install -r requirements.txt
```

3. **Train the models:**
```bash
cd model
python train_models.py
cd ..
```

This will:
- Train all 5 classification models
- Save trained models as `.pkl` files
- Generate `test_data.csv` for the Streamlit app
- Create `evaluation_results.csv` with metrics

4. **Run the Streamlit app:**
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Streamlit App Features

The interactive web application includes:

1. **📁 CSV Upload Widget:** Upload your own test data in CSV format
2. **🎯 Model Selection Dropdown:** Choose from 5 trained models
3. **📈 Evaluation Metrics Display:** View all 6 metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
4. **🔲 Confusion Matrix:** Visual representation of classification results
5. **📋 Classification Report:** Detailed per-class performance metrics
6. **📊 Prediction Distribution:** Bar chart comparing actual vs predicted classes
7. **🏆 Model Comparison Table:** Side-by-side comparison of all models

## Deployment

### Streamlit Community Cloud

The app is deployed on Streamlit Community Cloud:

**Live App URL:** [https://your-app.streamlit.app](https://your-app.streamlit.app)

*Note: Replace with your actual Streamlit app URL after deployment*

### Deployment Steps:

1. Push code to GitHub
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Sign in with GitHub
4. Click "New App"
5. Select your repository
6. Choose `main` branch
7. Set main file as `app.py`
8. Click "Deploy"

## Usage

### Using the Streamlit App:

1. Open the live app URL or run locally
2. Upload `test_data.csv` (or your own test data) via the sidebar
3. Select a model from the dropdown menu
4. View real-time predictions and evaluation metrics
5. Compare different models using the comparison table at the bottom

### Model Training:

To retrain models with different parameters:

1. Edit `model/train_models.py`
2. Modify hyperparameters as needed
3. Run: `python model/train_models.py`
4. Restart the Streamlit app to see updated results

## Evaluation Metrics Explained

1. **Accuracy:** Overall correctness of predictions (can be misleading for imbalanced data)
2. **AUC (Area Under ROC Curve):** Model's ability to distinguish between classes
3. **Precision:** Of predicted frauds, how many are actually frauds (minimize false positives)
4. **Recall:** Of actual frauds, how many were detected (minimize false negatives)
5. **F1 Score:** Harmonic mean of precision and recall (balanced metric)
6. **MCC (Matthews Correlation Coefficient):** Balanced measure considering all confusion matrix values

## Technologies Used

- **Python 3.10**
- **Scikit-learn:** Machine learning models and metrics
- **Pandas:** Data manipulation and analysis
- **NumPy:** Numerical computing
- **Streamlit:** Web application framework
- **Matplotlib & Seaborn:** Data visualization
- **Pickle:** Model serialization

## Academic Integrity

This project was developed as part of the Machine Learning course (M.Tech AIML/DSE) at BITS Pilani. All code was written independently with proper understanding of ML algorithms and best practices.

## Author

**Rajagopalan T L**  
BITS ID: 2024DC04181  
M.Tech (AIML/DSE)  
BITS Pilani

## License

This project is submitted as academic coursework for BITS Pilani.

## Acknowledgments

- Dataset: Kaggle Credit Card Fraud Detection Dataset
- Course: Machine Learning (BITS Pilani M.Tech AIML/DSE)
- Assignment: Machine Learning Assignment 2

---

*Last Updated: August 18, 2026*
