"""
Credit Card Fraud Detection - Model Training and Evaluation
Implements 5 classification models with 6 evaluation metrics each
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef
)
import pickle
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(filepath='creditcard_csv.csv'):
    """Load and prepare the credit card dataset"""
    print("Loading dataset...")
    df = pd.read_csv(filepath)
    
    # Convert Class to numeric if it's string
    df['Class'] = df['Class'].astype(str).str.strip("'").astype(int)
    
    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution:\n{y.value_counts()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Standardize features (important for KNN and Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save test data for Streamlit app
    test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    test_df['Class'] = y_test.values
    # Save only first 1000 rows for Streamlit (free tier limitation)
    test_df.head(1000).to_csv('../test_data.csv', index=False)
    print("Test data saved to test_data.csv")
    
    # Save scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Scaler saved to scaler.pkl")
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def evaluate_model(y_true, y_pred, y_pred_proba=None):
    """Calculate all 6 evaluation metrics"""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }
    
    # AUC requires probability scores
    if y_pred_proba is not None:
        try:
            metrics['AUC'] = roc_auc_score(y_true, y_pred_proba)
        except:
            metrics['AUC'] = 0.0
    else:
        metrics['AUC'] = 0.0
    
    return metrics

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train all 5 models and evaluate them"""
    
    results = {}
    
    print("\n" + "="*80)
    print("Training and Evaluating Models")
    print("="*80)
    
    # 1. Logistic Regression
    print("\n1. Training Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_pred_proba = lr_model.predict_proba(X_test)[:, 1]
    results['Logistic Regression'] = evaluate_model(y_test, lr_pred, lr_pred_proba)
    
    # Save model
    with open('logistic_regression.pkl', 'wb') as f:
        pickle.dump(lr_model, f)
    print("✓ Logistic Regression trained and saved")
    
    # 2. Decision Tree
    print("\n2. Training Decision Tree Classifier...")
    dt_model = DecisionTreeClassifier(random_state=42, max_depth=10)
    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)
    dt_pred_proba = dt_model.predict_proba(X_test)[:, 1]
    results['Decision Tree'] = evaluate_model(y_test, dt_pred, dt_pred_proba)
    
    with open('decision_tree.pkl', 'wb') as f:
        pickle.dump(dt_model, f)
    print("✓ Decision Tree trained and saved")
    
    # 3. K-Nearest Neighbors
    print("\n3. Training K-Nearest Neighbors...")
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    knn_pred = knn_model.predict(X_test)
    knn_pred_proba = knn_model.predict_proba(X_test)[:, 1]
    results['kNN'] = evaluate_model(y_test, knn_pred, knn_pred_proba)
    
    with open('knn.pkl', 'wb') as f:
        pickle.dump(knn_model, f)
    print("✓ kNN trained and saved")
    
    # 4. Naive Bayes (Gaussian)
    print("\n4. Training Naive Bayes (Gaussian)...")
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    nb_pred = nb_model.predict(X_test)
    nb_pred_proba = nb_model.predict_proba(X_test)[:, 1]
    results['Naive Bayes'] = evaluate_model(y_test, nb_pred, nb_pred_proba)
    
    with open('naive_bayes.pkl', 'wb') as f:
        pickle.dump(nb_model, f)
    print("✓ Naive Bayes trained and saved")
    
    # 5. Random Forest (Ensemble)
    print("\n5. Training Random Forest (Ensemble)...")
    rf_model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    results['Random Forest'] = evaluate_model(y_test, rf_pred, rf_pred_proba)
    
    with open('random_forest.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    print("✓ Random Forest trained and saved")
    
    return results

def display_results(results):
    """Display results in a formatted table"""
    print("\n" + "="*80)
    print("MODEL EVALUATION RESULTS")
    print("="*80)
    
    # Create DataFrame for better visualization
    df_results = pd.DataFrame(results).T
    df_results = df_results[['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']]
    
    print("\n" + df_results.to_string())
    print("\n" + "="*80)
    
    # Find best model based on F1 score (good for imbalanced data)
    best_model = df_results['F1'].idxmax()
    print(f"\n🏆 Best Model (by F1 Score): {best_model}")
    print(f"   F1 Score: {df_results.loc[best_model, 'F1']:.4f}")
    
    # Save results to CSV
    df_results.to_csv('evaluation_results.csv')
    print("\nResults saved to evaluation_results.csv")
    
    return df_results

if __name__ == "__main__":
    # Load and prepare data
    X_train, X_test, y_train, y_test = load_and_prepare_data('../creditcard_csv.csv')
    
    # Train and evaluate all models
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Display results
    df_results = display_results(results)
    
    print("\n✅ All models trained successfully!")
    print("✅ Models saved in model/ directory")
    print("✅ Test data saved as test_data.csv")
    print("\nNext step: Run the Streamlit app with: streamlit run app.py")
