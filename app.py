"""
Credit Card Fraud Detection - Streamlit App
Interactive web application for model comparison and evaluation
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# Title and description
st.title("💳 Credit Card Fraud Detection")
st.markdown("""
This application demonstrates **5 machine learning classification models** 
for detecting fraudulent credit card transactions.
""")

# Sidebar for model selection
st.sidebar.header("⚙️ Configuration")

# Model selection dropdown
model_options = {
    'Logistic Regression': 'model/logistic_regression.pkl',
    'Decision Tree': 'model/decision_tree.pkl',
    'kNN': 'model/knn.pkl',
    'Naive Bayes': 'model/naive_bayes.pkl',
    'Random Forest': 'model/random_forest.pkl'
}

selected_model_name = st.sidebar.selectbox(
    "Select Model",
    options=list(model_options.keys())
)

# File upload section
st.sidebar.header("📁 Upload Test Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file",
    type=['csv'],
    help="Upload a CSV file with the same structure as the training data"
)

@st.cache_resource
def load_model(model_path):
    """Load the trained model"""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_scaler():
    """Load the fitted scaler"""
    try:
        with open('model/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return scaler
    except:
        return None

def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """Calculate all evaluation metrics"""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, zero_division=0),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }
    
    if y_pred_proba is not None:
        try:
            metrics['AUC'] = roc_auc_score(y_true, y_pred_proba)
        except:
            metrics['AUC'] = 0.0
    else:
        metrics['AUC'] = 0.0
    
    return metrics

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud'],
                ax=ax)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    ax.set_title(f'Confusion Matrix - {selected_model_name}')
    
    return fig

# Main content
if uploaded_file is not None:
    # Load test data
    try:
        test_data = pd.read_csv(uploaded_file)
        st.success(f"✅ Data loaded successfully! Shape: {test_data.shape}")
        
        # Check if 'Class' column exists
        if 'Class' not in test_data.columns:
            st.error("❌ Error: 'Class' column not found in the uploaded data.")
        else:
            # Separate features and target
            X_test = test_data.drop('Class', axis=1)
            y_test = test_data['Class']
            
            # Load selected model
            model_path = model_options[selected_model_name]
            model = load_model(model_path)
            
            if model is not None:
                st.subheader(f"📊 Results for: {selected_model_name}")
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                try:
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                except:
                    y_pred_proba = None
                
                # Calculate metrics
                metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
                
                # Display metrics in columns
                st.markdown("### 📈 Evaluation Metrics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
                    st.metric("AUC Score", f"{metrics['AUC']:.4f}")
                
                with col2:
                    st.metric("Precision", f"{metrics['Precision']:.4f}")
                    st.metric("Recall", f"{metrics['Recall']:.4f}")
                
                with col3:
                    st.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
                    st.metric("MCC Score", f"{metrics['MCC']:.4f}")
                
                # Display confusion matrix and classification report
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🔲 Confusion Matrix")
                    fig = plot_confusion_matrix(y_test, y_pred)
                    st.pyplot(fig)
                
                with col2:
                    st.markdown("### 📋 Classification Report")
                    report = classification_report(y_test, y_pred, 
                                                   target_names=['Legitimate', 'Fraud'],
                                                   output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df.style.format("{:.4f}"), height=300)
                
                # Show prediction distribution
                st.markdown("### 📊 Prediction Distribution")
                pred_df = pd.DataFrame({
                    'Actual': y_test.value_counts().sort_index(),
                    'Predicted': pd.Series(y_pred).value_counts().sort_index()
                })
                pred_df.index = ['Legitimate (0)', 'Fraud (1)']
                st.bar_chart(pred_df)
                
    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
        st.info("Please ensure your CSV file has the correct structure with a 'Class' column.")

else:
    # Show instructions when no file is uploaded
    st.info("👈 Please upload a CSV file from the sidebar to begin")
    
    # Show sample data structure
    st.markdown("### 📝 Expected Data Format")
    st.markdown("""
    Your CSV file should contain:
    - All feature columns (Time, V1-V28, Amount)
    - A 'Class' column (0 = Legitimate, 1 = Fraud)
    """)
    
    # If test_data.csv exists, show it as example
    if os.path.exists('test_data.csv'):
        st.markdown("### 📋 Sample Test Data")
        sample_data = pd.read_csv('test_data.csv')
        st.dataframe(sample_data.head(10))
        st.info(f"Sample data shape: {sample_data.shape}")
        
        # Option to use the sample data
        if st.button("🚀 Use Sample Test Data"):
            st.rerun()

# Show model comparison table
st.markdown("---")
st.markdown("### 🏆 Model Comparison")

# Load evaluation results if available
if os.path.exists('model/evaluation_results.csv'):
    results_df = pd.read_csv('model/evaluation_results.csv', index_col=0)
    
    # Reorder columns
    results_df = results_df[['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']]
    
    # Highlight the best values in each column
    st.dataframe(
        results_df.style.highlight_max(axis=0, color='lightgreen').format("{:.4f}"),
        use_container_width=True
    )
    
    # Show best model
    best_model = results_df['F1'].idxmax()
    st.success(f"🏆 Best Model (by F1 Score): **{best_model}** with F1 = {results_df.loc[best_model, 'F1']:.4f}")
else:
    st.warning("⚠️ Model evaluation results not found. Please run train_models.py first.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Credit Card Fraud Detection | Machine Learning Assignment 2</p>
    <p><strong>2024DC04181 - Rajagopalan T L</strong></p>
    <p>BITS Pilani M.Tech (AIML/DSE)</p>
</div>
""", unsafe_allow_html=True)
