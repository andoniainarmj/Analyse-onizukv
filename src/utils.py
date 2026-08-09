import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import joblib

def load_model(filepath):
    return joblib.load(filepath)

def plot_confusion_matrix(y_true, y_pred, model_name='Model'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    return plt

def plot_roc_curve(y_true, y_pred_proba, model_name='Model'):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    return plt

def plot_feature_importance(model, feature_names, top_n=15):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[-top_n:][::-1]
        plt.figure(figsize=(10, 6))
        plt.title(f'Top {top_n} Feature Importances')
        plt.bar(range(len(indices)), importances[indices])
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        return plt
    else:
        print("Ce modèle n'a pas d'attribut feature_importances_")
        return None

def plot_class_distribution(y, title='Class Distribution'):
    plt.figure(figsize=(8, 6))
    y.value_counts().plot(kind='bar', color=['green', 'red'])
    plt.title(title)
    plt.xlabel('Churn')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.tight_layout()
    return plt

def predict_single(model, X_single, feature_names):
    prediction = model.predict(X_single)[0]
    probability = model.predict_proba(X_single)[0][1]
    return {
        'prediction': 'Churn' if prediction == 1 else 'No Churn',
        'probability': probability,
        'confidence': max(model.predict_proba(X_single)[0])
    }

def print_model_insights(model_results):
    print("INSIGHTS DU MODÈLE:\n")
    for metric, value in model_results.items():
        if metric not in ['confusion_matrix', 'classification_report']:
            try:
                print(f"  - {metric.replace('_', ' ').title()}: {value:.4f}")
            except Exception:
                print(f"  - {metric.replace('_', ' ').title()}: {value}")

def create_prediction_dataframe(predictions, probabilities, feature_names):
    df = pd.DataFrame({
        'Prediction': ['Churn' if p == 1 else 'No Churn' for p in predictions],
        'Churn_Probability': probabilities,
        'Confidence': np.max(probabilities.reshape(-1, 1), axis=1)
    })
    return df
