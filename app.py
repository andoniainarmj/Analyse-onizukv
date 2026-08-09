
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from src.data_preprocessing import DataPreprocessor
from src.utils import (
    plot_confusion_matrix, plot_roc_curve, plot_feature_importance,
    plot_class_distribution, predict_single, print_model_insights
)
import matplotlib.pyplot as plt

# Configuration Streamlit
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("Customer Churn Prediction Dashboard")
st.markdown("---")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Sélectionne une page:", 
                        ["Home", "Données", "Modèles", "Prédictions", "Résultats"])

# Charger les données et modèles
@st.cache_resource
def load_data():
    try:
        preprocessor = DataPreprocessor()
        df = preprocessor.load_data('data/churn_data.csv')
        return df
    except:
        return None

@st.cache_resource
def load_model():
    try:
        return joblib.load('models/best_model.pkl')
    except:
        return None

@st.cache_resource
def load_results():
    try:
        with open('results/model_performance.json', 'r') as f:
            return json.load(f)
    except:
        return None

df = load_data()
model = load_model()
results = load_results()

# PAGE: Home
if page == "Home":
    st.header("Bienvenue!")
    
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Clients", len(df) if df is not None else "N/A")
    with col2:
        if df is not None and 'Churn' in df.columns:
            churn_rate = (df['Churn'].value_counts().get('Yes', 0) / len(df) * 100)
            st.metric("Taux de Churn", f"{churn_rate:.1f}%")
    with col3:
        if results:
            best_auc = max([v['auc_roc'] for v in results.values()])
            st.metric("Meilleur AUC-ROC", f"{best_auc:.4f}")

# PAGE: Données
elif page == "Données":
    st.header("Exploration des Données")
    
    if df is not None:
        st.subheader("Aperçu des Données")
        st.dataframe(df.head(10))
        
        st.subheader("Statistiques Descriptives")
        st.dataframe(df.describe())
        
        st.subheader("Distribution du Churn")
        if 'Churn' in df.columns:
            fig = plot_class_distribution(df['Churn'], 'Distribution du Churn')
            st.pyplot(fig)
        
        st.subheader("Informations sur les Données")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Nombre de Lignes", df.shape[0])
        with col2:
            st.metric("Nombre de Colonnes", df.shape[1])
        with col3:
            st.metric("Valeurs Manquantes", df.isnull().sum().sum())
    else:
        st.error("Impossible de charger les données")

# PAGE: Modèles
elif page == "Modèles":
    st.header("Comparaison des Modèles")
    
    if results:
        # Tableau comparatif
        st.subheader("Performances des Modèles")
        
        comparison_data = []
        for model_name, model_results in results.items():
            comparison_data.append({
                'Modèle': model_name,
                'Accuracy': f"{model_results['accuracy']:.4f}",
                'Precision': f"{model_results['precision']:.4f}",
                'Recall': f"{model_results['recall']:.4f}",
                'F1-Score': f"{model_results['f1_score']:.4f}",
                'AUC-ROC': f"{model_results['auc_roc']:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Graphiques
        st.subheader("Visualisations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Accuracy vs AUC-ROC")
            fig, ax = plt.subplots(figsize=(8, 5))
            model_names = [r['model_name'] for r in results.values()]
            accuracy = [r['accuracy'] for r in results.values()]
            auc_roc = [r['auc_roc'] for r in results.values()]
            
            x = np.arange(len(model_names))
            width = 0.35
            ax.bar(x - width/2, accuracy, width, label='Accuracy', alpha=0.8)
            ax.bar(x + width/2, auc_roc, width, label='AUC-ROC', alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            ax.legend()
            ax.set_ylabel('Score')
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.write("### F1-Score Comparison")
            fig, ax = plt.subplots(figsize=(8, 5))
            f1_scores = [r['f1_score'] for r in results.values()]
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            ax.bar(model_names, f1_scores, color=colors, alpha=0.7)
            ax.set_ylabel('F1-Score')
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.error("Résultats des modèles non disponibles")

# PAGE: Prédictions
elif page == "Prédictions":
    st.header("Faire une Prédiction")
    
    if model is not None:
        st.info("Fonctionnalité à implémenter: Entrer les données d'un client pour obtenir une prédiction")
        st.write("Cette section permettrait de faire des prédictions en temps réel sur de nouveaux clients.")
    else:
        st.error("Modèle non disponible")

# PAGE: Résultats
elif page == "Résultats":
    st.header("Détails des Résultats")
    
    if results:
        selected_model = st.selectbox("Sélectionne un modèle:", list(results.keys()))
        model_result = results[selected_model]
        
        st.subheader(f"Résultats: {selected_model}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{model_result['accuracy']:.4f}")
        with col2:
            st.metric("Precision", f"{model_result['precision']:.4f}")
        with col3:
            st.metric("Recall", f"{model_result['recall']:.4f}")
        with col4:
            st.metric("F1-Score", f"{model_result['f1_score']:.4f}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("AUC-ROC", f"{model_result['auc_roc']:.4f}")
        with col2:
            st.metric("Specificity", f"{model_result['specificity']:.4f}")
        
        st.subheader("Matrice de Confusion")
        cm = np.array(model_result['confusion_matrix'])
        fig, ax = plt.subplots(figsize=(8, 6))
        import seaborn as sns
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['No Churn', 'Churn'],
                    yticklabels=['No Churn', 'Churn'])
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        st.pyplot(fig)
        
        st.subheader("Classification Report")
        report_df = pd.DataFrame(model_result['classification_report']).transpose()
        st.dataframe(report_df)
    else:
        st.error("Résultats non disponibles")

st.markdown("---")
