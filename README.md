# Analyse-onizukv: Customer Churn Prediction

## 📊 Présentation du Projet

Ce projet est une **analyse complète et une prédiction du churn client** (taux d'attrition). C'est un cas d'usage réel très demandé par les entreprises et les recruteurs.

### Objectif
Prédire quels clients sont susceptibles de quitter l'entreprise pour pouvoir mettre en place des stratégies de rétention.

### Données utilisées
- **Dataset** : Telecom Customer Churn (Kaggle)
- **Taille** : ~7,000 clients avec 20+ variables
- **Type** : Classification binaire (Churn: Oui/Non)

## 🏗️ Structure du Projet

```
Analyse-onizukv/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── churn_data.csv
├── notebooks/
│   ├── 01_EDA.ipynb
│   └── 02_modeling.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── utils.py
├── models/
│   └── best_model.pkl
├── results/
│   └── model_performance.json
└── app.py (Streamlit Dashboard)
```

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/andoniainarmj/Analyse-onizukv.git
cd Analyse-onizukv
pip install -r requirements.txt
```

### Exécuter l'analyse
```bash
# Entraîner le modèle
python src/model_training.py

# Lancer le dashboard interactif
streamlit run app.py
```

## 📈 Pipeline de Travail

1. **Exploration des Données (EDA)**
   - Analyse descriptive
   - Visualisations
   - Détection des valeurs manquantes

2. **Préparation des Données**
   - Nettoyage
   - Feature engineering
   - Normalisation
   - Balancement des classes (si nécessaire)

3. **Modélisation**
   - Logistic Regression (baseline)
   - Random Forest
   - Gradient Boosting (XGBoost)
   - Comparaison des modèles

4. **Évaluation**
   - Accuracy, Precision, Recall, F1-Score
   - AUC-ROC
   - Confusion Matrix
   - Cross-validation

5. **Déploiement**
   - Dashboard Streamlit pour les prédictions en temps réel

## 📊 Résultats Attendus

- **Accuracy** : ~80%+
- **AUC-ROC** : ~0.85+
- **Insights** : Identifier les facteurs clés du churn

## 🛠️ Technologies Utilisées

- **Python 3.9+**
- **Pandas** : Manipulation de données
- **NumPy** : Calculs numériques
- **Scikit-learn** : Modèles ML
- **XGBoost** : Gradient Boosting
- **Matplotlib & Seaborn** : Visualisations
- **Streamlit** : Dashboard interactif
- **Joblib** : Sauvegarde des modèles

## 📚 Compétences Démontrées

✅ Exploration et analyse de données (EDA)
✅ Nettoyage et préparation des données
✅ Feature engineering
✅ Machine Learning (classification)
✅ Évaluation des modèles
✅ Comparaison de modèles
✅ Déploiement avec Streamlit
✅ Bonnes pratiques de code

## 👨‍💼 Pour les Recruteurs

Ce projet démontre :
- **Compétences techniques** : Python, ML, Data Science
- **Pensée analytique** : Comprendre les données et le problème
- **Rigueur scientifique** : Méthodologie solide et évaluation robuste
- **Communication** : Code lisible et bien documenté
- **Sens métier** : Cas d'usage réel avec impact business

## 📞 Contact

Pour des questions sur ce projet, n'hésitez pas à me contacter!

---

**Créé avec ❤️ pour les Data Scientists en herbe**
