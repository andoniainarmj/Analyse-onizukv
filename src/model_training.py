"""Model Training Module for Churn Prediction"""

import json
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from sklearn.model_selection import cross_val_score
import numpy as np
import xgboost as xgb
from data_preprocessing import DataPreprocessor


class ModelTrainer:
    """Classe pour l'entraînement et l'évaluation des modèles"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def train_logistic_regression(self, X_train, y_train):
        """Entraîner un modèle Logistic Regression"""
        print("\n📍 Entraînement: Logistic Regression...")
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        print("✅ Logistic Regression entraîné")
        return model
    
    def train_random_forest(self, X_train, y_train):
        """Entraîner un modèle Random Forest"""
        print("\n🌲 Entraînement: Random Forest...")
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        print("✅ Random Forest entraîné")
        return model
    
    def train_gradient_boosting(self, X_train, y_train):
        """Entraîner un modèle Gradient Boosting"""
        print("\n📈 Entraînement: Gradient Boosting...")
        model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        model.fit(X_train, y_train)
        print("✅ Gradient Boosting entraîné")
        return model
    
    def train_xgboost(self, X_train, y_train):
        """Entraîner un modèle XGBoost"""
        print("\n⚡ Entraînement: XGBoost...")
        model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, use_label_encoder=False)
        model.fit(X_train, y_train, verbose=False)
        print("✅ XGBoost entraîné")
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Évaluer un modèle"""
        print(f"\n🔍 Évaluation: {model_name}...")
        
        # Prédictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Métriques
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)
        
        # Rapport de classification
        report = classification_report(y_test, y_pred, output_dict=True)
        
        results = {
            'model_name': model_name,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc_roc': float(auc_roc),
            'specificity': float(specificity),
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
        
        print(f"\n📊 Résultats {model_name}:")
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
        print(f"   AUC-ROC:   {auc_roc:.4f}")
        
        return results
    
    def cross_validate(self, model, X_train, y_train, model_name, cv=5):
        """Effectuer une validation croisée"""
        print(f"\n🔄 Cross-validation ({cv}-fold): {model_name}...")
        
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
        print(f"   Scores: {scores}")
        print(f"   Moyenne: {scores.mean():.4f} (+/- {scores.std():.4f})")
        
        return scores.mean(), scores.std()
    
    def train_all_models(self, X_train, y_train, X_test, y_test):
        """Entraîner tous les modèles et les évaluer"""
        print("\n" + "="*60)
        print("🤖 ENTRAÎNEMENT DES MODÈLES")
        print("="*60)
        
        # Entraîner tous les modèles
        self.models['Logistic Regression'] = self.train_logistic_regression(X_train, y_train)
        self.models['Random Forest'] = self.train_random_forest(X_train, y_train)
        self.models['Gradient Boosting'] = self.train_gradient_boosting(X_train, y_train)
        self.models['XGBoost'] = self.train_xgboost(X_train, y_train)
        
        # Évaluer tous les modèles
        print("\n" + "="*60)
        print("📊 ÉVALUATION DES MODÈLES")
        print("="*60)
        
        for name, model in self.models.items():
            results = self.evaluate_model(model, X_test, y_test, name)
            self.results[name] = results
            
            # Cross-validation
            cv_mean, cv_std = self.cross_validate(model, X_train, y_train, name)
            self.results[name]['cv_mean'] = cv_mean
            self.results[name]['cv_std'] = cv_std
    
    def get_best_model(self):
        """Obtenir le meilleur modèle selon l'AUC-ROC"""
        best_model_name = max(self.results, key=lambda x: self.results[x]['auc_roc'])
        best_model = self.models[best_model_name]
        best_score = self.results[best_model_name]['auc_roc']
        
        print(f"\n🏆 MEILLEUR MODÈLE: {best_model_name}")
        print(f"   AUC-ROC: {best_score:.4f}")
        
        return best_model_name, best_model
    
    def save_model(self, model, filepath):
        """Sauvegarder le modèle"""
        joblib.dump(model, filepath)
        print(f"✅ Modèle sauvegardé: {filepath}")
    
    def save_results(self, filepath):
        """Sauvegarder les résultats"""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=4)
        print(f"✅ Résultats sauvegardés: {filepath}")
    
    def print_summary(self):
        """Afficher un résumé des résultats"""
        print("\n" + "="*60)
        print("📈 RÉSUMÉ COMPARATIF")
        print("="*60)
        print(f"{'Modèle':<20} {'Accuracy':<12} {'AUC-ROC':<12} {'F1-Score':<12}")
        print("-"*60)
        
        for name, results in self.results.items():
            print(f"{name:<20} {results['accuracy']:<12.4f} {results['auc_roc']:<12.4f} {results['f1_score']:<12.4f}")


def main():
    """Pipeline principal"""
    
    # Préparation des données
    preprocessor = DataPreprocessor()
    try:
        X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline('data/churn_data.csv')
    except FileNotFoundError:
        print("❌ Fichier data/churn_data.csv non trouvé")
        print("⚠️  Téléchargez le dataset depuis Kaggle: https://www.kaggle.com/blastchar/telco-customer-churn")
        return
    
    # Entraînement des modèles
    trainer = ModelTrainer()
    trainer.train_all_models(X_train, y_train, X_test, y_test)
    
    # Obtenir le meilleur modèle
    best_model_name, best_model = trainer.get_best_model()
    
    # Sauvegarder
    trainer.save_model(best_model, 'models/best_model.pkl')
    trainer.save_results('results/model_performance.json')
    
    # Résumé
    trainer.print_summary()
    
    print("\n✅ Entraînement terminé!")


if __name__ == "__main__":
    main()
