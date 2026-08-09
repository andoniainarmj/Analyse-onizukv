import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

class DataPreprocessor:
    
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def load_data(self, filepath):
        print(f"Chargement des données depuis {filepath}...")
        df = pd.read_csv(filepath)
        print(f"Données chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes")
        return df
    
    def clean_data(self, df):
        print("Nettoyage des données...")
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)
        if df.isnull().sum().sum() > 0:
            print(f"Valeurs manquantes détectées: {df.isnull().sum().sum()}")
            df = df.dropna()
        numeric_cols = df.select_dtypes(include=['object']).columns
        for col in numeric_cols:
            if df[col].str.replace(' ', '').str.replace('.', '', 1).str.isdigit().all():
                df[col] = pd.to_numeric(df[col], errors='coerce')
        print("Nettoyage terminé")
        return df
    
    def encode_categorical(self, df, target_col='Churn'):
        print("Encodage des variables catégories...")
        df_encoded = df.copy()
        categorical_cols = df_encoded.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            if col == target_col:
                df_encoded[col] = df_encoded[col].map({'No': 0, 'Yes': 1})
            else:
                df_encoded = pd.get_dummies(df_encoded, columns=[col], drop_first=True)
        print(f"Encodage terminé: {df_encoded.shape[1]} features")
        return df_encoded
    
    def split_data(self, X, y):
        print(f"Division des données (test_size={self.test_size})...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train, X_test):
        print("Normalisation des features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        print("Normalisation terminée")
        return X_train_scaled, X_test_scaled
    
    def balance_data(self, X_train, y_train):
        print("Équilibrage des classes avec SMOTE...")
        smote = SMOTE(random_state=self.random_state)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        print(f"Classes équilibrées: {y_train_balanced.value_counts().to_dict()}")
        return X_train_balanced, y_train_balanced
    
    def preprocess_pipeline(self, filepath):
        print("PIPELINE DE PRÉPARATION DES DONNÉES")
        df = self.load_data(filepath)
        df = self.clean_data(df)
        df = self.encode_categorical(df)
        target_col = 'Churn'
        if target_col not in df.columns:
            raise ValueError(f"Colonne cible '{target_col}' non trouvée")
        y = df[target_col]
        X = df.drop(target_col, axis=1)
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        X_train_balanced, y_train_balanced = self.balance_data(X_train_scaled, y_train)
        print("PIPELINE TERMINÉ")
        return X_train_balanced, X_test_scaled, y_train_balanced, y_test
