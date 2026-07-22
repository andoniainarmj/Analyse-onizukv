# Dataset

## Customer Churn Dataset

Ce répertoire contient le dataset utilisé pour le projet de prédiction du churn client.

### Téléchargement

**Source**: Kaggle - Telecom Customer Churn
**URL**: https://www.kaggle.com/blastchar/telco-customer-churn

### Instructions

1. Créer un compte Kaggle (si ce n'est pas déjà fait)
2. Télécharger le fichier `WA_Fn-UseC_-Telco-Customer-Churn.csv`
3. Renommer en `churn_data.csv`
4. Placer dans ce répertoire (`data/`)

### Contenu

Le dataset contient:
- ~7,000 clients
- 20+ variables
- Information sur:
  - Données démographiques
  - Services souscripts
  - Informations de compte
  - **Target**: Churn (Yes/No)

### Structure

```
churn_data.csv
├── customerID
├── gender
├── SeniorCitizen
├── Tenure
├── PhoneService
├── InternetService
├── OnlineSecurity
├── OnlineBackup
├── DeviceProtection
├── TechSupport
├── StreamingTV
├── StreamingMovies
├── Contract
├── InternetType
├── PaperlessBilling
├── PaymentMethod
├── MonthlyCharges
├── TotalCharges
└── **Churn** (Target)
```

---

**Note**: Le fichier `churn_data.csv` n'est pas versionné (voir `.gitignore`). Chaque utilisateur doit le télécharger.
