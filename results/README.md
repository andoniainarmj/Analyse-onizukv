# Résultats

Ce répertoire contient les résultats et les métriques des modèles.

## Fichiers

### model_performance.json
Contient les résultats d'évaluation de tous les modèles.

**Structure**:
```json
{
  "Logistic Regression": {
    "accuracy": 0.8,
    "precision": 0.65,
    "recall": 0.4,
    "f1_score": 0.5,
    "auc_roc": 0.82,
    "confusion_matrix": [[...], [...]],
    "classification_report": {...}
  },
  ...
}
```

---

**Note**: Les fichiers de résultats ne sont pas versionés (voir `.gitignore`).
