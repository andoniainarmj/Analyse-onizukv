# Modèles Sauvegardés

Ce répertoire contient les modèles ML entraînés.

## Fichiers

### best_model.pkl
- **Modèle**: Meilleur modèle sélectionné
- **Format**: Pickle (joblib)
- **Utilisation**: Charger avec `joblib.load('best_model.pkl')`

## Chargement d'un modèle

```python
import joblib

model = joblib.load('models/best_model.pkl')

# Faire une prédiction
prediction = model.predict(X_test)
```

---

**Note**: Les fichiers `.pkl` ne sont pas versionnés (voir `.gitignore`).
