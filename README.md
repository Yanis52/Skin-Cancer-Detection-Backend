# Skin-Cancer-Detection-Backend


Ce projet est une API Flask déployable via Docker, permettant de faire des prédictions de types de cancer de la peau à partir d'images dermatoscopiques, grâce à un modèle CNN avec module d’attention CBAM entraîné sur le dataset public ISIC (Kaggle).

---

##  Liens utiles

-  Dataset utilisé : [Kaggle - Skin Cancer 9 Classes (ISIC)](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic)
-  Notebook d’entraînement (CBAM) : [`cbam_skin_cancer_model.ipynb`](../notebooks/)
-  API Flask backend : `app.py`
-  Docker compatible

---
modèle trop lourd pour etre present dans le repo (+100m)

## Utilisation (sans docker)
``` 
pip install requirements.txt
python app.py 
```

##  Utilisation (via Docker)

### 1. Construction de l'image
```
bash
docker build -t cbam-api .
```
