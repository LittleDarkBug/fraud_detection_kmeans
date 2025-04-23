# Analyse des Transactions Financières avec K-means

Ce projet permet d'analyser des comportements financiers et de détecter des anomalies dans les transactions bancaires à l'aide de l'algorithme de clustering K-means.

## Structure du Projet

```
PROJET KMEANS/
│
├── data/                          # Dossier contenant les données
│
├── functions/                     # Modules fonctionnels
│   ├── apply_kmeans.py            # Application de l'algorithme K-means
│   ├── console_utils.py           # Utilitaires pour l'interface console
│   ├── detect_anomalies.py        # Fonctions de détection d'anomalies
│   ├── load_data.py               # Chargement des données
│   ├── normalize_features.py      # Normalisation des caractéristiques
│   ├── plot_clusters.py           # Visualisation des clusters
│   ├── select_features.py         # Sélection des caractéristiques
│   └── threshold_utils.py         # Utilitaires de définition des seuils
│
├── report/                        # Dossier contenant les rapports
├   ├── Rapport Projet Kmeans....      # Rapport du projet (PDF)
│
├── main.py                        # Point d'entrée du programme
├── README.md                      # Ce fichier
├── requirements.txt               # Dépendances du projet
```

## Installation

### 1. Création d'un environnement virtuel

#### Avec venv

```bash
# Création de l'environnement virtuel
python -m venv .venv

# Activation de l'environnement
# Sur Windows:
.venv\Scripts\activate

# Sur macOS et Linux:
source .venv/bin/activate
```


### 2. Installation des dépendances

Une fois l'environnement virtuel activé, installez les dépendances nécessaires:

```bash
pip install -r requirements.txt
```

## Utilisation

Pour lancer le programme d'analyse:

```bash
python main.py
```

### Programme Interactif

⚠️ **Attention**: Ce programme est interactif et nécessite des actions de l'utilisateur à différentes étapes de l'analyse.

1. À chaque étape majeure, le programme vous demandera de confirmer pour continuer.
2. Certaines visualisations nécessitent que vous les fermiez manuellement pour passer à l'étape suivante.

### Étapes du programme:

1. Chargement et prétraitement des données
2. Exploration et visualisation des caractéristiques
3. Application de l'algorithme K-means
4. Analyse des clusters et détection d'anomalies
5. Génération de rapports et recommandations

À chaque étape, suivez les instructions affichées dans la console.

## Rapport

Un rapport détaillé du projet est disponible dans le fichier "Rapport Projet Kmeans...." à la racine dans le Dossier report.