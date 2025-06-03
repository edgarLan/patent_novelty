import os

# Racine de ton projet
root = "patent_novelty"

# Arborescence des dossiers à créer
folders = [
    "metrics/metrics_raw",
    "metrics/top10",
    "metrics/cd_index",
    "metrics/metricAnalysis",
    "data/app_pat_match",
    "data/citationsData",
    "data/compressedData",
    "data/csv_clean/ES",
    "data/csv_clean/KS",
    "data/csv_clean/tE",
    "data/csv_raw/ES/text",
    "data/csv_raw/KS",
    "data/csv_raw/tE",
    "data/jsonData",
    "data/vocab/technet from github",
    "data/vocab/additional stopwords"
]

# Création des dossiers
for folder in folders:
    path = os.path.join(root, folder)
    os.makedirs(path, exist_ok=True)

print("Structure créée avec succès.")
