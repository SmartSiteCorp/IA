**SmartSite — IA de contrôle des travaux**

Le module vise à repérer des défauts visibles et des écarts géométriques, à les rattacher à une zone et à une tâche, puis à suivre leur correction.

Décision du 18 septembre 2026 : adapter **RF-DETR Seg Medium** pour les défauts sur photos ; utiliser **Open3D** pour les mesures 3D. Le périmètre actif concerne la détection d'anomalies. La prédiction d'avancement et de délais est reportée.

- [Registre des sources de données](config/datasets.json)

**État réel : première chaîne de préparation de données disponible.** L'archive de segmentation DamSegment v1 a été téléchargée, contrôlée par SHA-256 et importée localement : 1 500 images et 19 710 annotations. Aucun modèle n'est entraîné et aucun service d'inférence n'est encore livré.

L'import conserve les fichiers originaux, valide leurs correspondances, compare les annotations JSON/YOLO et produit un corpus COCO non partitionné avec un manifeste et un rapport. Les catégories restent `source_class_0` et `source_class_1` tant que leur correspondance sémantique n'est pas confirmée par une preuve documentée. Les différences entre les masques PNG fournis et la rasterisation COCO sont mesurées, sans correction silencieuse.

**Installation et vérification**

Python 3.11 et [uv](https://docs.astral.sh/uv/getting-started/installation/) sont nécessaires. Version d'uv utilisée pour le verrouillage : 0.12.16. Les dépendances sont verrouillées dans `uv.lock`.

```sh
uv sync --locked
uv run --no-sync ruff format --check .
uv run --no-sync ruff check .
uv run --no-sync mypy
uv run --no-sync pytest --cov --cov-report=term-missing
```

Les tests utilisent de petites données synthétiques, sans téléchargement ni GPU. Ils vérifient les comportements logiciels ; ils ne mesurent pas la qualité d'un modèle. La CI utilise les mêmes contrôles et n'a pas besoin de fichiers de suivi locaux.

**Préparer les données**

```sh
uv run --no-sync smartsite-data download --output data/raw/damsegment_v1/segmentation.zip
uv run --no-sync smartsite-data import data/raw/damsegment_v1/segmentation.zip --output data/processed/damsegment_v1
```

Le téléchargement est explicite, limité et vérifié contre la taille et l'empreinte publiées. Une archive existante est réutilisée uniquement si elle est intacte. Un téléchargement interrompu est supprimé et peut être relancé depuis le début. L'import refuse un dossier de sortie existant, les chemins dangereux, les archives excessives, les fichiers manquants/corrompus et les annotations incohérentes. Une préparation échouée ne publie pas de résultat partiel. Les dossiers parents des sorties doivent être des espaces locaux de confiance.

La sortie comprend `images/`, `masks/`, `source_annotations/`, `annotations.coco.json`, `manifest.json`, `report.json` et `ATTRIBUTION.json`. Les boîtes et surfaces COCO sont calculées avec `pycocotools` ; les coordonnées des polygones et les boîtes source sont conservées. Le résultat est un corpus d'audit, pas encore les dossiers `train/valid/test` d'un entraînement.

**Points de validation encore ouverts**

- Confirmer la signification des classes source et revoir les annotations douteuses, dont trois images sans instance annotée.
- Examiner les différences de rasterisation, particulièrement pour les fissures très fines ; une différence de bord ne démontre pas à elle seule une mauvaise annotation.
- Établir les groupes par photo/scène d'origine et rechercher les quasi-doublons avant toute séparation entraînement/validation/test. L'import recherche les doublons exacts de pixels ; cela ne remplace pas les groupes d'origine.
- Qualifier un jeu d'évaluation indépendant et représentatif des prises de vue SmartSite.

Les champs `scene_group` et `split` restent donc `null`, et `approved_for_training` reste `false`. Aucune précision de détection ne peut être déduite du succès de l'import.

**Données et attribution**

[DamSegment v1](https://data.mendeley.com/datasets/z5z6gtt5t4/1), DOI `10.17632/z5z6gtt5t4.1`, par Vahidreza Gharehbaghi, Caroline R. Bennett, Rémy Lequesne, Hang Zhao et Jian Li. La publication annonce [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) ; aucun fichier de licence distinct n'est inclus dans l'archive de segmentation inspectée. Le rapport conserve l'attribution et décrit les transformations effectuées par SmartSite.

Les images et sorties volumineuses ne sont pas distribuées avec le code. L'archive source demeure inchangée et permet de reproduire l'import.
