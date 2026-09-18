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

**Revoir les annotations et les images proches**

```sh
uv run --no-sync smartsite-data review data/processed/damsegment_v1 --output artifacts/review/damsegment_v1
```

Dans un environnement déjà installé, `uv run --no-sync smartsite-data` peut être remplacé par `.venv/bin/smartsite-data`. Ouvrir ensuite `index.html` dans le dossier de sortie avec un navigateur. Aucun serveur n'est nécessaire. Choisir un nouveau dossier à chaque exécution : aucune sortie existante n'est écrasée.

La revue vérifie les quatre fichiers de chaque image contre les empreintes du manifeste. Elle reconstruit les polygones source avec COCO, compare les pixels aux masques PNG et mesure séparément la superposition des deux classes avant qu'une couleur en masque une autre. Elle ne valide pas les noms des classes et n'approuve pas automatiquement les annotations. La référence revue est le JSON source, pas un fichier COCO modifié après l'import.

La recherche de similitude utilise un dHash horizontal/vertical de 128 bits, huit rotations/symétries et une comparaison couleur à 32 × 32. Seuils par défaut : distance ≤ 8 bits et écart moyen ≤ 20 sur 255 ; options `--max-hamming-distance` et `--max-pixel-error`. Ce sont des seuils de présélection à examiner, pas des probabilités. Les groupes relient les paires candidates ; ils ne reconstituent pas les scènes d'origine. Les recadrages, vues voisines et variations importantes peuvent échapper au contrôle.

La page affiche des vignettes pour toutes les images et des vues détaillées pour les cas prioritaires et les paires proches. `review.json` contient toutes les mesures ; `review-decisions.template.json` fournit une fiche à compléter avec auteur et justification. Les seuils de recouvrement (0,85) et de superposition (0,90 de la plus petite classe) servent à demander une revue. Aucune photo n'est supprimée ni étiquette corrigée. La comparaison est bornée à 5 000 images et 10 000 paires ; les limites produisent une erreur, jamais une liste tronquée silencieusement.

**Complément réservé à l'évaluation des fissures**

```sh
uv run --no-sync smartsite-data download \
  --dataset concrete_crack_segmentation_v1 \
  --output data/raw/concrete_crack_segmentation_v1/concreteCrackSegmentationDataset.rar
```

[Concrete Crack Segmentation Dataset v1](https://data.mendeley.com/datasets/jwsn7tfbrp/1), par Çağlar Fırat Özgenel, DOI `10.17632/jwsn7tfbrp.1`, licence publiée CC BY 4.0 : 458 photos de bâtiments avec 458 masques. L'archive RAR de 745 914 150 octets est épinglée par SHA-256. Elle a été téléchargée et inspectée localement ; la commande `import` reste dédiée au ZIP DamSegment et n'importe pas ce RAR.

Les originaux sont conservés séparément. Les masques sont des JPEG avec des niveaux intermédiaires : il reste à fixer une règle de binarisation et à qualifier leur alignement. Les photos utilisent trois orientations EXIF ; les 458 paires ont des dimensions cohérentes après application de ces orientations, ce qui ne prouve pas à lui seul la justesse des annotations. Ce complément reste un candidat d'évaluation des fissures, sans approbation d'apprentissage ni de recette SmartSite. Les extraits de classification du jeu apparenté `5y9wdsg2zt` proviennent de ces mêmes photos : les utiliser en entraînement compromettrait cette réserve d'évaluation.

**Points de validation encore ouverts**

- Confirmer la signification des classes source et revoir les annotations douteuses, dont trois images sans instance annotée.
- Les différences de rasterisation observées restent dans un voisinage d'un pixel des contours. Résoudre les annotations ambiguës et la superposition des classes ; une différence de bord ne démontre pas à elle seule une mauvaise annotation.
- Établir les groupes par photo/scène d'origine et appliquer les décisions sur les paires proches avant toute séparation entraînement/validation/test. Les 12 paires repérées ne démontrent pas l'absence d'autres scènes partagées.
- Qualifier un jeu d'évaluation indépendant et représentatif des prises de vue SmartSite.

Les champs `scene_group` et `split` restent donc `null`, et `approved_for_training` reste `false`. Aucune précision de détection ne peut être déduite du succès de l'import.

**Données et attribution**

[DamSegment v1](https://data.mendeley.com/datasets/z5z6gtt5t4/1), DOI `10.17632/z5z6gtt5t4.1`, par Vahidreza Gharehbaghi, Caroline R. Bennett, Rémy Lequesne, Hang Zhao et Jian Li. La publication annonce [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) ; aucun fichier de licence distinct n'est inclus dans l'archive de segmentation inspectée. Le rapport conserve l'attribution et décrit les transformations effectuées par SmartSite.

Les images et sorties volumineuses ne sont pas distribuées avec le code. L'archive source demeure inchangée et permet de reproduire l'import.
