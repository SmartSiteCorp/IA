# SmartSite IA

Préparation contrôlée des données de défauts du bâtiment. Le projet télécharge, importe, inspecte et prépare les corpus ; aucun modèle n'est encore entraîné et aucun service d'inférence n'est livré.

Deux parcours sont disponibles :

- **DamSegment** : segmentation d'instances du béton, avec partitions COCO `train/valid/test` pour les premiers essais.
- **Concrete Crack Segmentation** : photos externes et masques sémantiques de fissures, conservés séparément de l'apprentissage et encore à qualifier pour l'évaluation.

## Installation et tests

Python 3.11 et [uv](https://docs.astral.sh/uv/getting-started/installation/) sont nécessaires. Les dépendances sont verrouillées dans `uv.lock` ; version d'uv utilisée : 0.12.16.

```sh
uv sync --locked
uv run --no-sync ruff format --check .
uv run --no-sync ruff check .
uv run --no-sync mypy
uv run --no-sync pytest --cov --cov-report=term-missing
```

Dans un environnement déjà installé, `uv run --no-sync smartsite-data` peut être remplacé par `.venv/bin/smartsite-data`. Les tests courants utilisent des fixtures synthétiques sans téléchargement ni GPU ; ils ne mesurent pas la précision d'une IA. La CI exécute les mêmes vérifications.

## Télécharger et importer DamSegment

```sh
uv run --no-sync smartsite-data download --output data/raw/damsegment_v1/segmentation.zip
uv run --no-sync smartsite-data import data/raw/damsegment_v1/segmentation.zip --output data/processed/damsegment_v1
```

Le téléchargement vérifie taille et SHA-256. L'import contrôle les chemins, les limites des archives, les images et les annotations JSON/YOLO. Il conserve les quatre fichiers source de chaque échantillon et produit un COCO non partitionné, un manifeste et un rapport. Les catégories de cet import restent `source_class_0` et `source_class_1` : il sert de référence immuable.

L'archive DamSegment v1 contient 1 500 photos 640 × 640 et 19 710 instances. Les images, annotations et masques d'origine restent inchangés.

## Examiner les annotations

```sh
uv run --no-sync smartsite-data review data/processed/damsegment_v1 --output artifacts/review/damsegment_v1
```

Le rapport HTML compare les masques fournis aux polygones source reconstruits avec `pycocotools`. **Il affiche des annotations existantes, pas des prédictions.** Il signale aussi les annotations vides, les classes fortement superposées et les images similaires.

Les similitudes utilisent un dHash de 128 bits, huit rotations/symétries et une comparaison couleur 32 × 32 : distance maximale de 8 bits, écart moyen de 20/255. Ce sont des seuils de présélection, pas des probabilités. La recherche est bornée à 5 000 images et 10 000 paires. Les variantes recadrées ou prises d'un autre point de vue peuvent lui échapper.

## Préparer les partitions d'apprentissage

```sh
uv run --no-sync smartsite-data prepare data/processed/damsegment_v1 \
  --policy config/damsegment_review.json \
  --output data/prepared/damsegment_v1
```

La politique versionnée est liée à l'empreinte exacte du manifeste. Elle retient, pour SmartSite, `0 → crack` (fissure) et `1 → surface_loss` (perte de matière). Cette correspondance est une **interprétation de projet documentée après inspection de 31 images par classe** dans les trois difficultés. La publication annonce fissures et éclatement du béton, mais ne fournit pas de table numérique explicite vérifiée ; ce choix ne doit pas être présenté comme une confirmation des auteurs ni comme une validation de chaque annotation.

La commande :

1. Relit les fichiers et vérifie leurs empreintes ; reconstruit les instances depuis les polygones source.
2. Met à part `easy_0104`, `easy_0219`, `easy_0224` et `easy_0216`, avec leur motif. Les trois annotations vides ne deviennent pas des exemples « sans défaut ».
3. Applique les 12 regroupements revus et recalcule les similitudes. Un groupe lié à une image mise à part est exclu en entier.
4. Répartit les groupes avec une graine fixe (`--seed`, défaut `20260918`) et des strates de difficulté/présence des classes. Cible : environ 80/10/10 ; un groupe n'est jamais coupé. Les petites strates restent dans l'apprentissage. Les trois partitions doivent être non vides et contenir les classes observées.
5. Copie les photos sans réencodage ; conserve les polygones et leurs identifiants source. Les catégories COCO sont `1=crack` et `2=surface_loss`.

Résultat observé sur la politique livrée :

| Destination | Photos | Instances |
|---|---:|---:|
| `train` | 1 198 | 15 685 |
| `valid` | 149 | 1 950 |
| `test` | 149 | 2 072 |
| `quarantine` | 4 | Non utilisées |

Chaque partition contient ses photos et `_annotations.coco.json`, selon la structure documentée par [RF-DETR](https://rfdetr.roboflow.com/learn/train/). Le format a été contrôlé avec `pycocotools` et le chargeur publié de RF-DETR 1.10.1 a été inspecté ; aucun entraînement RF-DETR n'a été exécuté à cette étape.

`manifest.json`, `report.json`, `policy.json`, `ATTRIBUTION.json` et `index.html` conservent les décisions, empreintes, comptes et limites. Les fichiers masques source restent dans le corpus importé ; l'export utilise les instances COCO, qui préservent les recouvrements entre classes.

**Usage : entraînement expérimental uniquement.** Toutes les photos DamSegment proviennent d'un barrage et les identifiants des photos/scènes d'origine manquent. Les groupes de contenu évitent les fuites connues, sans garantir l'absence de scènes communes. Les scores internes ne prouvent pas une généralisation à d'autres ouvrages ni la performance sur des chantiers SmartSite. Le test ne doit pas servir à choisir le modèle ou ses seuils.

## Préparer les photos externes

```sh
uv run --no-sync smartsite-data download \
  --dataset concrete_crack_segmentation_v1 \
  --output data/raw/concrete_crack_segmentation_v1/concreteCrackSegmentationDataset.rar
```

`prepare-external` lit les fichiers extraits. Pour une nouvelle installation, extraire **l'archive vérifiée ci-dessus** avec un outil compatible RAR dans un dossier vide nommé `data/raw/concrete_crack_segmentation_v1/extracted`. On doit y trouver `rgb/` et `BW/`. Sur macOS, l'outil système `tar` utilisé lors de l'inspection sait lire ce RAR. Les empreintes des 916 fichiers sont ensuite vérifiées contre `config/ccsd_preparation.json` ; l'import ZIP DamSegment ne lit pas le RAR.

```sh
uv run --no-sync smartsite-data prepare-external data/raw/concrete_crack_segmentation_v1 \
  --policy config/ccsd_preparation.json \
  --output data/prepared/ccsd_v1
```

Les 458 photos sont orientées selon leur EXIF puis enregistrées en PNG, à résolution native, sans nouvelle compression avec perte. Les masques JPEG passent en niveaux de gris : valeur ≥ 128 = fissure blanche, sinon fond noir. Aucune dilatation, fermeture de trous ou création d'instances n'est effectuée. Une bande séparée et des comptes aux seuils 96/160 documentent la sensibilité au seuil, qui est une décision de préparation du projet.

Les similitudes sont recalculées après orientation. Chaque paire est comparée à résolution native avec sa rotation/symétrie, sur la photo et le masque. Les paramètres de revue sont dans la politique : écart RGB moyen maximal de 2/255 et IoU des masques minimal de 0,98. Ces seuils conservateurs demandent une revue ; ils ne prouvent pas que toutes les annotations écartées sont fausses.

Sur cette source, 199 paires forment 259 groupes de contenu. Les masques de plusieurs variantes diffèrent réellement : par exemple, `061/549` remplissent différemment une zone entre deux branches de fissure. Avec la politique livrée, 195 groupes (390 photos) sont mis à part ; 64 références restent candidates et quatre vues similaires supplémentaires sont conservées pour des essais de robustesse. Tous les originaux et les dérivés sont conservés.

La réserve externe reste **non approuvée pour l'apprentissage et pour une évaluation finale**. Les masques sont sémantiques : comparer ultérieurement leur union aux prédictions de fissures, sans inventer d'instances. La revue d'alignement complète, les cas négatifs difficiles et la qualification chantier restent à faire. Les groupes ne constituent pas des identifiants de bâtiments. Ne pas entraîner sur le dataset apparenté `5y9wdsg2zt` : ses petits extraits proviennent de ces mêmes photos.

## Ouvrir les rapports

```sh
.venv/bin/python -m http.server 8767 --bind 127.0.0.1 --directory data/prepared
```

Ouvrir [le corpus DamSegment](http://127.0.0.1:8767/damsegment_v1/) ou [la réserve externe](http://127.0.0.1:8767/ccsd_v1/). Garder le terminal ouvert ; `Ctrl+C` arrête le serveur. Le HTML fonctionne aussi hors ligne. Il permet d'ouvrir les photos, masques, groupes et motifs d'exclusion ; il ne modifie pas les décisions.

## Intégrité et reproductibilité

Choisir un nouveau dossier de sortie à chaque exécution : aucune sortie existante n'est écrasée. Une erreur ne publie pas de dossier partiel. Les chemins sortants, liens symboliques, fichiers altérés, décisions non compatibles et entrées excessives sont refusés. Les dossiers parents doivent être des espaces locaux de confiance. Les originaux ne sont jamais supprimés ; la graine, les paramètres et les empreintes des politiques rendent les décisions traçables.

Les tests couvrent notamment les groupes transitifs, les doubles présents dans plusieurs difficultés, la propagation des exclusions, les corruptions, l'orientation EXIF, les seuils de masques, les conflits entre variantes et la reproductibilité des sorties.

## Attribution

- [DamSegment v1](https://data.mendeley.com/datasets/z5z6gtt5t4/1), DOI `10.17632/z5z6gtt5t4.1`, Vahidreza Gharehbaghi, Caroline R. Bennett, Rémy Lequesne, Hang Zhao, Jian Li.
- [Concrete Crack Segmentation Dataset v1](https://data.mendeley.com/datasets/jwsn7tfbrp/1), DOI `10.17632/jwsn7tfbrp.1`, Çağlar Fırat Özgenel.

Les publications annoncent CC BY 4.0. Les exports conservent l'attribution et la description des modifications. Les images, archives et sorties volumineuses ne sont pas distribuées avec le code.
