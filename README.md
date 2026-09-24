# SmartSite IA

Préparation contrôlée des données et premiers essais de détection des défauts du bâtiment. Le projet télécharge, importe, inspecte et prépare les corpus, puis entraîne RF-DETR et affiche ses prédictions. Le premier modèle est expérimental ; aucun service chantier ni détecteur qualifié n'est encore livré.

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

## Préparer les nouvelles surfaces et leurs annotations

La sélection `config/moisture_collection_v1.json` décrit 71 photos PHELE v3/Commons,
avec origine, licence, empreinte et groupes de précaution. Les cadres fournis par
PHELE sont des **annotations de dataset**, pas des prédictions de notre modèle.
Le téléchargement est séquentiel et reprend les fichiers déjà vérifiés ; une erreur
réseau arrête la collecte. La préparation suivante fonctionne ensuite hors réseau :

```sh
.venv/bin/python -m smartsite_ia.collection fetch \
  --selection config/moisture_collection_v1.json \
  --cache data/raw/moisture_collection_v1/assets

.venv/bin/python -m smartsite_ia.collection_review \
  --selection config/moisture_collection_v1.json \
  --cache data/raw/moisture_collection_v1/assets \
  --review config/moisture_review_v1.json \
  --output artifacts/predictions/moisture_annotations_v1
```

Choisir un dossier de sortie neuf. Après installation du paquet, la deuxième commande
est aussi disponible sous le nom `smartsite-review-collection`.
La [galerie locale des corrections](http://127.0.0.1:8768/moisture_annotations_v1/)
s'ouvre avec le serveur de rapports sur le port 8768 décrit plus bas.

Cette revue visuelle est réalisée **par l'assistant**, sans validation experte ni
confirmation terrain. Chaque photo a une décision motivée ; chaque ancien cadre
est conservé, modifié ou retiré explicitement. Le dossier `source/` garde la collecte
complète, les originaux, les annotations et les attributions. Les corrections sont
séparées dans `review.json` et `report.json`.

Résultat du manifeste livré : 38 photos candidates et 33 écartées. Les 91 cadres
source ont 32 décisions de conservation, 13 de modification et 46 de retrait du lot
pilote, souvent par prudence plutôt que parce que l'annotation serait prouvée fausse.
Les sept Commons ont des propositions de rectangles ; deux restent exclues du pilote.
Deux recadrages revus (motif du sol et texture de plafond) constituent des négatifs
pour les trois cibles seulement ; ce ne sont pas de nouvelles scènes indépendantes.

| Lot pilote | Images | Groupes de précaution | Boîtes |
|---|---:|---:|---:|
| `train` | 32, dont 2 recadrages | 21 | 45 |
| `validation` | 8 | 5 | 13 |

Chaque lot contient ses JPEG et `_annotations.coco.json` : `1=mold_suspected`,
`2=moisture_trace`, `3=peeling_paint`. Coordonnées COCO en pixels `x, y, largeur, hauteur`
dans la photo orientée ; aire du rectangle, aucun masque inventé. Les sous-types
`surface_loss` et `peeling_paint` partagent une famille d'affichage `surface_damage`,
mais les classes et poids de l'ancien modèle restent inchangés.

**Ce corpus permet de préparer un essai de détection par boîtes, pas de qualifier
la fonctionnalité.** Il n'est pas compatible tel quel avec la commande actuelle
`smartsite-model train`, dédiée au corpus DamSegment et à ses masques. Le prochain
parcours est `smartsite-boxes`, décrit ci-dessous : il vérifie les empreintes,
charge ces boîtes et respecte ces groupes, sans lire les exclusions comme du fond sain.

Limites : seulement quatre groupes de moisissures dans l'apprentissage et un dans
la validation ; deux groupes de traces d'eau de chaque côté ; aucun exemple négatif
complet dans la validation. Les bâtiments PHELE sont souvent inconnus et les groupes
restent prudents. Aucun test final créé, aucune nouvelle performance mesurée, aucun
entraînement lancé. Les annotations proposent des signes visibles, jamais un diagnostic
d'humidité, de gravité ou une preuve de conformité. La convention complète figure
dans le manifeste de revue et le rapport exporté.

## Entraîner le pilote par rectangles

Ce parcours utilise **RF-DETR Nano 1.10.1**, en détection à 384 pixels, avec les
poids COCO officiels distincts du modèle historique de segmentation. Il apprend les
trois catégories de la revue ; il ne fusionne pas les poids fissures/perte de matière.
Le moteur et les poids Nano sont publiés sous Apache 2.0 selon la
[documentation auteur](https://rfdetr.roboflow.com/latest/learn/run/detection/).
Les licences des photos restent conservées dans les exports.

Prérequis : extra `training` installé (`uv sync --locked --extra training`) et
corpus de la section précédente présent. Les poids officiels peuvent être récupérés
explicitement ; leur taille et leur SHA-256 sont vérifiés, puis tous les calculs
utilisent les fichiers locaux :

```sh
.venv/bin/python -m smartsite_ia.box_cli weights \
  --output artifacts/models/pretrained/rf-detr-nano.pth

PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m smartsite_ia.box_cli check \
  artifacts/predictions/moisture_annotations_v1 \
  --config config/moisture_boxes_v1.json \
  --weights artifacts/models/pretrained/rf-detr-nano.pth --device mps
```

`check` vérifie les entrées et le matériel sans entraîner. Sur ce Mac, poids et corpus
sont déjà préparés. Le premier essai fixé dans `config/moisture_boxes_v1.json` utilise
5 passages, 32 images train (dont 2 recadrages), 8 images de validation, batch 2,
accumulation 2 et graine 42. Aucun test réservé n'est lu. Le dernier passage est retenu,
sans recherche du meilleur passage ni promotion automatique :

```sh
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m smartsite_ia.box_cli train \
  artifacts/predictions/moisture_annotations_v1 \
  --config config/moisture_boxes_v1.json \
  --weights artifacts/models/pretrained/rf-detr-nano.pth \
  --device mps --output artifacts/training/moisture_boxes_v1
```

Sur un autre matériel, choisir explicitement `--device cpu` ou `--device cuda`.
Ne pas annoncer la durée ou la précision d'un autre matériel sans mesure.
L'exécutable installé `smartsite-boxes` accepte les mêmes arguments.

Le dossier doit être neuf. Les copies vérifiées de `train` et `valid`, les attributions
et les groupes sont conservés sous `data/`. `run.json` indique l'état réel, les
versions, empreintes, métriques finales et limites. `checkpoints/metrics.csv` conserve
les mesures de chaque passage : AP des boîtes, AP par catégorie, pertes et métriques
natives du moteur. Son F1 utilise le seuil choisi par l'évaluateur RF-DETR ; ce n'est
pas un résultat au score fixe 0,30 ni un seuil chantier. Les AP de boîtes ne sont pas
directement comparables aux anciennes AP de masques.

`checkpoints/checkpoint_best_total.pth` contient ici les poids du **dernier passage**,
malgré son nom imposé par le moteur. `checkpoints/last.ckpt` conserve l'état complet
pour une interruption. Après un arrêt enregistré dans `run.json`, reprendre avec la
même commande en ajoutant `--resume` : données, configuration, périphérique, versions
et checkpoint doivent être inchangés. Si aucun checkpoint vérifié n'existe (arrêt très
précoce ou arrêt brutal non enregistré), conserver le dossier et choisir une nouvelle
sortie pour recommencer. Un essai terminé n'est pas prolongé par `--resume` ; deux
processus ne peuvent pas écrire simultanément dans ce même essai.

Ce lancement est un pilote technique sur un petit corpus revu par l'assistant,
**sans validation experte ni qualification chantier**. Aucun négatif complet dans la
validation ; seulement un groupe de moisissures et deux groupes de traces d'eau.
Examiner ensuite les résultats par classe et les erreurs visuelles avant de décider
d'une suite ; aucune réussite logicielle ne prouve la qualité de ces détections.

## Premier entraînement et prédictions

Le moteur est une dépendance **optionnelle** : les commandes de données et les tests habituels ne nécessitent ni PyTorch ni GPU. Pour l'apprentissage, installer l'extra verrouillé :

```sh
uv sync --locked --extra training
.venv/bin/smartsite-model doctor --device mps
.venv/bin/smartsite-model weights --output artifacts/models/pretrained/rf-detr-seg-medium.pt
```

`mps` désigne le GPU Apple Silicon. `cpu` et `cuda` sont aussi acceptés explicitement, mais seul le parcours MPS a été exécuté pour ce premier apprentissage. Une machine sans le périphérique demandé produit une erreur. Le moteur retenu est `RFDETRSegMedium`, paquet `rfdetr[train]==1.10.1`. Le téléchargement des poids officiels est borné et leur SHA-256 épinglé ; le chargement n'autorise pas une désérialisation Python sans restriction. [Distribution RF-DETR](https://pypi.org/project/rfdetr/1.10.1/).

Le premier essai utilise `config/train_smoke.json` : 32 photos d'apprentissage, 8 de validation, trois époques, résolution du modèle 432 × 432. La sélection est déterministe, stratifiée par difficulté et présence des classes, avec conservation des groupes entiers. Elle copie uniquement les images et annotations de `train` et `valid`, vérifie leurs empreintes et ne lit pas les fichiers du test réservé. Le plafond de photos peut donner un compte inférieur lorsqu'un groupe entier ne tient plus.

```sh
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/smartsite-model train data/prepared/damsegment_v1 \
  --config config/train_smoke.json \
  --weights artifacts/models/pretrained/rf-detr-seg-medium.pt \
  --output artifacts/training/damsegment_smoke_v1 \
  --device mps
```

Le dossier de sortie doit être nouveau. Sur Mac, la variable ci-dessus autorise un repli CPU pour les opérations MPS non prises en charge ; sa présence est enregistrée, sans prétendre que toutes les opérations sont exécutées sur GPU. Aucun journal cloud ni image n'est envoyé à un service. L'exécution est en pleine précision, lot de 1, sans processus de chargement parallèles ; ce profil vérifie le parcours, pas la vitesse maximale du moteur.

Le dossier contient la sélection exacte, `run.json`, les checkpoints et `checkpoints/metrics.csv`. `run.json` distingue préparation, apprentissage, succès, échec et interruption ; une erreur laisse les fichiers de diagnostic disponibles. Il enregistre configuration, versions, périphérique, empreintes, durée et limites. `checkpoint_best_total.pth` est choisi sur la validation. `last_epoch_metrics` décrit la dernière époque et ne doit pas être confondu avec les métriques du checkpoint choisi. Les poids de départ et ceux appris sont distincts. Les checkpoints complets sont conservés. `train --resume` permet de reprendre un essai enregistré comme interrompu ou échoué, dans le même dossier, après vérification de la configuration, du périphérique et des entrées. Il ne prolonge pas un essai terminé et ne récupère pas automatiquement une coupure brutale sans état sauvegardé.

Premier essai : 32 photos et 308 instances pour apprendre, 8 photos et 70 instances pour valider ; 96 mises à jour, environ 107 secondes pour préparation/apprentissage/écriture après le contrôle du moteur. Cette durée ne comprend pas installation, téléchargement ou génération de la galerie ; elle ne prédit pas le temps d'un apprentissage complet. Le pic de mémoire n'a pas été mesuré.

Analyser ensuite une photo avec les poids appris :

```sh
.venv/bin/smartsite-model predict data/prepared/damsegment_v1/valid/easy_0023.jpg \
  --run artifacts/training/damsegment_smoke_v1 \
  --output artifacts/predictions/ma_photo \
  --device mps
```

Remplacer le premier chemin par une photo JPEG ou PNG RGB/grise, non animée, jusqu'à 8 Mio et 16 mégapixels. L'orientation EXIF est appliquée avant le calcul. Le résultat fournit la photo orientée, les prédictions dessinées, un JSON avec classes/scores/boîtes/masques COCO RLE et un HTML. Les coordonnées correspondent à la photo orientée en taille originale. Les labels internes du réseau sont `0=crack`, `1=surface_loss`, issus des catégories COCO `1` et `2`. Aucun calcul de largeur physique, gravité ou conformité n'est effectué.

Le seuil d'affichage par défaut `0.3` est explicite et **non calibré**. Les huit photos de validation de ce premier essai ont produit huit propositions de perte de matière réparties sur quatre photos, aucune proposition de fissure à ce seuil. Les quatre autres n'ont aucune proposition. Cela valide l'exécution du parcours, **pas la fiabilité du détecteur de fissures**. L'apprentissage plus long, le contrôle des petites fissures et l'évaluation indépendante restent nécessaires. Aucun entraînement sur les photos externes ni réglage à partir du test réservé n'a eu lieu.

Pour voir les résultats locaux déjà générés :

```sh
.venv/bin/python -m http.server 8768 --bind 127.0.0.1 --directory artifacts/predictions
```

[Galerie des huit photos de validation](http://127.0.0.1:8768/smoke_validation/) : photo, annotations fournies et prédictions, sans sélection des meilleurs exemples. [Exemple de résultat détaillé](http://127.0.0.1:8768/first_validation/). Ces sorties restent locales ; elles ne sont pas incluses dans un clone. Un nouveau `predict` crée son propre `index.html` dans le dossier choisi.

Les graines, données, versions et poids sont enregistrés pour reproduire le protocole. Les calculs GPU ne sont pas garantis identiques bit pour bit. Les tests du dépôt vérifient les contrats avec un moteur simulé aux frontières ; les essais réels sont documentés séparément, sans imposer l'installation du moteur à la CI légère.

## Évaluer les contours et comparer les réglages

`evaluate` analyse toute la validation épinglée par le run, sans lire les fichiers du test réservé. Le modèle reste expérimental. `config/train_full.json` décrit l'essai de deux époques sur 1 198 photos train / 149 valid ; `config/train_profile.json` fournit le profil court à mesurer avant un apprentissage long.

Le comportement par défaut est conservé : `--preprocessing public-v1 --mask-threshold 0.5`. L'option `training-v1` réutilise les transformations déterministes de validation du moteur installé, puis son décodage vers la taille originale de la photo. Cet adaptateur est limité au modèle SmartSite non optimisé et à RF-DETR 1.10.1 ; il ne modifie ni la bibliothèque ni les poids. Les dépendances du moteur restent optionnelles.

Deux seuils distincts :

- `--threshold` de `predict` filtre les **propositions de défauts** (0,3 par défaut).
- `--mask-threshold` choisit les **pixels du défaut**, après interpolation des logits vers la photo originale (0,5 par défaut). Un autre seuil exige le profil `training-v1`.

Les réglages sont enregistrés dans chaque résultat. Ils ne sont pas des mesures de gravité ni des probabilités calibrées de conformité. Ne pas appliquer un réglage sélectionné sur un checkpoint à tous les modèles sans nouvelle évaluation.

```sh
.venv/bin/smartsite-model evaluate data/prepared/damsegment_v1 \
  --run artifacts/training/damsegment_full_v1 \
  --output artifacts/predictions/validation_reference --device mps

.venv/bin/smartsite-model evaluate data/prepared/damsegment_v1 \
  --run artifacts/training/damsegment_full_v1 \
  --output artifacts/predictions/validation_alignee --device mps \
  --preprocessing training-v1 --mask-threshold 0.5

.venv/bin/smartsite-model compare artifacts/predictions/validation_reference \
  artifacts/predictions/validation_alignee --output artifacts/predictions/comparaison
```

Les dossiers de sortie doivent être nouveaux. Le comparatif refuse des photos, annotations ou règles de mesure différentes ; les réglages d'inférence peuvent changer, car c'est précisément l'objet de cette expérience. Les références restent à leur résolution native. Les métriques internes d'entraînement, qui utilisent des références redimensionnées, ne sont pas interchangeables avec ce rapport.

L'évaluation conserve l'AP COCO des masques et les comptages fixes (score ≥ 0,3, IoU ≥ 0,5). Un diagnostic supplémentaire classe les propositions non appariées en doublons, recouvrements insuffisants et absence de recouvrement avec une référence de la même classe. Un recouvrement insuffisant ne prouve pas un défaut réel : contour, découpage des instances, annotation ou détection peuvent être en cause.

Sur les 149 photos internes avec les poids de l'essai complet, aucun des profils alignés testés à 0,4 / 0,5 / 0,6 / 0,7 n'améliore les fissures par rapport à la référence (915 instances retrouvées, AP 0,091842). Le meilleur candidat en retrouve 909 (AP 0,090689). Le traitement actuel reste donc le réglage par défaut. Les candidats et leurs régressions restent consultables dans le [rapport local des essais](http://127.0.0.1:8768/inference_study_v1/) après génération locale ; ces artefacts ne sont pas distribués dans Git. Aucun test final n'a servi à ces choix.

### Analyser une photo entière par fenêtres

Une façade de douze mégapixels ramenée à 432 pixels perd ses fissures fines. L'option
`--windowed` découpe la photo en fenêtres de 640 pixels, la taille des images d'apprentissage,
avec 128 pixels de recouvrement pour qu'un défaut coupé par une frontière reste entier dans la
fenêtre voisine :

```sh
.venv/bin/smartsite-model external data/prepared/ccsd_v1 \
  --run artifacts/training/damsegment_full_v1 --windowed \
  --output artifacts/predictions/windowed_v1 --device mps
```

Le recollage se fait en coordonnées de la photo d'origine. Deux sorties sont tenues séparément,
parce qu'elles ne coûtent pas la même mémoire : la zone couverte par classe, accumulée dans un
seul masque, et les propositions réduites à leur boîte et leur score. Garder un masque pleine
taille par proposition saturerait la mémoire sur une grande photo.

Les propositions d'une même classe qui décrivent le même défaut sont fusionnées. Le critère est
la part recouverte de la plus petite des deux boîtes, pas l'IoU : quand une fenêtre voit un défaut
entier et sa voisine seulement un morceau, l'IoU reste faible alors qu'il s'agit du même défaut.

Le découpage multiplie les passages du moteur, donc le temps par photo. Comparer les deux modes
sur le même corpus avant de choisir : le gain n'est pas acquis, et la charge d'alertes compte
autant que la couverture.

### Mesurer les fausses alertes sur des surfaces saines

SDNET2018 découpe 230 photos de béton en extraits de 256 pixels, étiquetés fissuré ou sain par
ses auteurs. L'éditeur refuse les téléchargements automatisés : l'archive se récupère à la main
depuis [sa page](https://digitalcommons.usu.edu/all_datasets/48/), puis le module la vérifie par
son empreinte.

```sh
.venv/bin/smartsite-data prepare-patches ~/Downloads/SDNET2018.zip \
  --output data/prepared/sdnet_walls_v1 --surface W --per-photo 10

.venv/bin/smartsite-model patch-alerts data/prepared/sdnet_walls_v1 \
  --run artifacts/training/damsegment_full_v1 \
  --output artifacts/predictions/sdnet_alerts_v1 --device mps
```

Le tirage prend le même nombre d'extraits par photo d'origine, pour qu'une scène ne pèse pas plus
qu'une autre, et écarte les extraits publiés deux fois. Les extraits fissurés sont mesurés dans
les mêmes conditions : sans ce contrôle, un taux d'alerte nul sur les surfaces saines pourrait
seulement vouloir dire que le modèle ne voit rien à cette échelle.

Les étiquettes des auteurs ne portent que sur la fissure : un extrait dit sain peut montrer de
l'écaillage ou un trou. Ces extraits ne servent ni à l'apprentissage ni au choix d'un seuil.

### Mesurer un modèle hors de son domaine d'entraînement

Les poids de référence ont appris sur un seul ouvrage en béton. `external` les mesure sur une
source indépendante, jamais utilisée pour l'apprentissage ni pour choisir un seuil :

```sh
.venv/bin/smartsite-model external data/prepared/ccsd_v1 \
  --run artifacts/training/damsegment_full_v1 \
  --output artifacts/predictions/external_coverage_v1 --device mps
```

Les références de cette source sont des masques sémantiques : il n'y a pas d'instances, donc pas
d'appariement un défaut pour un défaut. La commande mesure la **zone couverte** et le fait qu'une
photo soit signalée. Ces deux mesures complètent l'appariement strict d'`evaluate` ; elles ne le
remplacent pas et ne constituent pas une qualification chantier.

Une réserve importante accompagne la précision de zone : un masque de référence incomplet fait
compter une vraie fissure comme fausse alerte. Le corpus reste refusé à l'apprentissage, et la
commande s'arrête si son rapport de préparation l'autorise.

Les tests courants restent sans moteur. Les tests d'intégration CPU utilisent les vraies transformations et le vrai décodage RF-DETR avec des sorties synthétiques identifiées, sans téléchargement de poids ni GPU. Ils vérifient notamment les photos non carrées, les masques vides et la séparation des scores objets/pixels :

```sh
# Après installation de l'extra training uniquement.
.venv/bin/pytest tests/integration/test_aligned_inference.py
```

Le répertoire `tests/integration` n'est pas parcouru par défaut ; il faut le demander explicitement. Les inférences réelles sur les 149 photos complètent ces tests, mais ne remplacent pas la qualification indépendante du détecteur.

## Attribution

- [DamSegment v1](https://data.mendeley.com/datasets/z5z6gtt5t4/1), DOI `10.17632/z5z6gtt5t4.1`, Vahidreza Gharehbaghi, Caroline R. Bennett, Rémy Lequesne, Hang Zhao, Jian Li.
- [Concrete Crack Segmentation Dataset v1](https://data.mendeley.com/datasets/jwsn7tfbrp/1), DOI `10.17632/jwsn7tfbrp.1`, Çağlar Fırat Özgenel.

Les publications annoncent CC BY 4.0. Les exports conservent l'attribution et la description des modifications. Les images, archives et sorties volumineuses ne sont pas distribuées avec le code.

## Examiner les prédictions du pilote par boîtes

Après un entraînement par boîtes terminé, générer une galerie sur sa validation :

```sh
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m smartsite_ia.box_cli review \
  artifacts/training/moisture_boxes_v1 \
  --corpus artifacts/predictions/moisture_annotations_v1 \
  --config config/moisture_boxes_v1.json \
  --device mps --threshold 0.3 \
  --output artifacts/predictions/moisture_box_review_v1
```

La sortie doit être nouvelle. Le parcours vérifie les empreintes du modèle terminé, des données et des métriques, ainsi que les versions du moteur. Il utilise uniquement la validation (64 images maximum), reprend ses transformations déterministes et restitue les boîtes dans les pixels de la photo originale. Le chargement sûr des poids n'autorise pas de désérialisation arbitraire. Aucun apprentissage, téléchargement de poids ou service distant n'est nécessaire.

La galerie conserve les photos, références, prédictions, scores et attributions. Une zone est retrouvée si sa catégorie correspond et si l'IoU des boîtes atteint 0,5, avec un appariement un-à-un par score décroissant. Les sorties sont filtrées au score strictement supérieur au seuil choisi ; le seuil 0,3 est un réglage d'inspection non calibré. Les propositions non appariées ne prouvent pas à elles seules une fausse alerte physique : les annotations restent à confirmer. Ce comptage à seuil fixe n'est pas le F1 natif optimisé sur la validation, ni une AP de segmentation.

Les entrées et les poids restent inchangés. Une erreur ou interruption retire la sortie partielle ; relancer vers une sortie absente recommence seulement l'inférence. Les coordonnées brutes, les éventuels débordements tronqués et les sorties sans objet ou sans surface écartées sont tracés. Un modèle limité à cette petite validation n'est pas qualifié pour le chantier.

## Complément de données et contrôle des fausses alertes — v2

`config/moisture_collection_v2.json` conserve les 71 photos initiales et ajoute
35 originaux vérifiés. La revue de l'assistant retient 4 nouvelles photos de
moisissures suspectées, 11 photos entières sans cible visible et un recadrage
supplémentaire ; 20 nouvelles photos ambiguës ou hors périmètre sont exclues.
Les photos négatives font l'objet d'une décision explicite `negative`, d'une revue
des trois classes et de l'audit des annotations source. Une photo exclue ne devient
jamais automatiquement un exemple négatif. La validation humaine reste en attente.

La v2 comporte 44 images train et 12 validation, dont 4 nouvelles photos entières
sans cible. Les 8 photos de validation précédentes, leurs boîtes, leurs groupes
et leurs partitions restent inchangés. Les négatifs ne prouvent pas une conformité :
les fissures, risques électriques et défauts de carrelage sortent du périmètre de
ce détecteur à trois classes. Le corpus conserve seulement 2 boîtes de traces
d'humidité en apprentissage et un seul groupe de moisissures en validation.

Reconstitution depuis les originaux épinglés (le téléchargement reste explicite) :

```sh
.venv/bin/python -m smartsite_ia.collection fetch \
  --selection config/moisture_collection_v2.json \
  --cache data/raw/moisture_collection_v2/assets
.venv/bin/python -m smartsite_ia.collection_review \
  --selection config/moisture_collection_v2.json \
  --cache data/raw/moisture_collection_v2/assets \
  --review config/moisture_review_v2.json \
  --output artifacts/predictions/moisture_annotations_v2
```

Le pilote `config/moisture_boxes_v2.json` fixe 10 passages, la dernière époque,
la même graine et les mêmes paramètres que la v1, avec les nouvelles empreintes du
corpus. L'évaluation conserve un seuil de score de 0,30 et un IoU de 0,50. Les résultats
sur les huit anciennes photos et les quatre nouveaux négatifs se lisent séparément ;
les données et la durée d'apprentissage changent ensemble, donc il ne s'agit pas
d'une mesure de l'effet des seules données. Les commandes `smartsite-boxes` ci-dessus
acceptent ces chemins v2 dans de nouveaux dossiers, sans écraser les sorties v1.

La [galerie de revue v2](http://127.0.0.1:8768/moisture_annotations_v2/) conserve
les images, les rectangles proposés, les exclusions et les attributions. Aucun
modèle n'est promu automatiquement. L'objectif de notification avec photo et zone
suspecte nécessite encore des données représentatives du chantier, des annotations
validées et des critères mesurés de rappel et de fausses alertes.

Résultat de l'essai v2 exécuté : sur les huit photos communes, 4 références retrouvées
sur 13 pour chaque version, avec 6 propositions non appariées en v2 contre 5 en v1.
Les deux versions proposent quatre cadres sur deux des quatre nouveaux négatifs.
La v2 n'est pas promue. La [comparaison photo par photo](http://127.0.0.1:8768/moisture_comparison_v2/)
conserve les résultats et leurs limites ; aucun modèle n'est qualifié pour les notifications.


## Revue et préparation v3 — historique avant l’essai v3.1

La sélection `config/moisture_collection_v3.json` reprend les 53 photos utiles de v2
et ajoute 130 vues MBDD2025 et 9 Commons. Les 139 nouvelles photos sont revues :
10 candidates positives et 129 exclusions motivées. Les mousses sur bois, les vues
répétées et les cas ambigus ne sont pas convertis en négatifs. Les autres anciennes
photos restent documentées dans la revue v2.

`config/moisture_review_v3.json` conserve les annotations et partitions utiles de v2.
Le lot exporté propose 52 images d’apprentissage et 16 de validation : 63 photos
entières (52 positives, 11 négatives) et 5 recadrages négatifs, dont 2 nouveaux.
Les groupes visuels sont disjoints entre les lots, mais les bâtiments ne sont pas
certifiés. Les traces compatibles avec l’humidité passent de 2 à 18 boîtes train ;
les moisissures passent de 13 à 17. Ce sont des annotations proposées, pas des gains
mesurés du modèle. La préparation v3 initiale n’a pas été entraînée ; le résultat
de la version corrigée v3.1 figure plus bas. Aucun test final n’a été réalisé.

Rejouer la revue depuis le cache vérifié, vers un dossier absent :

```sh
.venv/bin/python -m smartsite_ia.collection_review \
  --selection config/moisture_collection_v3.json \
  --cache data/raw/moisture_collection_v3/assets \
  --review config/moisture_review_v3.json \
  --output artifacts/predictions/moisture_annotations_v3
```

Le téléchargement reste une commande séparée `smartsite-collection fetch`, avec
la même sélection et le même cache. Les limitations HTTP des sources sont à
respecter ; les originaux utilisés ici sont déjà reçus et vérifiés. Zenodo est
accepté sur son hôte HTTPS exact, avec les mêmes limites, plages et empreintes
que les autres fichiers. Les catégories et couleurs des trois cibles sont communes
à la collecte, à la revue et aux prédictions.

Contrôle technique sans entraîner :

```sh
.venv/bin/python -m smartsite_ia.box_cli check \
  artifacts/predictions/moisture_annotations_v3 \
  --config config/moisture_boxes_v3.json \
  --weights artifacts/models/pretrained/rf-detr-nano.pth --device mps
```

La recette v3 fixe dix passages, la dernière époque et les empreintes du lot.
La [revue des dix ajouts](http://127.0.0.1:8768/moisture_review_v3/) et la
[revue complète](http://127.0.0.1:8768/moisture_annotations_v3/) permettent de
contrôler catégories, rectangles et omissions ; elles conservent l’état préparatoire.
La validation humaine des annotations reste en attente, y compris sur les exemples
hérités. Les poids v1/v2 sont conservés ; aucune notification automatique n’est
qualifiée par cette préparation.


### Seconde vérification des annotations — v3.1

`config/moisture_review_v3_1.json` reprend la sélection v3 et corrige trois des dix
nouvelles photos après relecture détaillée : contours de coulures, séparation de
petites pertes de revêtement et deux omissions. Deux nouveaux recadrages négatifs
ont aussi été relus. Les 52 images train et 16 validation gardent leurs pixels,
groupes et partitions. Les annotations héritées et les exclusions sont inchangées.

Pour la préparation corrigée, la commande de revue précédente utilise toujours
`config/moisture_collection_v3.json` et son cache, avec la revue
`config/moisture_review_v3_1.json` et la sortie absente
`artifacts/predictions/moisture_annotations_v3_1`. La recette correspondante est
`config/moisture_boxes_v3_1.json`. La v3 précédente reste conservée.

La [comparaison avant/après](http://127.0.0.1:8768/moisture_review_v3_1/) explique
les corrections et les incertitudes. Cette seconde lecture est celle de l’assistant,
pas une validation humaine indépendante. Un essai exploratoire a ensuite été
exécuté sur cette préparation, sans changement de son statut de validation.


### Résultat exploratoire v3.1

Dix passages sur 52 images terminés en 181,45 secondes sur MPS, réseau désactivé,
dernier passage conservé. Sur les huit photos historiques : 2 zones retrouvées sur
13, contre 4 pour v1/v2. Sur les quatre négatifs : une proposition sur une photo,
contre quatre sur deux. Sur les seize validations : 3/26 références retrouvées,
5 propositions non appariées, 23 références manquées. Aucune des cinq moisissures
ni des treize traces n’est retrouvée dans sa classe au score > 0,30 et IoU ≥ 0,50.

La v3.1 n’est pas promue. La baisse des fausses propositions s’accompagne d’une
perte de rappel ; aucune notification automatique n’est qualifiée. Les modèles,
photos et annotations antérieurs sont conservés. La
[comparaison sur les mêmes photos](http://127.0.0.1:8768/moisture_comparison_v3_1/)
et le [rapport détaillé](http://127.0.0.1:8768/moisture_box_review_v3_1/)
présentent les résultats, les attributions et les limites. Les anciennes versions
ont aussi été exécutées sur les quatre nouveaux exemples, dans des rapports
complémentaires distincts de leurs validations originales.

Diagnostic séparé sur les images déjà apprises : 1/17 moisissure, 13/18 traces et
25/42 revêtements retrouvés. Ces chiffres ne mesurent pas la généralisation ; ils
montrent un apprentissage encore faible des moisissures et un écart train/validation
pour les traces. Les causes restent à isoler. Les comptes ont été vérifiés avec
COCO par photo, pour les trois modèles et pour le diagnostic. Annotations humaines
toujours en attente, aucun test final ni donnée chantier utilisateur disponible.

Pour reproduire une revue du modèle terminé, les commandes `smartsite-boxes review`
utilisent les chemins v3.1 indiqués plus haut et un dossier de sortie absent. Les
résultats de ce calcul local restent dans `artifacts/` ; ils ne sont pas distribués
avec le dépôt. Aucun code fonctionnel ni dépendance modifié pour cet essai.
