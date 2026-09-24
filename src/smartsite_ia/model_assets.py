"""Les poids de départ ont une origine et une empreinte fixes, comme les données."""

from smartsite_ia.categories import CLASS_NAMES as CLASS_NAMES
from smartsite_ia.source import Source

MODEL_VERSION = "1.10.1"
MODEL_NAME = "RFDETRSegMedium"
# Le téléchargeur est partagé avec les archives il contrôle taille durée et SHA
# Ce fichier ne contient aucune image
PRETRAINED = Source(
    url="https://storage.googleapis.com/rfdetr/rf-detr-seg-m-ft.pth",
    size=143024058,
    sha256="3ad325094735f431aee9962a8d204d68eb5bfc393d53e7e836e70998fef5ea58",
    expected_images=0,
)

# Variante de détection sans masques, distincte des poids historiques de segmentation.
# Origine et MD5 auteur vérifiés dans RF-DETR 1.10.1 ; SHA-256 calculé au téléchargement.
BOX_MODEL_NAME = "RFDETRNano"
BOX_PRETRAINED = Source(
    url="https://storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth",
    size=366287238,
    sha256="d8d6b9ee57d4d0ed2b1f305163624712a0532cb7bce0c747317984fc5457440d",
    expected_images=0,
)
