"""Les poids de départ ont une origine et une empreinte fixes, comme les données."""

from smartsite_ia.source import Source

MODEL_VERSION = "1.10.1"
MODEL_NAME = "RFDETRSegMedium"
CLASS_NAMES = ("crack", "surface_loss")
# Le téléchargeur est partagé avec les archives il contrôle taille durée et SHA
# Ce fichier ne contient aucune image
PRETRAINED = Source(
    url="https://storage.googleapis.com/rfdetr/rf-detr-seg-m-ft.pth",
    size=143024058,
    sha256="3ad325094735f431aee9962a8d204d68eb5bfc393d53e7e836e70998fef5ea58",
    expected_images=0,
)
