"""Synthetic fixtures for software behavior, never for model-quality claims."""

import hashlib
import io
import json
from dataclasses import replace
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from PIL import Image, ImageDraw

from smartsite_ia.source import DAMSEGMENT


def image_bytes(image, format):
    stream = io.BytesIO()
    image.save(stream, format=format)
    return stream.getvalue()


@pytest.fixture
def sample():
    polygon = [10, 10, 30, 10, 30, 20, 10, 20, 10, 10]
    document = {
        "image": {"file_name": "E (1).jpg", "width": 640, "height": 640},
        "annotations": [
            {"category_id": 0, "segmentation": [polygon], "bbox": [10, 10, 20, 10], "iscrowd": 0}
        ],
    }
    image = Image.new("RGB", (640, 640), (130, 130, 130))
    mask = Image.new("RGB", image.size)
    ImageDraw.Draw(mask).polygon(
        list(zip(polygon[::2], polygon[1::2], strict=True)), fill=(255, 0, 0)
    )
    yolo = "0 " + " ".join(str(v / 640) for v in polygon) + "\n"
    return {
        "image": image_bytes(image, "JPEG"),
        "mask": image_bytes(mask, "PNG"),
        "document": document,
        "yolo": yolo.encode(),
    }


@pytest.fixture
def archive_factory(tmp_path, sample):
    def create(changes=None, copies=1):
        entries = {}
        for i in range(1, copies + 1):
            base = "Damage Segmentaion/Easy"
            doc = json.loads(json.dumps(sample["document"]))
            doc["image"]["file_name"] = f"E ({i}).jpg"
            entries.update(
                {
                    f"{base}/Images/E ({i}).jpg": sample["image"],
                    f"{base}/Labels/Mask/E ({i})_mask.png": sample["mask"],
                    f"{base}/Labels/Pascal VOC/E ({i}).json": json.dumps(doc).encode(),
                    f"{base}/Labels/Yolo/E ({i}).txt": sample["yolo"],
                }
            )
        for key, value in (changes or {}).items():
            if value is None:
                entries.pop(key)
            else:
                entries[key] = value
        archive = tmp_path / "fixture.zip"
        with ZipFile(archive, "w", compression=ZIP_DEFLATED) as writer:
            for key, value in entries.items():
                writer.writestr(key, value)
        source = replace(
            DAMSEGMENT,
            size=archive.stat().st_size,
            sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            expected_images=copies,
        )
        return archive, source

    return create
