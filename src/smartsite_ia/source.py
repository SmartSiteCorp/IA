"""Pinned public source and bounded, integrity-checked download."""

import hashlib
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


@dataclass(frozen=True)
class Source:
    url: str
    size: int
    sha256: str
    expected_images: int


DAMSEGMENT = Source(
    url="https://data.mendeley.com/public-files/datasets/z5z6gtt5t4/files/"
    "bd928c77-1194-4ee3-ac1a-f6ba3036e193/file_downloaded",
    size=115355010,
    sha256="d87849a70d7a2e280e49d3687d89a5850902804554a35d779c941ea71ffcff1a",
    expected_images=1500,
)
CONCRETE_CRACK_SEGMENTATION = Source(
    url="https://data.mendeley.com/public-files/datasets/jwsn7tfbrp/files/"
    "88e685a6-e3c5-423d-845f-89e35a457867/file_downloaded",
    size=745914150,
    sha256="1b8458ab6f84dc5086e9af8579e2cabccd3d209df97b46622096a4c105d5a6b2",
    expected_images=458,
)
SOURCES = {
    "damsegment_v1": DAMSEGMENT,
    "concrete_crack_segmentation_v1": CONCRETE_CRACK_SEGMENTATION,
}
PROVENANCE = {
    "dataset": "DamSegment",
    "version": 1,
    "doi": "10.17632/z5z6gtt5t4.1",
    "url": "https://data.mendeley.com/datasets/z5z6gtt5t4/1",
    "license": "CC-BY-4.0",
    "license_url": "https://creativecommons.org/licenses/by/4.0/",
    "authors": [
        "Vahidreza Gharehbaghi",
        "Caroline R. Bennett",
        "Rémy Lequesne",
        "Hang Zhao",
        "Jian Li",
    ],
    "license_evidence": "Publisher dataset metadata; no licence file inside segmentation ZIP",
}
ALLOWED_HOSTS = {
    "data.mendeley.com",
    "prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com",
    "storage.googleapis.com",  # Hébergement officiel des poids RF-DETR
    "upload.wikimedia.org",  # Originaux Commons, avec attribution dans le manifeste
}


def check_url(url: str) -> None:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in ALLOWED_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in (None, 443)
    ):
        raise ValueError("Download URL is outside the approved HTTPS source hosts")


class SourceRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        check_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def verify_archive(path: Path, source: Source = DAMSEGMENT) -> str:
    if path.is_symlink() or not path.is_file() or path.stat().st_size != source.size:
        raise ValueError("Archive is missing, linked or differs from the published size")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != source.sha256:
        raise ValueError("Archive SHA-256 does not match the pinned source")
    return digest


def download_archive(destination: Path, source: Source = DAMSEGMENT) -> Path:
    """Reuse a verified archive; publish a complete download without overwriting files."""
    check_url(source.url)
    if destination.exists() or destination.is_symlink():
        verify_archive(destination, source)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(source.url, headers={"User-Agent": "SmartSite-dataset-tools/0.1"})
    started = time.monotonic()
    partial: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
            partial = Path(output.name)
            with build_opener(SourceRedirects()).open(request, timeout=30) as response:
                size = 0
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > source.size or time.monotonic() - started > 600:
                        raise ValueError("Download exceeded its size or time limit")
                    output.write(chunk)
        verify_archive(partial, source)
        try:
            os.link(partial, destination)
        except FileExistsError:
            verify_archive(destination, source)
        return destination
    finally:
        if partial is not None:
            partial.unlink(missing_ok=True)
