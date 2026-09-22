"""Récupérer les seuls fichiers sélectionnés, sans charger une grosse archive entière."""

import os
import re
import tempfile
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import Request, build_opener

from smartsite_ia.curation import digest
from smartsite_ia.review import MAX_FILE_BYTES, read_local
from smartsite_ia.source import SourceRedirects, check_url


@dataclass(frozen=True)
class Asset:
    url: str
    sha256: str
    size: int
    transfer_size: int
    encoding: str
    start: int | None
    total: int

    @classmethod
    def parse(cls, data: dict[str, Any]) -> "Asset":
        """Les limites s'appliquent avant le réseau et avant la décompression."""
        url, sha = data.get("url"), data.get("sha256")
        if not isinstance(url, str) or not isinstance(sha, str):
            raise ValueError("Asset URL and SHA-256 are required")
        check_url(url)
        if re.fullmatch(r"[0-9a-f]{64}", sha) is None:
            raise ValueError("Invalid asset SHA-256")
        size, transfer = data.get("size"), data.get("transfer_size")
        if (
            type(size) is not int
            or not 0 < size <= MAX_FILE_BYTES
            or type(transfer) is not int
            or not 0 < transfer <= MAX_FILE_BYTES
        ):
            raise ValueError("Asset exceeds the file size limit")
        start, total = data.get("start"), data.get("total")
        if type(total) is not int or not 0 < total <= 10_000_000_000:
            raise ValueError("Invalid remote object size")
        encoding = data.get("encoding")
        if encoding not in {"identity", "deflate"}:
            raise ValueError("Unsupported asset encoding")
        if start is None:
            if encoding != "identity" or total != transfer or size != transfer:
                raise ValueError("Whole files must use identity encoding and consistent sizes")
        elif type(start) is not int or start < 0 or start + transfer > total:
            raise ValueError("Invalid byte range")
        return cls(url, sha, size, transfer, encoding, start, total)


def unpack_asset(payload: bytes, asset: Asset) -> bytes:
    """Une plage ZIP est du deflate brut ; on borne aussi sa taille une fois ouverte."""
    if len(payload) != asset.transfer_size:
        raise ValueError("Incomplete or oversized download")
    raw = payload
    if asset.encoding == "deflate":
        decoder = zlib.decompressobj(-15)
        raw = decoder.decompress(payload, asset.size + 1)
        if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
            raise ValueError("Truncated, oversized or concatenated deflate data")
    if len(raw) != asset.size or digest(raw) != asset.sha256:
        raise ValueError("Downloaded asset differs from its pinned size or SHA-256")
    return raw


def fetch_asset(asset: Asset, cache: Path) -> Path:
    """Le cache se reprend fichier par fichier, en revérifiant les fichiers déjà présents."""
    cache.mkdir(parents=True, exist_ok=True)
    if cache.is_symlink():
        raise ValueError("Asset cache must not be a symbolic link")
    destination = cache / asset.sha256
    if destination.exists() or destination.is_symlink():
        raw = read_local(cache, asset.sha256, MAX_FILE_BYTES)
        if len(raw) != asset.size or digest(raw) != asset.sha256:
            raise ValueError("Cached asset differs from the manifest")
        return destination
    headers = {"User-Agent": "SmartSite-dataset-tools/0.1"}
    if asset.start is not None:
        headers["Range"] = f"bytes={asset.start}-{asset.start + asset.transfer_size - 1}"
    request = Request(asset.url, headers=headers)
    started = time.monotonic()
    with build_opener(SourceRedirects()).open(request, timeout=30) as response:
        expected = 200 if asset.start is None else 206
        if response.status != expected:
            raise ValueError("Unexpected download status")
        if asset.start is not None:
            expected_range = (
                f"bytes {asset.start}-{asset.start + asset.transfer_size - 1}/{asset.total}"
            )
            if response.headers.get("Content-Range") != expected_range:
                raise ValueError("Server did not honour the exact byte range")
        # Lire par petits blocs permet de vérifier la durée et d'éviter un flux sans fin.
        payload = bytearray()
        while chunk := response.read(min(65536, asset.transfer_size + 1 - len(payload))):
            payload.extend(chunk)
            if len(payload) > asset.transfer_size or time.monotonic() - started > 120:
                raise ValueError("Download exceeded its size or time limit")
    raw = unpack_asset(bytes(payload), asset)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=cache, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
        # Le lien publie le fichier complet sans écraser un fichier apparu entre-temps.
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination
