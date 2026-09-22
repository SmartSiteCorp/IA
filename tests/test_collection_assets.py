"""Contrôler les téléchargements partiels et la reprise, sans appeler Internet."""

import io
import zlib

import pytest

from smartsite_ia import collection_assets as assets
from smartsite_ia.curation import digest


@pytest.fixture
def asset():
    raw = b"Synthetic bytes for transport tests." * 100
    compressor = zlib.compressobj(wbits=-15)
    payload = compressor.compress(raw) + compressor.flush()
    record = {
        "url": "https://data.mendeley.com/public-files/test",
        "sha256": digest(raw),
        "size": len(raw),
        "transfer_size": len(payload),
        "encoding": "deflate",
        "start": 300,
        "total": 100_000,
    }
    return raw, payload, record


class Response(io.BytesIO):
    def __init__(self, raw, status, headers):
        super().__init__(raw)
        self.status = status
        self.headers = headers


def serve(monkeypatch, payload, record, *, status=206, content_range=None):
    requests = []

    class Opener:
        def open(self, request, timeout):
            requests.append((request, timeout))
            return Response(
                payload,
                status,
                {
                    "Content-Range": content_range
                    or (
                        f"bytes {record['start']}-"
                        f"{record['start'] + record['transfer_size'] - 1}/{record['total']}"
                    )
                },
            )

    monkeypatch.setattr(assets, "build_opener", lambda *_: Opener())
    return requests


def test_partial_download_is_verified_atomic_and_reused(asset, tmp_path, monkeypatch):
    raw, payload, record = asset
    parsed = assets.Asset.parse(record)
    requests = serve(monkeypatch, payload, record)
    result = assets.fetch_asset(parsed, tmp_path)
    assert result.read_bytes() == raw
    assert requests[0][0].get_header("Range") == f"bytes=300-{300 + len(payload) - 1}"
    assert len(list(tmp_path.iterdir())) == 1
    assert assets.fetch_asset(parsed, tmp_path) == result
    assert len(requests) == 1
    result.write_bytes(b"wrong cache")
    with pytest.raises(ValueError, match="Cached"):
        assets.fetch_asset(parsed, tmp_path)


def test_whole_file(asset, tmp_path, monkeypatch):
    raw, _, record = asset
    record.update(
        start=None, size=len(raw), transfer_size=len(raw), total=len(raw), encoding="identity"
    )

    class Opener:
        def open(self, request, timeout):
            assert request.get_header("Range") is None
            return Response(raw, 200, {})

    monkeypatch.setattr(assets, "build_opener", lambda *_: Opener())
    assert assets.fetch_asset(assets.Asset.parse(record), tmp_path).read_bytes() == raw


@pytest.mark.parametrize(
    "changes",
    [
        {"url": "http://data.mendeley.com/photo"},
        {"url": "https://localhost/x"},
        {"url": "https://user:password@upload.wikimedia.org/x"},
        {"url": None},
        {"sha256": "bad"},
        {"size": True},
        {"size": 0},
        {"size": 8 * 1024**2 + 1},
        {"transfer_size": -1},
        {"transfer_size": None},
        {"start": -1},
        {"start": True},
        {"start": 99999},
        {"total": 0},
        {"total": True},
        {"total": 10_000_000_001},
        {"encoding": "executable"},
        {"start": None},
    ],
)
def test_bad_asset_rejected_before_network(asset, changes):
    _, _, record = asset
    with pytest.raises(ValueError):
        assets.Asset.parse({**record, **changes})


@pytest.mark.parametrize(
    "failure",
    ["status", "range", "short", "long", "sha", "size", "truncated", "trailing", "bomb", "timeout"],
)
def test_invalid_transfer_never_publishes(asset, tmp_path, monkeypatch, failure):
    raw, payload, record = asset
    status = 206
    crange = None
    if failure == "status":
        status = 200
    elif failure == "range":
        crange = "bytes 0-10/11"
    elif failure == "short":
        payload = payload[:-1]
    elif failure == "long":
        payload += b"xx"
    elif failure == "sha":
        record["sha256"] = "0" * 64
    elif failure == "size":
        record["size"] += 1
    elif failure == "truncated":
        payload = payload[:-1]
        record["transfer_size"] = len(payload)
    elif failure == "trailing":
        payload += b"extra"
        record["transfer_size"] = len(payload)
    elif failure == "bomb":
        record["size"] = 20
    elif failure == "timeout":
        calls = iter([0, 121])
        monkeypatch.setattr(assets.time, "monotonic", lambda: next(calls))
    serve(monkeypatch, payload, record, status=status, content_range=crange)
    with pytest.raises(ValueError):
        assets.fetch_asset(assets.Asset.parse(record), tmp_path)
    assert not list(tmp_path.iterdir())


def test_linked_cache_rejected(asset, tmp_path):
    raw, _, record = asset
    cache = tmp_path / "cache"
    cache.mkdir()
    target = tmp_path / "original"
    target.write_bytes(raw)
    (cache / digest(raw)).symlink_to(target)
    with pytest.raises(ValueError):
        assets.fetch_asset(assets.Asset.parse(record), cache)
    linked = tmp_path / "linked"
    linked.symlink_to(cache, target_is_directory=True)
    with pytest.raises(ValueError):
        assets.fetch_asset(assets.Asset.parse(record), linked)


def test_redirect_policy_blocks_unapproved_hosts():
    handler = assets.SourceRedirects()
    request = assets.Request("https://data.mendeley.com/test")
    with pytest.raises(ValueError):
        handler.redirect_request(request, None, 302, "", {}, "http://127.0.0.1/private")
