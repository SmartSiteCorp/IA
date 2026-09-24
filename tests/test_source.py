import hashlib
import io
from dataclasses import replace
from types import SimpleNamespace

import pytest

from smartsite_ia import source as module
from smartsite_ia.source import (
    DAMSEGMENT,
    SourceRedirects,
    check_url,
    download_archive,
    verify_archive,
)


@pytest.mark.parametrize(
    "url",
    [
        "http://data.mendeley.com/file",
        "https://localhost/file",
        "https://data.mendeley.com.evil.test/file",
        "https://user@data.mendeley.com/file",
        "https://data.mendeley.com:8443/file",
        "file:///etc/passwd",
    ],
)
def test_only_approved_https_sources(url):
    with pytest.raises(ValueError):
        check_url(url)


def test_zenodo_originals_use_only_the_exact_public_host():
    check_url("https://zenodo.org/api/records/15622584/files/MBDD2025.zip/content")
    for url in (
        "http://zenodo.org/file",
        "https://zenodo.org.evil.test/file",
        "https://user@zenodo.org/file",
        "https://zenodo.org:8443/file",
        "https://localhost/file",
    ):
        with pytest.raises(ValueError, match="approved HTTPS"):
            check_url(url)


def test_redirect_cannot_leave_source_hosts():
    with pytest.raises(ValueError):
        SourceRedirects().redirect_request(None, None, 302, "", {}, "http://127.0.0.1/")


def test_checksum_failure_and_size_validation(tmp_path, archive_factory):
    path, source = archive_factory()
    assert verify_archive(path, source) == source.sha256
    with pytest.raises(ValueError, match="SHA-256"):
        verify_archive(path, replace(source, sha256="0" * 64))
    with pytest.raises(ValueError, match="size"):
        verify_archive(path, replace(source, size=1))
    symlink = tmp_path / "linked.zip"
    symlink.symlink_to(path)
    with pytest.raises(ValueError):
        verify_archive(symlink, source)


def test_download_and_cached_reuse(tmp_path, monkeypatch):
    data = b"synthetic archive bytes"
    source = replace(DAMSEGMENT, size=len(data), sha256=hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(
        module, "build_opener", lambda *_: SimpleNamespace(open=lambda *a, **k: io.BytesIO(data))
    )
    output = tmp_path / "archive.zip"
    assert download_archive(output, source) == output
    assert output.read_bytes() == data
    monkeypatch.setattr(module, "build_opener", lambda *_: pytest.fail("Unexpected network access"))
    assert download_archive(output, source) == output
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.parametrize("data", [b"", b"ab", b"abcd"])
def test_failed_download_leaves_no_partial_file(tmp_path, monkeypatch, data):
    source = replace(DAMSEGMENT, size=3, sha256=hashlib.sha256(b"abc").hexdigest())
    monkeypatch.setattr(
        module, "build_opener", lambda *_: SimpleNamespace(open=lambda *a, **k: io.BytesIO(data))
    )
    with pytest.raises(ValueError):
        download_archive(tmp_path / "archive.zip", source)
    assert not list(tmp_path.iterdir())


def test_network_failure_cleans_staging(tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise OSError("Network interrupted")

    monkeypatch.setattr(module, "build_opener", lambda *_: SimpleNamespace(open=unavailable))
    with pytest.raises(OSError, match="interrupted"):
        download_archive(tmp_path / "archive.zip")
    assert not list(tmp_path.iterdir())
