import json

import pytest

from smartsite_ia import cli
from smartsite_ia.importer import import_archive


def test_cli_import_uses_real_converter(tmp_path, archive_factory, monkeypatch, capsys):
    archive, source = archive_factory()
    output = tmp_path / "prepared"
    monkeypatch.setattr(cli, "import_archive", lambda a, o: import_archive(a, o, source))
    monkeypatch.setattr(
        "sys.argv", ["smartsite-data", "import", str(archive), "--output", str(output)]
    )
    assert cli.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["images"] == 1 and result["approved_for_training"] is False
    assert (output / "manifest.json").exists()


def test_cli_reports_failure(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        [
            "smartsite-data",
            "import",
            str(tmp_path / "missing.zip"),
            "--output",
            str(tmp_path / "prepared"),
        ],
    )
    assert cli.main() == 1
    assert "Preparation failed" in capsys.readouterr().err


def test_cli_download_dispatch(tmp_path, monkeypatch, capsys):
    output = tmp_path / "download.zip"
    calls = []

    def download(path, source):
        calls.append((path, source))
        return path

    monkeypatch.setattr(cli, "download_archive", download)
    monkeypatch.setattr("sys.argv", ["smartsite-data", "download", "--output", str(output)])
    assert cli.main() == 0
    assert calls == [(output, cli.SOURCES["damsegment_v1"])]
    assert str(output) in capsys.readouterr().out


def test_cli_external_download_selects_pinned_archive(tmp_path, monkeypatch, capsys):
    output = tmp_path / "external.rar"
    calls = []
    monkeypatch.setattr(cli, "download_archive", lambda p, s: calls.append((p, s)) or p)
    monkeypatch.setattr(
        "sys.argv",
        [
            "smartsite-data",
            "download",
            "--output",
            str(output),
            "--dataset",
            "concrete_crack_segmentation_v1",
        ],
    )
    assert cli.main() == 0
    assert calls == [(output, cli.SOURCES["concrete_crack_segmentation_v1"])]
    assert str(output) in capsys.readouterr().out


def test_cli_requires_command(monkeypatch):
    monkeypatch.setattr("sys.argv", ["smartsite-data"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
