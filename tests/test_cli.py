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

    def download(path):
        calls.append(path)
        return path

    monkeypatch.setattr(cli, "download_archive", download)
    monkeypatch.setattr("sys.argv", ["smartsite-data", "download", "--output", str(output)])
    assert cli.main() == 0
    assert calls == [output] and str(output) in capsys.readouterr().out


def test_cli_requires_command(monkeypatch):
    monkeypatch.setattr("sys.argv", ["smartsite-data"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
