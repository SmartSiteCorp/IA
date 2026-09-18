import io
import stat
from zipfile import ZipFile, ZipInfo

import pytest

from smartsite_ia import archive as module
from smartsite_ia.archive import checked_members, read_member


@pytest.mark.parametrize(
    "name", ["../escape", "/absolute", "a/../b", "a//b", "a/./b", "C:/escape", "a\\b", "a\nb"]
)
def test_unsafe_paths_rejected(name):
    stream = io.BytesIO()
    with ZipFile(stream, "w") as writer:
        writer.writestr(name, b"data")
    with ZipFile(stream) as reader, pytest.raises(ValueError, match="Unsafe"):
        checked_members(reader)


def test_symlink_rejected():
    stream = io.BytesIO()
    item = ZipInfo("link")
    item.create_system = 3
    item.external_attr = (stat.S_IFLNK | 0o777) << 16
    with ZipFile(stream, "w") as writer:
        writer.writestr(item, "../outside")
    with ZipFile(stream) as reader, pytest.raises(ValueError, match="Unsafe"):
        checked_members(reader)


def test_case_collision_rejected():
    stream = io.BytesIO()
    with ZipFile(stream, "w") as writer:
        writer.writestr("A.jpg", b"a")
        writer.writestr("a.jpg", b"b")
    with ZipFile(stream) as reader, pytest.raises(ValueError, match="Unsafe"):
        checked_members(reader)


@pytest.mark.parametrize("limit", ["MAX_MEMBERS", "MAX_MEMBER_BYTES", "MAX_EXPANDED_BYTES"])
def test_resource_limits(archive_factory, monkeypatch, limit):
    path, _ = archive_factory()
    monkeypatch.setattr(module, limit, 1)
    with ZipFile(path) as reader, pytest.raises(ValueError):
        checked_members(reader)


def test_read_member_enforces_bound(archive_factory, monkeypatch):
    path, _ = archive_factory()
    with ZipFile(path) as reader:
        members = checked_members(reader)
        monkeypatch.setattr(module, "MAX_MEMBER_BYTES", 1)
        with pytest.raises(ValueError, match="size"):
            read_member(reader, next(iter(members.values())))
