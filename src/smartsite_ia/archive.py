"""Validate ZIP members without extracting untrusted paths."""

import stat
from pathlib import PurePosixPath
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo

MAX_MEMBERS = 10000
MAX_MEMBER_BYTES = 8 * 1024 * 1024
MAX_EXPANDED_BYTES = 512 * 1024 * 1024


def checked_members(archive: ZipFile) -> dict[str, ZipInfo]:
    members = archive.infolist()
    if len(members) > MAX_MEMBERS:
        raise ValueError("Archive has too many entries")
    result: dict[str, ZipInfo] = {}
    names: set[str] = set()
    total = 0
    for member in members:
        name = member.filename.rstrip("/")
        parts = name.split("/")
        mode = stat.S_IFMT(member.external_attr >> 16)
        if (
            not name
            or member.orig_filename != member.filename
            or PurePosixPath(name).is_absolute()
            or any(part in ("", ".", "..") for part in parts)
            or any(c in name for c in ("\\", ":", "\x00"))
            or any(ord(c) < 32 for c in name)
            or mode not in (0, stat.S_IFREG, stat.S_IFDIR)
            or (mode == stat.S_IFDIR and not member.is_dir())
            or member.flag_bits & 1
            or member.compress_type not in (ZIP_DEFLATED, ZIP_STORED)
            or member.file_size > MAX_MEMBER_BYTES
            or member.file_size > max(1, member.compress_size) * 1000
            or name.casefold() in names
        ):
            raise ValueError(f"Unsafe or unsupported ZIP entry: {member.filename!r}")
        names.add(name.casefold())
        total += member.file_size
        if total > MAX_EXPANDED_BYTES:
            raise ValueError("Archive exceeds expanded size limit")
        if not member.is_dir():
            result[name] = member
    return result


def read_member(archive: ZipFile, member: ZipInfo) -> bytes:
    with archive.open(member) as stream:
        data = stream.read(MAX_MEMBER_BYTES + 1)
    if len(data) != member.file_size or len(data) > MAX_MEMBER_BYTES:
        raise ValueError(f"Invalid member size: {member.filename}")
    return data
