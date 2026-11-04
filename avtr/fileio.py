"""Local and fsspec file operations, including checkpoint staging."""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import IO

import fsspec


def is_remote(path: str) -> bool:
    return "://" in str(path) and not str(path).startswith("file://")


def _fs(path: str):
    fs, rest = fsspec.core.url_to_fs(str(path))
    return fs, rest


def open_file(path: str, mode: str = "rb") -> IO:
    if not is_remote(path):
        if "w" in mode or "a" in mode:
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        return open(path, mode)
    return fsspec.open(str(path), mode).open()


def exists(path: str) -> bool:
    if not is_remote(path):
        return os.path.exists(path)
    fs, rest = _fs(path)
    return fs.exists(rest)


def isdir(path: str) -> bool:
    if not is_remote(path):
        return os.path.isdir(path)
    fs, rest = _fs(path)
    return fs.isdir(rest)


def listdir(path: str) -> list[str]:
    """Base names of the entries directly under ``path``."""
    if not is_remote(path):
        return sorted(os.listdir(path))
    fs, rest = _fs(path)
    return sorted(os.path.basename(p.rstrip("/")) for p in fs.ls(rest, detail=False))


def makedirs(path: str) -> None:
    if not is_remote(path):
        os.makedirs(path, exist_ok=True)
        return
    fs, rest = _fs(path)
    fs.makedirs(rest, exist_ok=True)


def remove(path: str) -> None:
    """Delete a file or a whole directory tree; missing paths are ignored."""
    # Several ranks may delete the same path at once, so a path that vanishes mid-call is fine too.
    try:
        if not is_remote(path):
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            elif os.path.exists(path):
                os.remove(path)
            return
        fs, rest = _fs(path)
        if fs.exists(rest):
            fs.rm(rest, recursive=True)
    except FileNotFoundError:
        pass


def copy(src: str, dst: str) -> None:
    """Copy a file or a directory tree between any two locations (local or remote)."""
    if not is_remote(src) and not is_remote(dst):
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            os.makedirs(os.path.dirname(os.path.abspath(dst)) or ".", exist_ok=True)
            shutil.copy2(src, dst)
        return
    # A directory source is passed with a trailing "/": fsspec then copies its contents into dst,
    # instead of nesting it as dst/<basename> when dst already exists.
    if is_remote(src) and not is_remote(dst):
        fs, rest = _fs(src)
        if fs.isdir(rest):
            fs.get(rest.rstrip("/") + "/", dst, recursive=True)
        else:
            fs.get(rest, dst)
        return
    if not is_remote(src):
        fs, rest = _fs(dst)
        if os.path.isdir(src):
            fs.put(src.rstrip("/") + "/", rest, recursive=True)
        else:
            fs.put(src, rest)
        return
    with tempfile.TemporaryDirectory() as tmp:
        local = os.path.join(tmp, os.path.basename(str(src).rstrip("/")))
        copy(src, local)
        copy(local, dst)


def local_path(path: str, cache_dir: str | None = None) -> str:
    """A local path holding ``path``: the path itself when local, else a copy under ``cache_dir`` (default
    ``$AVTR_CACHE`` or ``~/.cache/avtr``). Copies are reused across calls."""
    if not is_remote(path):
        return path
    root = cache_dir or os.environ.get("AVTR_CACHE", os.path.expanduser("~/.cache/avtr"))
    target = os.path.join(root, str(path).split("://", 1)[1].rstrip("/"))
    marker = target + ".complete"
    if not os.path.exists(marker):
        remove(target)  # a copy interrupted earlier
        copy(path, target)
        open(marker, "w").close()
    return target
