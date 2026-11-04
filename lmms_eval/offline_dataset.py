"""Loading a task's dataset from a local directory, a downloaded hub snapshot or remote files."""

import fcntl
import hashlib
import io
import os
import posixpath
import re
import shutil
import tarfile
import time
import zipfile
from collections.abc import Iterator, Mapping
from glob import glob

from loguru import logger as eval_logger

# Forward EXACTLY ONE key. `data_files` is the only one a local load needs; every other key in a
# task yaml is ignored here, which keeps the load independent of keys meant for something else:
#   * force_download / local_files_only -- DownloadConfig fields, not load_dataset parameters.
#     They fall into **config_kwargs and the packaged builder rejects them
#     (ParquetConfig(force_download=False) -> TypeError).
#   * revision -- IS a load_dataset parameter, but against an already-downloaded snapshot it is
#     meaningless; the snapshot was taken at that revision.
#   * cache_dir -- the media extraction SUBDIR name under $HF_HOME (e.g. "longvideobench"),
#     consumed by the video branch, not an HF cache path.
#   * video_subdir -- read by the TVG tasks' own utils.
#   * name -- api/task.py already passes name=self.DATASET_NAME; forwarding it too would raise
#     TypeError for a duplicate keyword argument.
_FORWARDED_KWARGS = frozenset({"data_files"})

# Unlisted extensions and extensionless globs such as "test-*" load with the parquet builder.
_EXTENSION_TO_BUILDER = {".csv": "csv", ".json": "json", ".jsonl": "json"}

_ENV_RE = re.compile(r"\$\{(\w+)\}|\$(\w+)")


def resolve_dataset_path(path: str) -> str:
    """``dataset_path`` with ``${VAR}`` / ``$VAR`` expanded; an unset variable is an error rather
    than a silently wrong path."""
    missing = [a or b for a, b in _ENV_RE.findall(str(path)) if (a or b) not in os.environ]
    if missing:
        raise ValueError(
            f"dataset_path {path!r} needs environment variable(s) {', '.join(missing)}"
        )
    return os.path.expanduser(os.path.expandvars(str(path)))


def _localize_data_files(local_dataset_path: str, value: object) -> object:
    if isinstance(value, str):
        if os.path.isabs(value) or "://" in value:
            return value
        return os.path.join(local_dataset_path, value)
    if isinstance(value, list):
        return [_localize_data_files(local_dataset_path, item) for item in value]
    if isinstance(value, tuple):
        return tuple(_localize_data_files(local_dataset_path, item) for item in value)
    if isinstance(value, Mapping):
        return {
            key: _localize_data_files(local_dataset_path, item)
            for key, item in value.items()
        }
    return value


def _iter_data_file_paths(value: object) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_data_file_paths(item)
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_data_file_paths(item)


def _builder_for_data_files(data_files: object) -> str:
    builders = {
        _EXTENSION_TO_BUILDER.get(os.path.splitext(path)[1].lower(), "parquet")
        for path in _iter_data_file_paths(data_files)
    }
    if len(builders) > 1:
        raise ValueError(
            "data_files mixes file types that need different load_dataset builders "
            f"({', '.join(sorted(builders))}): {data_files!r}"
        )
    return builders.pop() if builders else "parquet"


def local_dataset_load_spec(
    local_dataset_path: str,
    dataset_kwargs: Mapping[str, object],
) -> tuple[str, dict[str, object]]:
    """``(path, kwargs)`` for ``load_dataset`` on a local dataset directory: the directory itself,
    or, with ``data_files`` (relative to the directory), the builder those files need."""
    load_kwargs = {
        key: value for key, value in dataset_kwargs.items() if key in _FORWARDED_KWARGS
    }
    if "data_files" not in load_kwargs:
        return local_dataset_path, load_kwargs

    load_kwargs["data_files"] = _localize_data_files(
        local_dataset_path, load_kwargs["data_files"]
    )
    return _builder_for_data_files(load_kwargs["data_files"]), load_kwargs


def _join_parts(parts: list[str], joined: str) -> None:
    """Concatenate the sorted ``parts`` of a split archive into ``joined``.

    The bytes go to ``<joined>.tmp``, renamed into place when complete, so ``joined`` is never a
    truncated archive; one left behind by an interrupted extraction is reused while it still has the
    parts' total size and is newer than every part.
    """
    parts = sorted(parts)
    if (
        os.path.isfile(joined)
        and os.path.getsize(joined) == sum(os.path.getsize(p) for p in parts)
        and os.path.getmtime(joined) >= max(os.path.getmtime(p) for p in parts)
    ):
        eval_logger.info(f"Reusing {joined}, joined from {parts}")
        return
    tmp = joined + ".tmp"
    try:
        with open(tmp, "wb") as out_tar:
            for part in parts:
                with open(part, "rb") as part_file:
                    shutil.copyfileobj(part_file, out_tar)
        os.replace(tmp, joined)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    eval_logger.info(f"Concatenated parts {parts} into {joined}")


def _extract_into(
    root: str, zip_files: list[str], tar_files: list[str], target: str, scratch: str
) -> None:
    for zip_file in zip_files:
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(target)
        eval_logger.info(f"Extracted all files from {zip_file} to {target}")

    # Group the parts of a split archive (name.tar.part.aa, name_1.tar, ...) together.
    tar_parts_dict: dict[str, list[str]] = {}
    for tar_file in tar_files:
        head, tail = os.path.split(tar_file)
        base_name = os.path.join(head, re.sub(r"_\d+$", "", tail.split(".tar")[0]))
        tar_parts_dict.setdefault(base_name, []).append(tar_file)
    for base_name, parts in tar_parts_dict.items():
        eval_logger.info(f"Extracting following tar files: {parts}")
        split = len(parts) > 1
        if split:
            # Joined next to cache_dir (the source may be read-only), deleted once extracted.
            name = os.path.relpath(base_name, root).replace(os.sep, "_")
            archive = f"{scratch}.{name}.tar"
            _join_parts(parts, archive)
        else:
            # A single archive is read in place; "r:*" also opens .tar.gz / .bz2 / .xz.
            archive = parts[0]
        with tarfile.open(archive, "r:*") as tar_ref:
            tar_ref.extractall(target)
        eval_logger.info(f"Extracted all files from {archive} to {target}")
        if split:
            os.remove(archive)


def check_media_cache(cache_dir: str, create_link: bool = False) -> None:
    """Raise if ``cache_dir`` is a symlink whose target is gone and ``create_link`` would not
    replace it: every media lookup under it would miss, and an extraction would only fail at the
    final rename, after unpacking everything."""
    cache_dir = cache_dir.rstrip("/")
    if os.path.islink(cache_dir) and not os.path.exists(cache_dir) and not create_link:
        raise FileNotFoundError(
            f"{cache_dir} is a symlink to {os.readlink(cache_dir)!r}, which does not exist; "
            "point it at the extracted media, or remove it to extract the archives there"
        )


def extract_media_archives(
    local_dataset_path: str, cache_dir: str, create_link: bool = False
) -> None:
    """Extract every zip / (split) tar under ``local_dataset_path`` into ``cache_dir``, once.

    Skipped when ``cache_dir`` already exists (a directory, or a link that resolves), so media
    extracted, copied or linked there by hand is used as is. Extraction goes to
    ``<cache_dir>.partial`` and is renamed into place when complete, under a file lock, so an
    interrupted run or a second process (another rank, another eval) never sees a half-extracted
    ``cache_dir``. The archives themselves are left in place; the parts of a split archive are
    joined into ``<cache_dir>.<name>.tar``, deleted once extracted. ``create_link`` points
    ``cache_dir`` at ``local_dataset_path`` when there is nothing to extract. A ``cache_dir`` link
    that no longer resolves is an error (``check_media_cache``), except that ``create_link``
    replaces it.
    """
    cache_dir = cache_dir.rstrip("/")
    check_media_cache(cache_dir, create_link)
    if os.path.exists(cache_dir):
        return
    zip_files = glob(os.path.join(local_dataset_path, "**/*.zip"), recursive=True)
    tar_files = glob(os.path.join(local_dataset_path, "**/*.tar*"), recursive=True)
    if not (zip_files or tar_files or create_link):
        return

    os.makedirs(os.path.dirname(os.path.abspath(cache_dir)), exist_ok=True)
    with open(cache_dir + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if os.path.exists(cache_dir):
            return
        if os.path.islink(cache_dir):
            os.remove(cache_dir)
        if zip_files or tar_files:
            partial = cache_dir + ".partial"
            shutil.rmtree(partial, ignore_errors=True)
            _extract_into(local_dataset_path, zip_files, tar_files, partial, cache_dir)
            os.rename(partial, cache_dir)
        else:
            os.symlink(local_dataset_path, cache_dir)
            eval_logger.info(f"Symbolic link created: {local_dataset_path} -> {cache_dir}")


# ---------------------------------------------------------------------------------------------
# Remote files: annotation files fetched by URL, and members streamed out of a remote archive.
# ---------------------------------------------------------------------------------------------


def is_url(path: object) -> bool:
    return isinstance(path, str) and path.startswith(("http://", "https://"))


def has_remote_data_files(data_files: object) -> bool:
    return any(is_url(p) for p in _iter_data_file_paths(data_files))


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_url(url: str, cache_root: str, sha256: str | None = None) -> str:
    """Download ``url`` once into ``cache_root`` and return the local path.

    The file is written to a temporary name and renamed when complete, under a file lock, so
    concurrent ranks wait for one download. With ``sha256`` the content is verified (a cached
    copy too), and a mismatch raises instead of evaluating on different data. A verified cached
    copy is returned without taking the lock, so a read-only cache works.
    """
    import requests

    name = os.path.basename(url.split("?", 1)[0]) or "data"
    key = sha256 or hashlib.sha256(url.encode()).hexdigest()[:16]
    local = os.path.join(cache_root, key[:16], name)
    if os.path.exists(local) and (not sha256 or _sha256(local) == sha256):
        return local
    os.makedirs(os.path.dirname(local), exist_ok=True)
    lock = os.open(local + ".lock", os.O_RDONLY | os.O_CREAT, 0o666)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if os.path.exists(local) and sha256 and _sha256(local) != sha256:
            try:
                os.remove(local)  # a stale or corrupt copy: download again
            except OSError as exc:
                raise ValueError(f"{local} does not match the pinned sha256 and cannot be replaced: {exc}") from exc
        if not os.path.exists(local):
            eval_logger.info(f"Downloading {url}")
            tmp = local + ".partial"
            with requests.get(url, stream=True, timeout=60) as resp:
                resp.raise_for_status()
                with open(tmp, "wb") as out:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        out.write(chunk)
            os.replace(tmp, local)
        if sha256 and _sha256(local) != sha256:
            os.remove(local)
            raise ValueError(f"{url}: sha256 does not match the pinned {sha256}; the file changed upstream")
    finally:
        os.close(lock)
    return local


def remote_data_files_load_spec(
    dataset_kwargs: Mapping[str, object], cache_root: str
) -> tuple[str, dict[str, object]]:
    """``data_files`` whose entries are URLs, fetched into ``cache_root`` (verified against the
    optional ``data_files_sha256`` mapping of split -> sha256), as (builder, load kwargs)."""
    data_files = dataset_kwargs["data_files"]
    if not isinstance(data_files, Mapping):
        data_files = {"train": data_files}
    sha = dict(dataset_kwargs.get("data_files_sha256") or {})
    local = {}
    for split, value in data_files.items():
        urls = value if isinstance(value, (list, tuple)) else [value]
        local[split] = [fetch_url(u, cache_root, sha.get(split)) if is_url(u) else u for u in urls]
    return _builder_for_data_files(local), {"data_files": local}


class _NotResumable(IOError):
    """A download error that a retry cannot fix (a 4xx other than 408/429, or Range ignored)."""


class _ResumableHTTPReader(io.RawIOBase):
    """A forward-only file object over an HTTP(S) download that reconnects with a Range request
    at the current offset when the connection drops or a (re)connect fails with a network error,
    a timeout, a 5xx, 408 or 429, so a long stream survives long outages (``retries`` attempts with
    capped exponential backoff)."""

    def __init__(self, url: str, retries: int = 40, timeout: float = 60.0) -> None:
        self.url, self.retries, self.timeout = url, retries, timeout
        self.pos = 0
        self._resp = None

    def _open(self) -> None:
        import requests

        headers = {"Range": f"bytes={self.pos}-"} if self.pos else {}
        resp = requests.get(self.url, stream=True, timeout=self.timeout, headers=headers)
        status = resp.status_code
        if 400 <= status < 500 and status not in (408, 429):
            resp.close()
            raise _NotResumable(f"{self.url}: HTTP {status}")
        resp.raise_for_status()  # 5xx / 408 / 429: retried by readinto
        if self.pos and status != 206:
            resp.close()
            raise _NotResumable(f"{self.url}: the server ignored a Range request; cannot resume at byte {self.pos}")
        self._resp = resp

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def readinto(self, buf) -> int:
        for attempt in range(self.retries + 1):
            try:
                if self._resp is None:
                    self._open()
                n = self._resp.raw.readinto(buf)
                self.pos += n
                return n
            except _NotResumable:
                raise
            except Exception as exc:  # noqa: BLE001 -- network error / timeout / 5xx: reconnect at self.pos
                if attempt == self.retries:
                    raise
                eval_logger.warning(f"{self.url}: {exc}; retrying at byte {self.pos}")
                if self._resp is not None:
                    self._resp.close()
                    self._resp = None
                time.sleep(min(60, 2**attempt))
        return 0

    def close(self) -> None:
        if self._resp is not None:
            self._resp.close()
            self._resp = None
        super().close()


def stream_extract_members(
    source: str, wanted: Mapping[str, str], log_every_gb: float = 5.0, give_up_after: int | None = 1000
) -> list[str]:
    """Stream the tar(.gz) archive at ``source`` (a URL or a local path) once and write the
    members named in ``wanted`` (member name -> destination path; several names may share a
    destination, for archives that lay the same file out differently; a leading ``./`` in the
    archive is ignored). Everything else is skipped, so a large archive is never stored. Each
    file is written to a temporary name and renamed when complete, so an existing destination is
    complete and kept; pass every member you need, present or not, so the layout check below sees
    the matches. Stops as soon as every missing destination is written, and raises early when the
    first ``give_up_after`` regular files match nothing (a different layout). Returns the
    destinations still missing."""
    missing = {dest for dest in wanted.values() if not os.path.exists(dest)}
    if not missing:
        return []
    raw = _ResumableHTTPReader(source) if is_url(source) else open(source, "rb")
    seen = matched = 0
    try:
        with tarfile.open(fileobj=io.BufferedReader(raw, buffer_size=8 << 20), mode="r|*") as tar:
            next_log = log_every_gb
            for member in tar:
                if raw.tell() / 2**30 >= next_log:
                    eval_logger.info(f"{source}: {raw.tell() / 2**30:.0f} GB read, {len(missing)} file(s) to go")
                    next_log += log_every_gb
                if not member.isfile():
                    continue
                seen += 1
                dest = wanted.get(posixpath.normpath(member.name))
                if dest is None:
                    if give_up_after and not matched and seen >= give_up_after:
                        raise IOError(
                            f"{source}: none of the first {seen} files is a wanted member (e.g. archive "
                            f"member {member.name!r}, wanted e.g. {next(iter(wanted))!r}); wrong archive layout?"
                        )
                    continue
                matched += 1
                if dest not in missing:
                    continue
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                tmp = dest + ".partial"
                with tar.extractfile(member) as src, open(tmp, "wb") as out:
                    shutil.copyfileobj(src, out, 1 << 20)
                if os.path.getsize(tmp) != member.size:
                    raise IOError(f"{member.name}: wrote {os.path.getsize(tmp)} of {member.size} bytes")
                os.replace(tmp, dest)
                missing.discard(dest)
                if not missing:
                    break
    finally:
        raw.close()
    return sorted(missing)
