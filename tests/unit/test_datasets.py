"""
sidescanpy.datasets: registry, cache and verified fetch, all offline.

A local zip stands in for an agency archive; one test serves it over
HTTP with range support to check the range reader itself.

2026 rshomberg@uri.edu
"""

import dataclasses
import hashlib
import http.server
import threading
import zipfile
from pathlib import Path

import pytest

from sidescanpy import datasets
from sidescanpy.datasets import RemoteFile

PAYLOAD = bytes(range(256)) * 4000  # ~1 MB, compressible
MEMBER = "Box/line_001.xtf"


@pytest.fixture
def archive(tmp_path) -> Path:
    path = tmp_path / "archive.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("Box/other.txt", b"not this one")
        zf.writestr(MEMBER, PAYLOAD)
    return path


@pytest.fixture
def registered(monkeypatch, archive):
    """Register one file whose archive opens from the local zip."""
    entry = RemoteFile(
        source="usgs_grand_bay", archive_url="https://example.invalid/archive.zip",
        member=MEMBER, size=len(PAYLOAD),
        sha256=hashlib.sha256(PAYLOAD).hexdigest(), description="test",
    )
    monkeypatch.setitem(datasets.REGISTRY, "test_line.xtf", entry)
    calls = []

    def fake_open(url):
        calls.append(url)
        return zipfile.ZipFile(archive)

    monkeypatch.setattr(datasets, "_open_remote_zip", fake_open)
    return calls


@pytest.mark.unit
class TestRegistry:
    def test_every_entry_has_a_known_source(self):
        assert datasets.list_datasets()
        for entry in datasets.REGISTRY.values():
            assert entry.source in datasets.SOURCES
            assert len(entry.sha256) == 64 and entry.size > 0

    def test_citation_names_licence(self):
        for name in datasets.list_datasets():
            source = datasets.SOURCES[datasets.REGISTRY[name].source]
            assert source.licence in datasets.citation(name)

    def test_unknown_name_raises_with_choices(self):
        with pytest.raises(KeyError, match="choose from"):
            datasets.fetch("nope.xtf")


@pytest.mark.unit
class TestDataDir:
    def test_env_var_wins(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SIDESCANPY_DATA_DIR", str(tmp_path))
        assert datasets.data_dir() == tmp_path

    def test_xdg_cache_when_no_env_var(self, monkeypatch, tmp_path):
        monkeypatch.delenv("SIDESCANPY_DATA_DIR", raising=False)
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        assert datasets.data_dir() == tmp_path / "sidescanpy"


@pytest.mark.unit
class TestFetch:
    def test_fetch_extracts_member_when_not_cached(self, registered, tmp_path):
        path = datasets.fetch("test_line.xtf", directory=tmp_path / "cache")
        assert path.read_bytes() == PAYLOAD
        assert len(registered) == 1
        assert not list(path.parent.glob("*.part"))

    def test_fetch_uses_cache_when_checksum_matches(self, registered, tmp_path):
        datasets.fetch("test_line.xtf", directory=tmp_path)
        datasets.fetch("test_line.xtf", directory=tmp_path)
        assert len(registered) == 1

    def test_fetch_uses_env_dir_when_no_directory(self, registered, monkeypatch, tmp_path):
        monkeypatch.setenv("SIDESCANPY_DATA_DIR", str(tmp_path))
        assert datasets.fetch("test_line.xtf").parent == tmp_path

    def test_fetch_replaces_corrupt_cache(self, registered, tmp_path):
        (tmp_path / "test_line.xtf").write_bytes(b"corrupt")
        path = datasets.fetch("test_line.xtf", directory=tmp_path)
        assert path.read_bytes() == PAYLOAD
        assert len(registered) == 1

    def test_fetch_retries_when_archive_read_fails(self, registered, archive, monkeypatch, tmp_path):
        monkeypatch.setattr(datasets, "_RETRY_DELAYS_S", (0.0, 0.0))
        attempts = []

        def flaky_open(url):
            attempts.append(url)
            if len(attempts) < 3:
                raise zipfile.BadZipFile("File is not a zip file")
            return zipfile.ZipFile(archive)

        monkeypatch.setattr(datasets, "_open_remote_zip", flaky_open)
        path = datasets.fetch("test_line.xtf", directory=tmp_path)
        assert path.read_bytes() == PAYLOAD
        assert len(attempts) == 3

    def test_fetch_raises_oserror_when_every_attempt_fails(self, registered, monkeypatch, tmp_path):
        monkeypatch.setattr(datasets, "_RETRY_DELAYS_S", (0.0, 0.0))

        def broken_open(url):
            raise zipfile.BadZipFile("File is not a zip file")

        monkeypatch.setattr(datasets, "_open_remote_zip", broken_open)
        with pytest.raises(OSError, match="after 3 attempts"):
            datasets.fetch("test_line.xtf", directory=tmp_path)
        assert not list(tmp_path.glob("*.part"))

    def test_fetch_raises_and_cleans_up_when_checksum_differs(
        self, registered, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(datasets, "_RETRY_DELAYS_S", (0.0, 0.0))
        bad = dataclasses.replace(datasets.REGISTRY["test_line.xtf"], sha256="0" * 64)
        monkeypatch.setitem(datasets.REGISTRY, "test_line.xtf", bad)
        cache = tmp_path / "cache"
        with pytest.raises(OSError, match="SHA-256"):
            datasets.fetch("test_line.xtf", directory=cache)
        assert list(cache.iterdir()) == []


class _RangeHandler(http.server.SimpleHTTPRequestHandler):
    """Serves one file, honouring single byte ranges."""

    def do_HEAD(self):
        data = self.server.data
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()

    def do_GET(self):
        data = self.server.data
        start, end = self.headers["Range"].removeprefix("bytes=").split("-")
        chunk = data[int(start): int(end) + 1]
        self.send_response(206)
        self.send_header("Content-Length", str(len(chunk)))
        self.end_headers()
        self.wfile.write(chunk)

    def log_message(self, *args):
        pass


@pytest.mark.unit
class TestRangeReader:
    def test_remote_zip_reads_member_through_ranges(self, archive):
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
        server.data = archive.read_bytes()
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_address[1]}/archive.zip"
            with datasets._open_remote_zip(url) as zf:
                assert zf.read(MEMBER) == PAYLOAD
        finally:
            server.shutdown()
