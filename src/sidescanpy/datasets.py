"""
Openly licensed sidescan recordings for examples and tests.

Each registered file is one raw recording inside a public agency archive
(a multi-gigabyte zip). :func:`fetch` pulls just that member out with HTTP
range requests, checks its SHA-256 and caches it, so an example needs a
few tens of megabytes instead of the whole archive.

>>> from sidescanpy import datasets
>>> path = datasets.fetch("jncc_wight_barfleur_20130321174522.jsf")
>>> print(datasets.citation("jncc_wight_barfleur_20130321174522.jsf"))

The cache is ``SIDESCANPY_DATA_DIR`` when set, otherwise
``~/.cache/sidescanpy`` (``$XDG_CACHE_HOME/sidescanpy`` if that is set).
A file already in the cache with the right checksum is used as is, so a
directory of pre-downloaded files works offline.

Only standard-library modules are used. Downloads log at DEBUG; call
:func:`sidescanpy.log.configure_logging` with ``"DEBUG"`` to see them.

2026 rshomberg@uri.edu
"""

import hashlib
import io
import logging
import os
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Some archive servers refuse range requests from clients that don't look
# like a browser.
_USER_AGENT = "Mozilla/5.0 (compatible; sidescanpy dataset fetch)"
_RANGE_BUFFER = 4 * 1024 * 1024
_TIMEOUT_S = 60


@dataclass(frozen=True)
class Source:
    """A public archive and the terms its files come under."""

    title: str
    url: str
    licence: str
    attribution: str


@dataclass(frozen=True)
class RemoteFile:
    """One recording inside a source archive."""

    source: str
    archive_url: str
    member: str
    size: int
    sha256: str
    description: str


SOURCES: dict[str, Source] = {
    "jncc_wight_barfleur": Source(
        title=(
            "Joint Nature Conservation Committee, Raw side scan sonar data "
            "from Wight-Barfleur Reef SCI (CEND 03/13, 2013)"
        ),
        url="https://www.data.gov.uk/dataset/5298a3dd-af03-4509-b806-ab9f884816ae",
        licence="Open Government Licence v3.0",
        attribution=(
            "Contains public sector information licensed under the Open "
            "Government Licence v3.0."
        ),
    ),
    "jncc_mid_irish_sea": Source(
        title=(
            "Joint Nature Conservation Committee, Raw sidescan sonar data from "
            "Submarine structures in the mid-Irish Sea Area of Search (CEND 11/08, 2008)"
        ),
        url="https://www.data.gov.uk/dataset/4a67eeb2-3dce-40b6-96ca-66cc25eeeada",
        licence="Open Government Licence v3.0",
        attribution=(
            "Contains public sector information licensed under the Open "
            "Government Licence v3.0."
        ),
    ),
    "usgs_grand_bay": Source(
        title=(
            "Locker, S.D., Forde, A.S., and Smith, C.G., 2018, Subbottom and "
            "Sidescan Sonar Data Acquired in 2015 From Grand Bay, Mississippi "
            "and Alabama: U.S. Geological Survey data release"
        ),
        url="https://doi.org/10.5066/P9374DKQ",
        licence="U.S. public domain",
        attribution="U.S. Geological Survey, originator of the data.",
    ),
}

_JNCC_WB = (
    "http://s3-eu-west-1.amazonaws.com/data.defra.gov.uk/Marine/"
    "JNCC2013_03WightBReefRawAcousticss_raw.zip"
)
_JNCC_MIS = (
    "http://s3-eu-west-1.amazonaws.com/data.defra.gov.uk/Marine/"
    "JNCC2008_05IrishSolanGeophysSidescanMidIrishRawData.zip"
)
_USGS_GB = "https://coastal.er.usgs.gov/data-release/doi-P9374DKQ/data/2015-315-FA_xtf.zip"

REGISTRY: dict[str, RemoteFile] = {
    "jncc_wight_barfleur_20130321174522.jsf": RemoteFile(
        source="jncc_wight_barfleur", archive_url=_JNCC_WB,
        member="BoxB2/20130321174522.jsf", size=40369190,
        sha256="4a657d2d7bdb82e00ed3a99e83f72ccf0db8ca88baac14980e59665d337b0b51",
        description="EdgeTech JSF (Discover), 219 + 521 kHz, 288 pings, no recorded altitude",
    ),
    "jncc_wight_barfleur_20130321174522H.xtf": RemoteFile(
        source="jncc_wight_barfleur", archive_url=_JNCC_WB,
        member="BoxB2/20130321174522H.xtf", size=26782720,
        sha256="bd59c2fabed70610582b2d3d8e690f4b439b9862eabad025c23259da3c90cd69",
        description="XTF (Discover 7.22), high frequency of the same pings as the JSF",
    ),
    "jncc_wight_barfleur_20130321174522L.xtf": RemoteFile(
        source="jncc_wight_barfleur", archive_url=_JNCC_WB,
        member="BoxB2/20130321174522L.xtf", size=13409664,
        sha256="b61e773df4d39480f10128e0b4e0770b31129e3a5406addbe8408845c1c981b9",
        description="XTF (Discover 7.22), low frequency of the same pings as the JSF",
    ),
    "jncc_mid_irish_sea_MIS11_0.xtf": RemoteFile(
        source="jncc_mid_irish_sea", archive_url=_JNCC_MIS,
        member="MIS11_0.XTF", size=41068544,
        sha256="2e387f1d3503530668f00ece8d282d7a74875b8ec84376629586ad337a11c7ff",
        description="XTF (Isis 7.0.41), Benthos SIS-1624, 100 + 400 kHz, altitude recorded",
    ),
    "jncc_mid_irish_sea_MIS13_2.xtf": RemoteFile(
        source="jncc_mid_irish_sea", archive_url=_JNCC_MIS,
        member="MIS13_2.XTF", size=3794944,
        sha256="32a1073ab2819da79bc393d759a327b21048480de0eae9cc02a1dec51b68f412",
        description="XTF (Isis 7.0.41), Benthos SIS-1624, a short line",
    ),
    "usgs_grand_bay_150602205800.xtf": RemoteFile(
        source="usgs_grand_bay", archive_url=_USGS_GB,
        member="15CCT03_SSS_153_150602205800.xtf", size=5228224,
        sha256="c6acb314bae7f20896d5a890f312332350b6b77262d6c249cdb1fe0b1de1f27a",
        description="XTF (KleinXlt 4.0 / SonarPro 12), Klein, shallow estuary",
    ),
    "usgs_grand_bay_150531200100.xtf": RemoteFile(
        source="usgs_grand_bay", archive_url=_USGS_GB,
        member="15CCT03_SSS_150531200100.xtf", size=11581824,
        sha256="1f18a5906a40bf6ad23a3ab59a7ffcbd3282c4f3d1c892179b85e402c2efec2c",
        description="XTF (SonarPro), Klein; recorded altitude is a lost bottom track",
    ),
}


def list_datasets() -> list[str]:
    """Names :func:`fetch` accepts, sorted."""
    return sorted(REGISTRY)


def data_dir() -> Path:
    """The cache directory :func:`fetch` uses when none is given."""
    env = os.environ.get("SIDESCANPY_DATA_DIR")
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "sidescanpy"


def citation(name: str) -> str:
    """Source, licence and attribution text for a registered file."""
    source = SOURCES[_entry(name).source]
    return f"{source.title}. {source.url}. {source.licence}. {source.attribution}"


def fetch(name: str, *, directory: str | Path | None = None) -> Path:
    """Return a local path to a registered recording, downloading it if needed.

    Parameters
    ----------
    name : str
        A name from :func:`list_datasets`.
    directory : str or Path, optional
        Where to look for and store the file. Defaults to :func:`data_dir`.

    Returns
    -------
    Path
        The verified local file.

    Raises
    ------
    KeyError
        If ``name`` is not registered.
    OSError
        If the download fails or the downloaded file's size or SHA-256
        does not match the registry (the partial file is removed).
    """
    entry = _entry(name)
    root = Path(directory).expanduser() if directory is not None else data_dir()
    path = root / name
    if path.exists():
        if _sha256(path) == entry.sha256:
            logger.debug("Using cached %s", path)
            return path
        logger.warning("Cached %s has the wrong checksum; downloading it again", path)

    root.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    logger.debug("Fetching %s from %s (member %s)", name, entry.archive_url, entry.member)
    try:
        with _open_remote_zip(entry.archive_url) as zf, zf.open(entry.member) as src, \
                open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst, length=_RANGE_BUFFER)
        size = tmp.stat().st_size
        digest = _sha256(tmp)
        if size != entry.size or digest != entry.sha256:
            raise OSError(
                f"{name}: downloaded {size} bytes with SHA-256 {digest}, "
                f"expected {entry.size} bytes with {entry.sha256}"
            )
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def _entry(name: str) -> RemoteFile:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown dataset {name!r}; choose from {list_datasets()}") from None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class _HttpRangeReader(io.RawIOBase):
    """Read-only, seekable view of a remote file through HTTP range requests."""

    def __init__(self, url: str):
        self.url = url
        self.pos = 0
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
            self.length = int(resp.headers["Content-Length"])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.length}[whence]
        self.pos = max(0, base + offset)
        return self.pos

    def readinto(self, buffer) -> int:
        n = min(len(buffer), self.length - self.pos)
        if n <= 0:
            return 0
        req = urllib.request.Request(self.url, headers={
            "User-Agent": _USER_AGENT,
            "Range": f"bytes={self.pos}-{self.pos + n - 1}",
        })
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
            if resp.status != 206:
                raise OSError(f"{self.url} ignored the range request (HTTP {resp.status})")
            data = resp.read()
        buffer[: len(data)] = data
        self.pos += len(data)
        return len(data)


def _open_remote_zip(url: str) -> zipfile.ZipFile:
    reader = io.BufferedReader(_HttpRangeReader(url), buffer_size=_RANGE_BUFFER)
    return zipfile.ZipFile(reader)
