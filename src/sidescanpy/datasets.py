"""
Openly licensed sidescan recordings for examples and tests.

Each registered file is one raw recording, or one published product
made from recordings, in a public agency archive. Most archives are
multi-gigabyte zips: :func:`fetch` pulls just the one member out with
HTTP range requests, so an example needs a few tens of megabytes instead
of the whole archive. A few archives serve each file on its own URL;
those are downloaded directly. Either way the SHA-256 is checked and the
file cached.

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
import time
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
# Archive servers occasionally answer a range request with something else
# (rate limiting, a gateway error page); wait and try the whole download
# again before giving up.
_RETRY_DELAYS_S = (5.0, 20.0)


@dataclass(frozen=True)
class Source:
    """A public archive and the terms its files come under."""

    title: str
    url: str
    licence: str
    attribution: str


@dataclass(frozen=True)
class RemoteFile:
    """One file in a source archive.

    ``member`` is the file's path inside the zip at ``archive_url``; when
    it is ``None``, ``archive_url`` is the file itself.
    """

    source: str
    archive_url: str
    member: str | None
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
    "usgs_grand_bay_mosaic": Source(
        title=(
            "Locker, S.D., Forde, A.S., and Smith, C.G., 2018, Subbottom and "
            "Sidescan Sonar Data Acquired in 2015 From Grand Bay, Mississippi "
            "and Alabama: U.S. Geological Survey data release (sidescan "
            "mosaic, CleanSweep 3.9, 0.30 m)"
        ),
        url="https://doi.org/10.5066/P9374DKQ",
        licence="U.S. public domain",
        attribution="U.S. Geological Survey, originator of the data.",
    ),
    "jncc_solan_bank": Source(
        title=(
            "Joint Nature Conservation Committee, Raw sidescan sonar data from "
            "Solan Bank Area of Search (CEND 11/08, 2008)"
        ),
        url="https://www.data.gov.uk/dataset/5773e015-c014-416c-aa43-a0397a8fd9e9",
        licence="Open Government Licence v3.0",
        attribution=(
            "Contains public sector information licensed under the Open "
            "Government Licence v3.0."
        ),
    ),
    "jncc_solan_bank_mosaics": Source(
        title=(
            "Joint Nature Conservation Committee, Processed sidescan mosaics "
            "from Solan Bank Area of Search (CEND 11/08, 2008)"
        ),
        url="https://www.data.gov.uk/dataset/6c1e65a9-d45d-443e-b563-f723ac266ac4",
        licence="Open Government Licence v3.0",
        attribution=(
            "Contains public sector information licensed under the Open "
            "Government Licence v3.0."
        ),
    ),
    "jncc_east_rockall": Source(
        title=(
            "Joint Nature Conservation Committee, Raw side scan data from "
            "Anton Dohrn and East Rockall Bank (MV Franklin, 2009)"
        ),
        url="https://www.data.gov.uk/dataset/09203cee-92dd-486b-8838-a525507765fd",
        licence="Open Government Licence v3.0",
        attribution=(
            "Contains public sector information licensed under the Open "
            "Government Licence v3.0."
        ),
    ),
    "jncc_east_rockall_mosaics": Source(
        title=(
            "Joint Nature Conservation Committee, Processed side scan data "
            "from Anton Dohrn and East Rockall Bank (MV Franklin, 2009)"
        ),
        url="https://www.data.gov.uk/dataset/efc2ea5f-c376-4904-b7fb-34cdde65744e",
        licence="Open Government Licence v3.0",
        attribution=(
            "Contains public sector information licensed under the Open "
            "Government Licence v3.0."
        ),
    ),
    "awi_ofobs_ps124": Source(
        title=(
            "Hehemann, L., Purser, A., Boehringer, L., et al., 2021, Side scan "
            "sonar and swath bathymetry collected along OFOBS profiles PS124_26-7 "
            "during POLARSTERN cruise PS124, binned. PANGAEA"
        ),
        url="https://doi.org/10.1594/PANGAEA.939172",
        licence="Creative Commons Attribution 4.0 International (CC BY 4.0)",
        attribution="Alfred Wegener Institute; cite the dataset as given.",
    ),
    "awi_ofobs_ps124_maps": Source(
        title=(
            "Hehemann, L., Boehringer, L., Purser, A., Werner, E., and "
            "Wenzhöfer, F., 2021, Seafloor bathymetric maps derived from side "
            "scan data collected along OFOBS profiles during POLARSTERN cruise "
            "PS124 to the southern Weddell Sea. PANGAEA"
        ),
        url="https://doi.org/10.1594/PANGAEA.939087",
        licence="Creative Commons Attribution 4.0 International (CC BY 4.0)",
        attribution="Alfred Wegener Institute; cite the dataset as given.",
    ),
    "bgs_london_gateway": Source(
        title=(
            "British Geological Survey, National Geoscience Data Centre, "
            "Geophysical Survey WESSEX_London_Gateway: Thames Estuary (2002, "
            "2007, 2008, 2011), item 140325"
        ),
        url="https://webapps.bgs.ac.uk/services/ngdc/accessions/index.html#item140325",
        licence="Open Government Licence v3.0",
        attribution="Contains data supplied by UKRI.",
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
_USGS_GB_MOSAIC = "https://coastal.er.usgs.gov/data-release/doi-P9374DKQ/data/2015-315-FA_SSSMosaic.zip"
_JNCC_SOLAN = (
    "http://s3-eu-west-1.amazonaws.com/data.defra.gov.uk/Marine/"
    "JNCC2008_05IrishSolanGeophysSidescanSolanRawData.zip"
)
_JNCC_SOLAN_MOSAICS = (
    "http://s3-eu-west-1.amazonaws.com/data.defra.gov.uk/Marine/"
    "JNCC2008_05IrishSolanGeophysSidescanSolanBProcessedImages.zip"
)
_JNCC_ROCKALL = (
    "http://s3-eu-west-1.amazonaws.com/data.defra.gov.uk/Marine/"
    "JNCC2009_07AntonDohrnEastRockallGeophysSideScanRawData.zip"
)
_JNCC_ROCKALL_MOSAICS = "http://data.jncc.gov.uk/data/d4ddd363-97eb-4ef6-9b8b-6f019f434103-ProcessedImages.zip"
_PANGAEA_PS124 = "https://download.pangaea.de/dataset/939172/files/"
_PANGAEA_PS124_MAPS = "https://download.pangaea.de/dataset/939087/files/"
_BGS_LG = "https://webservices.bgs.ac.uk/accessions/download/140325?fileName="
_BGS_LG_2007 = _BGS_LG + "88635_61207_LG_2007_SSS_Wrecks.zip"
_BGS_LG_2008 = _BGS_LG + "88635_61208_LG_2008_SSS_Wrecks_Aisha_WK_5057.zip"

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
    "jncc_mid_irish_sea_MIS15_2.xtf": RemoteFile(
        source="jncc_mid_irish_sea", archive_url=_JNCC_MIS,
        member="MIS15_2.XTF", size=54380544,
        sha256="0d29aea61f61ab83cb01bb3f9224ad3074a6e001003f9b4d959014b28c816c4d",
        description="XTF (Isis 7.0.41), Benthos SIS-1624, 100 + 400 kHz, bedforms",
    ),
    "usgs_grand_bay_150602205800.xtf": RemoteFile(
        source="usgs_grand_bay", archive_url=_USGS_GB,
        member="15CCT03_SSS_153_150602205800.xtf", size=5228224,
        sha256="c6acb314bae7f20896d5a890f312332350b6b77262d6c249cdb1fe0b1de1f27a",
        description="XTF (KleinXlt 4.0 / SonarPro 12), Klein, shallow estuary",
    ),
    "usgs_grand_bay_150602133300.xtf": RemoteFile(
        source="usgs_grand_bay", archive_url=_USGS_GB,
        member="15CCT03_SSS_150602133300.xtf", size=16245824,
        sha256="d055ecc68e441340a0c3765187099025ef6d6b072fa07b6264f9e2d86ac8ee00",
        description="XTF (SonarPro), Klein, a sharp seabed edge; recorded altitude unusable",
    ),
    "usgs_grand_bay_150531200100.xtf": RemoteFile(
        source="usgs_grand_bay", archive_url=_USGS_GB,
        member="15CCT03_SSS_150531200100.xtf", size=11581824,
        sha256="1f18a5906a40bf6ad23a3ab59a7ffcbd3282c4f3d1c892179b85e402c2efec2c",
        description="XTF (SonarPro), Klein; recorded altitude is a lost bottom track",
    ),
    "usgs_grand_bay_150602202800.xtf": RemoteFile(
        source="usgs_grand_bay", archive_url=_USGS_GB,
        member="15CCT03_SSS_153_150602202800.xtf", size=235858624,
        sha256="621f02bfc775983d40c873d97e201d713fbd5ba612e77f3654510e6d5219ee34",
        description="XTF (SonarPro), Klein 100 kHz, a whole line under the release's mosaic; "
                    "recorded altitude unusable",
    ),
    "usgs_grand_bay_mosaic_30cmgray.tif": RemoteFile(
        source="usgs_grand_bay_mosaic", archive_url=_USGS_GB_MOSAIC,
        member="30cmgray/30cmgray_LFSS.tif", size=715352534,
        sha256="62c7a8db2f71e69bee5d071d1aed97e4201dbb14d640f080372aa4d3d0dbfd16",
        description="GeoTIFF, the release's sidescan mosaic (CleanSweep 3.9), 0.30 m, UTM 16N; "
                    "8-bit palette, index rises with echo strength (drawn dark), 255 = no data",
    ),
    "jncc_solan_bank_SB04_10.xtf": RemoteFile(
        source="jncc_solan_bank", archive_url=_JNCC_SOLAN,
        member="SB04_10.XTF", size=78805888,
        sha256="fcf448a7a29dedee1c3e709b86ec18a065a17423d96abdf34dde3b9c43965dfb",
        description="XTF (Isis), Benthos SIS-1500, 200 kHz; positions are the ship's, no layback recorded",
    ),
    "jncc_solan_bank_SB4_10.tif": RemoteFile(
        source="jncc_solan_bank_mosaics", archive_url=_JNCC_SOLAN_MOSAICS,
        member="SB4_10.TIF", size=16414280,
        sha256="285e41529d31f7a649f3aabd6e2edf2e488e807432fd3458d14af25372a2dec9",
        description="GeoTIFF (Isis/TritonMap) of line SB04_10, 0.5 m, UTM 30N; 8-bit palette, 0 = no data",
    ),
    "jncc_east_rockall_ER_C1_SS_001_6.xtf": RemoteFile(
        source="jncc_east_rockall", archive_url=_JNCC_ROCKALL,
        member="RawData/Raw_East_Rockall_C1/6_JNCC_FRANKLIN_2009_ER_C1_SS_001.XTF", size=75873792,
        sha256="d57ed7aa813d77f830b86cf4a679ab53da67ded915cd167eaa18dd8efab45170",
        description="XTF (Isis 7.2.69), EdgeTech 4200, 120 + 410 kHz; navigation stored as UTM 29N "
                    "(nav_crs='EPSG:32629')",
    ),
    "jncc_east_rockall_ER_C1_SS_001_6_CMG.tif": RemoteFile(
        source="jncc_east_rockall_mosaics", archive_url=_JNCC_ROCKALL_MOSAICS,
        member="Tif_East_Rockall_C1/6_JNCC_FRANKLIN_2009_ER_C1_SS_001_CMG.TIF", size=56538014,
        sha256="3cc3b5a9bc6db17acc13b9e1a014aa6fae8caa500a34c5d27902d2e99e633e21",
        description="GeoTIFF (Isis/TritonMap) of the same line, 0.4 m, UTM 28N; 8-bit palette, 0 = no data",
    ),
    "awi_ofobs_ps124_20210217004957.jsf": RemoteFile(
        source="awi_ofobs_ps124", archive_url=_PANGAEA_PS124 + "20210217004957_Binned.jsf",
        member=None, size=51075935,
        sha256="d7fe2f041ce05d33273b792349d2a26776c0e56f883982b1331720d4b12fae1c",
        description="EdgeTech 2205 JSF (Discover Bathymetric, binned), towed OFOBS frame, Weddell Sea, "
                    "one of five consecutive files on profile PS124_26-7; USBL positions",
    ),
    "awi_ofobs_ps124_20210217005306.jsf": RemoteFile(
        source="awi_ofobs_ps124", archive_url=_PANGAEA_PS124 + "20210217005306_Binned.jsf",
        member=None, size=51095077,
        sha256="88b4f7081f93fcd8ff2818f927ed5e0981e61bc21884082ae8c1cb8b2b9715f4",
        description="EdgeTech 2205 JSF (Discover Bathymetric, binned), towed OFOBS frame, Weddell Sea, "
                    "one of five consecutive files on profile PS124_26-7; USBL positions",
    ),
    "awi_ofobs_ps124_20210217005615.jsf": RemoteFile(
        source="awi_ofobs_ps124", archive_url=_PANGAEA_PS124 + "20210217005615_Binned.jsf",
        member=None, size=51058773,
        sha256="40d893e40c53fa8dcb207ab060e843ade42a05b7d37b259e689f01e48ded5e7b",
        description="EdgeTech 2205 JSF (Discover Bathymetric, binned), towed OFOBS frame, Weddell Sea, "
                    "one of five consecutive files on profile PS124_26-7; USBL positions",
    ),
    "awi_ofobs_ps124_20210217005924.jsf": RemoteFile(
        source="awi_ofobs_ps124", archive_url=_PANGAEA_PS124 + "20210217005924_Binned.jsf",
        member=None, size=51076371,
        sha256="88538f641d0853226dd31708066b6f115268f399b5118994547ded233def2ee6",
        description="EdgeTech 2205 JSF (Discover Bathymetric, binned), towed OFOBS frame, Weddell Sea, "
                    "one of five consecutive files on profile PS124_26-7; USBL positions",
    ),
    "awi_ofobs_ps124_20210217010233.jsf": RemoteFile(
        source="awi_ofobs_ps124", archive_url=_PANGAEA_PS124 + "20210217010233_Binned.jsf",
        member=None, size=51069870,
        sha256="dfbe72e2a35b0682f01c8c21556a1314351282f51f59715517c9333c1ce09070",
        description="EdgeTech 2205 JSF (Discover Bathymetric, binned), towed OFOBS frame, Weddell Sea, "
                    "one of five consecutive files on profile PS124_26-7; USBL positions",
    ),
    "awi_ofobs_ps124_026_07_ssh_10cm_part1.tiff": RemoteFile(
        source="awi_ofobs_ps124_maps", archive_url=_PANGAEA_PS124_MAPS + "PS124_026_07_SSH_10cm_part1.tiff",
        member=None, size=25387360,
        sha256="dafd87b066aabd3ef2c332f7d3ec3de2ab1eea585192733de6d0bc996ada5062",
        description="GeoTIFF (CARIS HIPS and SIPS), high-frequency sidescan intensity mosaic of profile "
                    "PS124_26-7, 0.10 m, float32, UPS South (EPSG:32761)",
    ),
    "bgs_london_gateway_dynamo_78.xtf": RemoteFile(
        source="bgs_london_gateway", archive_url=_BGS_LG_2007,
        member="140325/88635_61207_LG_2007_SSS_Wrecks/61207_LG_2007_SSS_Dynamo_5100/61207_LG_2007_SSS_Dynamo_WK_5100_78_2007_219_130935_oic.xtf",
        size=27208272,
        sha256="00f101a20a4a55456d247836483615cf4fd6779f21aa471de3f6b000c618fa30",
        description="XTF (Klein, GeoDAS/SonarPro), 400 kHz, 2007, line over the Dynamo (WK 5100) wreck; "
                    "projected navigation (nav_crs='EPSG:32631'), no heading, recorder TVG applied",
    ),
    "bgs_london_gateway_dynamo_80.xtf": RemoteFile(
        source="bgs_london_gateway", archive_url=_BGS_LG_2007,
        member="140325/88635_61207_LG_2007_SSS_Wrecks/61207_LG_2007_SSS_Dynamo_5100/61207_LG_2007_SSS_Dynamo_WK_5100_80_2007_219_130348_oic.xtf",
        size=11237536,
        sha256="f976e64504e50160718256b4175cff1ac39aa4ceca0febedb9b70e7d6cd1bb6b",
        description="XTF (Klein, GeoDAS/SonarPro), 400 kHz, 2007, line over the Dynamo (WK 5100) wreck; "
                    "projected navigation (nav_crs='EPSG:32631'), no heading, recorder TVG applied",
    ),
    "bgs_london_gateway_dynamo_85.xtf": RemoteFile(
        source="bgs_london_gateway", archive_url=_BGS_LG_2007,
        member="140325/88635_61207_LG_2007_SSS_Wrecks/61207_LG_2007_SSS_Dynamo_5100/61207_LG_2007_SSS_Dynamo_WK_5100_85_2007_219_131706_oic.xtf",
        size=16425408,
        sha256="5eb1d74d2fb31aed027e01edf3f07be2574bd6b2f0bda0501ab94ae5c8aed56e",
        description="XTF (Klein, GeoDAS/SonarPro), 400 kHz, 2007, line over the Dynamo (WK 5100) wreck; "
                    "projected navigation (nav_crs='EPSG:32631'), no heading, recorder TVG applied",
    ),
    "bgs_london_gateway_dynamo_86.xtf": RemoteFile(
        source="bgs_london_gateway", archive_url=_BGS_LG_2007,
        member="140325/88635_61207_LG_2007_SSS_Wrecks/61207_LG_2007_SSS_Dynamo_5100/61207_LG_2007_SSS_Dynamo_WK_5100_86_2007_219_132253_oic.xtf",
        size=26789520,
        sha256="95066ed3458fabe5bb39fd54b22d47dfcf5ef746cbea845f239891b5cb380b6b",
        description="XTF (Klein, GeoDAS/SonarPro), 400 kHz, 2007, line over the Dynamo (WK 5100) wreck; "
                    "projected navigation (nav_crs='EPSG:32631'), no heading, recorder TVG applied",
    ),
    "bgs_london_gateway_dynamo_88.xtf": RemoteFile(
        source="bgs_london_gateway", archive_url=_BGS_LG_2007,
        member="140325/88635_61207_LG_2007_SSS_Wrecks/61207_LG_2007_SSS_Dynamo_5100/61207_LG_2007_SSS_Dynamo_WK_5100_88_2007_219_133426_oic.xtf",
        size=14354912,
        sha256="4d1ff5815714b9184d31f0d0da85e36f2770b1ca194299b7386fe94be6ab5cfc",
        description="XTF (Klein, GeoDAS/SonarPro), 400 kHz, 2007, line over the Dynamo (WK 5100) wreck; "
                    "projected navigation (nav_crs='EPSG:32631'), no heading, recorder TVG applied",
    ),
    "bgs_london_gateway_dynamo_90.xtf": RemoteFile(
        source="bgs_london_gateway", archive_url=_BGS_LG_2007,
        member="140325/88635_61207_LG_2007_SSS_Wrecks/61207_LG_2007_SSS_Dynamo_5100/61207_LG_2007_SSS_Dynamo_WK_5100_90_2007_219_135010_oic.xtf",
        size=9178672,
        sha256="1091721a4fb9b29d283fcc8bb0426fcebc6d40ca33ed81e49353577dc46381b2",
        description="XTF (Klein, GeoDAS/SonarPro), 400 kHz, 2007, line over the Dynamo (WK 5100) wreck; "
                    "projected navigation (nav_crs='EPSG:32631'), no heading, recorder TVG applied",
    ),
    "bgs_london_gateway_eastoaze_31.xtf": RemoteFile(
        source="bgs_london_gateway", archive_url=_BGS_LG_2007,
        member="140325/88635_61207_LG_2007_SSS_Wrecks/61207_LG_2007_EastOaze_5056/61207_LG_2007_SSS_EastOaze_WK_5056_31_2007_219_173707_oic.xtf",
        size=25928752,
        sha256="ff60f009b66cb9d5086a47b3510bd085c108bab4e1984cdc43810e3cf0d13c08",
        description="XTF (Klein, GeoDAS/SonarPro), 400 kHz, 2007, line over the East Oaze wreck (WK 5056) wreck; "
                    "projected navigation (nav_crs='EPSG:32631'), no heading, recorder TVG applied",
    ),
    "bgs_london_gateway_ernaboldt_13.xtf": RemoteFile(
        source="bgs_london_gateway", archive_url=_BGS_LG_2007,
        member="140325/88635_61207_LG_2007_SSS_Wrecks/61207_LG_2007_ErnaBoldt_5961/61207_LG_2007_SSS_ErnaBoldt_WK_5961_13_2007_219_115801_oic.xtf",
        size=9504368,
        sha256="7fa9cedd8f15b006f9cf6802344ce0f22743bdc2c9c63dca2c2cc61682b4c4a4",
        description="XTF (Klein, GeoDAS/SonarPro), 400 kHz, 2007, line over the Erna Boldt (WK 5961) wreck; "
                    "projected navigation (nav_crs='EPSG:32631'), no heading, recorder TVG applied",
    ),
    "bgs_london_gateway_storm_04.xtf": RemoteFile(
        source="bgs_london_gateway", archive_url=_BGS_LG_2007,
        member="140325/88635_61207_LG_2007_SSS_Wrecks/61207_LG_2007_Storm_5960/61207_LG_2007_SSS_Storm_WK_5960_400_04_2007_219_105516_oic.xtf",
        size=13470880,
        sha256="70d69ff4ba3fbe6944ce43a28d76f0441054ae4ba499cf7ead7465566ca10d51",
        description="XTF (Klein, GeoDAS/SonarPro), 400 kHz, 2007, line over the SS Storm (WK 5960) wreck; "
                    "projected navigation (nav_crs='EPSG:32631'), no heading, recorder TVG applied",
    ),
    "bgs_london_gateway_amethyst_25.xtf": RemoteFile(
        source="bgs_london_gateway", archive_url=_BGS_LG_2007,
        member="140325/88635_61207_LG_2007_SSS_Wrecks/61207_LG_2007_SSS_Amethyst_5063/61207_LG_2007_SSS_Amethyst_WK_5063_25_2007_219_164208_oic.xtf",
        size=11860424,
        sha256="707e2499e5ffa7fc661ac341d3b87edea69ac14042ff7303ac61f32a932f2000",
        description="XTF (Klein, GeoDAS/SonarPro), 400 kHz, 2007, line over the Amethyst (WK 5063) wreck; "
                    "projected navigation (nav_crs='EPSG:32631'), no heading, recorder TVG applied",
    ),
    "bgs_london_gateway_ash_11.xtf": RemoteFile(
        source="bgs_london_gateway", archive_url=_BGS_LG_2007,
        member="140325/88635_61207_LG_2007_SSS_Wrecks/61207_LG_2007_SSS_Ash_5013/61207_LG_2007_SSS_Ash_WK_5013_11_2007_220_121722_oic.xtf",
        size=11109584,
        sha256="738c6f9f9629231ca9367dabb0f8e50dfac683eed909398d1243841b6f893699",
        description="XTF (Klein, GeoDAS/SonarPro), 400 kHz, 2007, line over the Ash (WK 5013) wreck; "
                    "projected navigation (nav_crs='EPSG:32631'), no heading, recorder TVG applied",
    ),
    "bgs_london_gateway_aisha_05.xtf": RemoteFile(
        source="bgs_london_gateway", archive_url=_BGS_LG_2008,
        member="140325/88635_61208_LG_2008_SSS_Wrecks_Aisha_WK_5057/88635_61208_LG_2008_SSS_Wrecks_Aisha_WK_5057/61208_LG_2008_SSS_Aisha_WK_5057_400_05_095_113434_oic.xtf",
        size=27749808,
        sha256="cb8bf7a31833b6b5e00e92528fe4cae11fdb27071f1d4a6676cdd915c6e25073",
        description="XTF (Klein, GeoDAS/SonarPro), 400 kHz, 2008, line over the Aisha (WK 5057) wreck; "
                    "projected navigation (nav_crs='EPSG:32631'), no heading, recorder TVG applied",
    ),
}

BGS_LONDON_GATEWAY_WRECKS: dict[str, tuple[float, float]] = {
    "Amethyst": (364468.0, 5708659.0),
    "Ancient": (325490.0, 5708230.0),
    "Argus": (359499.0, 5706071.0),
    "Ash": (360905.0, 5706497.0),
    "Atherton": (359708.0, 5706186.0),
    "Dynamo": (401449.0, 5743755.0),
    "East Oaze": (362786.0, 5707385.0),
    "Erna Boldt": (403551.0, 5746997.0),
    "Letchworth": (357544.0, 5705592.0),
    "London": (343115.0, 5707365.0),
    "Pottery": (346619.0, 5706276.0),
    "SS Storm": (406001.0, 5747115.0),
    "Unknown wreck": (375530.0, 5714052.0),
    "Aisha": (363982.0, 5707656.0),
}
"""Catalogued wreck centres of the London Gateway survey, as given in the BGS
record (UTM zone 31N, WGS84; easting, northing in meters). Centres only: a
line can show its wreck tens of meters away along track, so treat them as
approximate positions, not boxes."""


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
        does not match the registry, after two retries (5 s and 20 s
        apart, each logged as a WARNING). The partial file is removed.
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
    delays = (*_RETRY_DELAYS_S, None)
    for attempt, delay in enumerate(delays, start=1):
        try:
            _download(name, entry, path)
            return path
        except (OSError, zipfile.BadZipFile) as exc:
            if delay is None:
                raise OSError(f"{name}: download failed after {attempt} attempts: {exc}") from exc
            logger.warning("%s: download attempt %d failed (%s); retrying in %.0f s",
                           name, attempt, exc, delay)
            time.sleep(delay)
    raise AssertionError("unreachable")


def _download(name: str, entry: "RemoteFile", path: Path) -> None:
    """Extract one archive member to ``path``, checking size and SHA-256."""
    tmp = path.with_name(path.name + ".part")
    logger.debug("Fetching %s from %s (member %s)", name, entry.archive_url, entry.member)
    try:
        if entry.member is None:
            req = urllib.request.Request(entry.archive_url, headers={"User-Agent": _USER_AGENT})
            with _urlopen(req) as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst, length=_RANGE_BUFFER)
        else:
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


def _urlopen(req: urllib.request.Request):
    """``urlopen`` with the module timeout (a seam for the tests)."""
    return urllib.request.urlopen(req, timeout=_TIMEOUT_S)


def _open_remote_zip(url: str) -> zipfile.ZipFile:
    reader = io.BufferedReader(_HttpRangeReader(url), buffer_size=_RANGE_BUFFER)
    return zipfile.ZipFile(reader)
