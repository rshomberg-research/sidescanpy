"""
Tests for sidescanpy.export.ml.tiles: swath tiles written for annotation,
COCO boxes read back, and boxes turned into targets in the sonar data.

2026 rshomberg@uri.edu
"""

import json

import numpy as np
import pytest
from PIL import Image

from sidescanpy.export.ml.tiles import (
    TileBox,
    boxes_to_targets,
    read_coco_boxes,
    read_tile_index,
    write_swath_tiles,
)
from tests.unit.export.conftest import make_test_channel


@pytest.fixture
def channel():
    # 200 pings (~0.56 m apart) x 200 samples (0.75 m), georeferenced.
    return make_test_channel(n_pings=200, n_samples=200, georef=True)


def _tiles(channel, tmp_path, **kw):
    return write_swath_tiles(channel, tmp_path, ss_h5="line_a", freq_key="hz_1495000",
                             tile_m=40.0, px_m=0.5, **kw)


def _box_at(record, ping, sample, channel_idx, size_px=6.0):
    x, y = record.window.to_pixel(ping + 0.5, sample + 0.5, channel_idx)
    return TileBox(x - size_px / 2, y - size_px / 2, size_px, size_px)


class TestWriteSwathTiles:
    def test_writes_pngs_and_index(self, channel, tmp_path):
        records = _tiles(channel, tmp_path)
        assert len(records) >= 2
        for r in records:
            im = Image.open(tmp_path / r.file_name)
            assert im.mode == "L"
            assert im.size == (r.window.width, r.window.height)
        index = read_tile_index(tmp_path)
        assert set(index) == {r.file_name for r in records}
        assert index[records[0].file_name].window == records[0].window

    def test_index_extended_when_second_channel_written(self, channel, tmp_path):
        a = _tiles(channel, tmp_path)
        b = write_swath_tiles(channel, tmp_path, ss_h5="line_b", freq_key="hz_1495000",
                              tile_m=40.0, px_m=0.5)
        assert set(read_tile_index(tmp_path / "tiles.json")) == {r.file_name for r in a + b}


class TestReadCocoBoxes:
    def test_reads_boxes_categories_and_scores(self, tmp_path):
        coco = {
            "images": [{"id": 1, "file_name": "tiles/t_0000.png"}, {"id": 2, "file_name": "t_0001.png"}],
            "categories": [{"id": 7, "name": "boulder"}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 7, "bbox": [10, 20, 5, 6]},
                {"id": 2, "image_id": 1, "category_id": 7, "bbox": [1, 2, 3, 4], "score": 0.8},
            ],
        }
        path = tmp_path / "ann.json"
        path.write_text(json.dumps(coco))
        boxes = read_coco_boxes(path)
        assert boxes["t_0001.png"] == []
        assert boxes["t_0000.png"] == [TileBox(10, 20, 5, 6, "boulder", None),
                                       TileBox(1, 2, 3, 4, "boulder", 0.8)]


class TestBoxesToTargets:
    @pytest.mark.parametrize("side", [0, 1])
    def test_box_maps_back_to_ping_sample_side(self, channel, tmp_path, side):
        record = _tiles(channel, tmp_path)[1]
        ping = (record.window.ping_start + record.window.ping_stop) // 2
        [t] = boxes_to_targets(channel, record, [_box_at(record, ping, 120, side)])
        assert t.channel == side
        assert t.extra_attributes["ping_index"] == pytest.approx(ping, abs=1)
        assert t.sample == pytest.approx(120, abs=1)
        assert t.extra_attributes["ss_h5"] == "line_a"
        assert t.ping == int(channel["ping"][t.extra_attributes["ping_index"]])
        assert t.image_id == record.file_name

    def test_position_from_georeferenced_sample(self, channel, tmp_path):
        record = _tiles(channel, tmp_path)[0]
        [t] = boxes_to_targets(channel, record, [_box_at(record, 30, 100, 1)])
        p, s = t.extra_attributes["ping_index"], t.sample
        assert t.latitude == pytest.approx(float(channel["latitude_per_sample"][p, 1, s]))
        assert t.longitude == pytest.approx(float(channel["longitude_per_sample"][p, 1, s]))

    def test_no_position_when_not_georeferenced(self, tmp_path):
        ds = make_test_channel(n_pings=200, n_samples=200, georef=False)
        record = _tiles(ds, tmp_path)[0]
        [t] = boxes_to_targets(ds, record, [_box_at(record, 30, 100, 0)])
        assert t.latitude is None and t.longitude is None
        assert t.extra_attributes["ping_index"] == pytest.approx(30, abs=1)

    def test_detection_is_approximate_annotation_verified(self, channel, tmp_path):
        record = _tiles(channel, tmp_path)[0]
        drawn, detected = boxes_to_targets(channel, record, [
            TileBox(100, 20, 6, 6, "boulder"), TileBox(100, 40, 6, 6, "boulder", score=0.6)])
        assert drawn.position_confidence == "verified"
        assert detected.position_confidence == "approximate"
        assert detected.extra_attributes["score"] == 0.6
        assert detected.category == "boulder"

    def test_box_size_in_metres(self, channel, tmp_path):
        record = _tiles(channel, tmp_path)[0]
        [t] = boxes_to_targets(channel, record, [TileBox(100, 20, 10, 8)])  # 0.5 m pixels
        assert t.extra_attributes["width_m"] == pytest.approx(5.0, rel=0.1)
        assert t.extra_attributes["length_m"] == pytest.approx(4.0, rel=0.15)

    def test_target_ids_unique_per_tile(self, channel, tmp_path):
        record = _tiles(channel, tmp_path)[0]
        targets = boxes_to_targets(channel, record, [TileBox(100, 20 + 10 * k, 4, 4) for k in range(3)])
        assert len({t.target_id for t in targets}) == 3
        assert np.all([t.target_id.startswith("line_a_hz_1495000_0000") for t in targets])
