"""Tests for Prepare.check_tns target-identification validation and
input-file deduplication."""

import pytest

from autophot.pipeline.prepare import Prepare, _dedupe_identical_files


def _prep(**overrides):
    cfg = {"wdir": "/tmp", "catalog": {}, "wcs": {}}
    cfg.update(overrides)
    return Prepare(default_input=cfg)


def test_name_and_coords_conflict_exits():
    prep = _prep(target_name="SN2024pba", target_ra=25.3, target_dec=49.0)
    with pytest.raises(SystemExit):
        prep.check_tns()


def test_name_and_single_coord_conflict_exits():
    # A stray lone coordinate is the same ambiguity as a full pair.
    prep = _prep(target_name="SN2024pba", target_ra=25.3)
    with pytest.raises(SystemExit):
        prep.check_tns()


def test_coords_only_returned_directly(tmp_path):
    prep = _prep(wdir=str(tmp_path), target_ra=25.3, target_dec=49.0)
    coords = prep.check_tns()
    assert coords == {"ra": 25.3, "dec": 49.0}


def test_default_transient_name_not_a_conflict(tmp_path):
    # "Transient" is the shipped placeholder, not a real lookup target.
    prep = _prep(
        wdir=str(tmp_path),
        target_name="Transient",
        target_ra=25.3,
        target_dec=49.0,
    )
    coords = prep.check_tns()
    assert coords["ra"] == 25.3


def test_blank_name_not_a_conflict(tmp_path):
    prep = _prep(
        wdir=str(tmp_path),
        target_name="   ",
        target_ra=25.3,
        target_dec=49.0,
    )
    coords = prep.check_tns()
    assert coords["ra"] == 25.3


def test_transient_placeholder_alone_skips_tns(tmp_path):
    # The shipped default name must not trigger a real TNS/SIMBAD query.
    prep = _prep(wdir=str(tmp_path), target_name="Transient")
    assert prep.check_tns() == {}


# ---------------------------------------------------------------------------
# _dedupe_identical_files
# ---------------------------------------------------------------------------


def test_dedupe_removes_byte_identical(tmp_path):
    a = tmp_path / "2024-01-01.fits"
    b = tmp_path / "2024-01-02.fits"
    c = tmp_path / "other.fits"
    a.write_bytes(b"\x01" * 2048)
    b.write_bytes(b"\x01" * 2048)  # identical content, different name
    c.write_bytes(b"\x02" * 2048)  # same size, different content

    kept, dropped = _dedupe_identical_files([str(a), str(b), str(c)])
    assert len(kept) == 2
    assert len(dropped) == 1
    kept_path, dropped_path = dropped[0]
    # canonical survivor is the first path in sorted order
    assert kept_path == str(a)
    assert dropped_path == str(b)
    assert str(c) in kept


def test_dedupe_unique_files_unchanged(tmp_path):
    files = []
    for i, payload in enumerate((b"a" * 100, b"b" * 200, b"c" * 300)):
        p = tmp_path / f"f{i}.fits"
        p.write_bytes(payload)
        files.append(str(p))
    kept, dropped = _dedupe_identical_files(files)
    assert kept == files
    assert dropped == []
