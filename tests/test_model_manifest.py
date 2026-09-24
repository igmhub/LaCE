from pathlib import Path

import pytest

from lace.emulator.model_manifest import ModelBundleError, load_manifest, write_manifest


def _bundle(tmp_path, label):
    meta = tmp_path / ("meta.npy" if label == "full.pkl" else "meta_nyx_1.npy")
    checkpoint = tmp_path / f"n0_{label}"
    normalization = tmp_path / "ff_mpgcen.npy"
    for path in (meta, checkpoint, normalization):
        path.write_bytes(path.name.encode())
    write_manifest(tmp_path, "CH24_nyxcen_gpr", label, [meta, checkpoint], normalization,
                   None if label == "full.pkl" else "nyx_1", {"archive_class": "NyxArchive"})
    return meta, checkpoint, normalization


def test_each_l1o_model_gets_its_own_manifest(tmp_path):
    _, _, normalization = _bundle(tmp_path, "drop_nyx_1.pkl")
    _bundle(tmp_path, "drop_nyx_2.pkl")
    assert (tmp_path / "manifest_drop_nyx_1.json").is_file()
    assert (tmp_path / "manifest_drop_nyx_2.json").is_file()
    assert load_manifest(tmp_path, "CH24_nyxcen_gpr", "drop_nyx_1.pkl", normalization)


def test_manifest_rejects_altered_normalization_before_loading(tmp_path):
    _, _, normalization = _bundle(tmp_path, "full.pkl")
    normalization.write_bytes(b"altered")
    with pytest.raises(ModelBundleError, match="normalization"):
        load_manifest(tmp_path, "CH24_nyxcen_gpr", "full.pkl", normalization)


def test_manifest_rejects_incomplete_inventory(tmp_path):
    _, checkpoint, normalization = _bundle(tmp_path, "full.pkl")
    checkpoint.unlink()
    with pytest.raises(ModelBundleError, match="missing"):
        load_manifest(tmp_path, "CH24_nyxcen_gpr", "full.pkl", normalization)


def test_manifest_rejects_incompatible_scikit_learn(tmp_path):
    import json
    _, _, normalization = _bundle(tmp_path, "full.pkl")
    manifest = tmp_path / "manifest.json"
    content = json.loads(manifest.read_text())
    content["dependencies"]["scikit-learn"] = "0.0.0"
    manifest.write_text(json.dumps(content))
    with pytest.raises(ModelBundleError, match="scikit-learn"):
        load_manifest(tmp_path, "CH24_nyxcen_gpr", "full.pkl", normalization)
