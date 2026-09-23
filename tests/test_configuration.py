from pathlib import Path

from lace.configuration import get_data_path, get_nyx_path, set_data_path, set_nyx_path


def test_paths_precedence_and_preservation(tmp_path):
    config = tmp_path / "paths.toml"
    config.write_text("[other]\nvalue = 2\n[paths]\nnyx_path = '/nyx'\n")
    assert get_nyx_path(config_path=config) == Path("/nyx")
    assert get_nyx_path("/explicit", config) == Path("/explicit")
    set_data_path("/assets", config)
    assert get_data_path(config_path=config) == Path("/assets")
    assert "[other]" in config.read_text()
    set_nyx_path("/new-nyx", config)
    assert get_nyx_path(config_path=config) == Path("/new-nyx")

def test_blank_data_path_rejected():
    import pytest
    with pytest.raises(ValueError, match="blank"):
        get_data_path("   ")
