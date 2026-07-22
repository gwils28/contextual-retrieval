from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("*.yaml")), ids=lambda p: p.name)
def test_protocol_is_valid_yaml(path):
    assert isinstance(yaml.safe_load(path.read_text()), dict)
