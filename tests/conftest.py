import pytest

from research_agent.builtin_tools import default_registry


@pytest.fixture
def registry(tmp_path):
    return default_registry(tmp_path / "notes")
