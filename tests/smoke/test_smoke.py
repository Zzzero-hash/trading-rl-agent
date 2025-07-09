import pytest


@pytest.mark.smoke
def test_basic_math():
    assert 1 + 1 == 2
