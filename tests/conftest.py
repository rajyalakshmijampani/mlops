import pytest
import os

def pytest_addoption(parser):
    parser.addoption(
        "--uri", action="store"
    )

@pytest.fixture
def uri(request):
    return request.config.getoption("--uri")
