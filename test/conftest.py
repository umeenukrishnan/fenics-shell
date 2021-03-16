import pytest

def pytest_addoption(parser):
    parser.addoption("--output-results", action="store_true", default=False,
                     help="Output results to files.")

@pytest.fixture
def output_results(request):
    return request.config.getoption("--output-results")
