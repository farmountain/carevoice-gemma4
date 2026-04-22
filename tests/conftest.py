"""pytest configuration for CareVoice stress-test suite."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pytest
from gemma_hackathon.intake_assistant import IntakeAssistant
from gemma_hackathon.scenarios import SAMPLE_SCENARIOS

def pytest_addoption(parser):
    parser.addoption("--use-real-model", action="store_true", default=False,
                     help="Run integration tests against real Gemma 4 model (requires GPU)")
    parser.addoption("--max-scenarios", type=int, default=50,
                     help="Max scenarios to run in extended corpus tests (default: 50)")

@pytest.fixture(scope="session")
def mock_assistant():
    return IntakeAssistant.mock()

@pytest.fixture(scope="session")
def sample_scenarios():
    return SAMPLE_SCENARIOS

@pytest.fixture(scope="session")
def extended_scenarios():
    try:
        from tests.corpus.generator import EXTENDED_CORPUS
        return EXTENDED_CORPUS
    except ImportError:
        # Try relative import
        try:
            sys.path.insert(0, os.path.dirname(__file__))
            from corpus.generator import EXTENDED_CORPUS
            return EXTENDED_CORPUS
        except ImportError:
            pytest.skip("Extended corpus not available")

@pytest.fixture(scope="session")
def max_scenarios(request):
    return request.config.getoption("--max-scenarios")

@pytest.fixture(scope="session")
def using_real_model(request):
    """True when --use-real-model flag is passed; False otherwise (mock mode)."""
    return request.config.getoption("--use-real-model")

@pytest.fixture(scope="session")
def assistant(request, mock_assistant):
    if request.config.getoption("--use-real-model"):
        try:
            return IntakeAssistant.load()
        except Exception as e:
            pytest.skip(f"Real model load failed: {e}")
    return mock_assistant
