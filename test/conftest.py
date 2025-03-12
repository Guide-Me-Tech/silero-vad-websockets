import pytest
import sys
import os
import logging

# Add the parent directory to the path so we can import the server module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


@pytest.fixture(scope="session")
def test_audio_file():
    """Return the path to the test audio file."""
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "input.wav"
    )


@pytest.fixture(scope="session")
def test_output_file():
    """Return the path to the test output file."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "output.wav")
