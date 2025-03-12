import unittest
import asyncio
import sys
import os
import pytest

# Add the parent directory to the path so we can import the server module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import SileroVAD, initialize_vad


class TestServer(unittest.TestCase):
    """Test cases for the server module."""

    def test_silero_vad_init(self):
        """Test that SileroVAD can be initialized with default parameters."""
        vad = SileroVAD()
        self.assertIsNotNone(vad)
        self.assertEqual(vad.threshold, 0.5)
        self.assertEqual(vad.min_silence_duration_ms, 100)
        self.assertEqual(vad.speech_pad_ms, 30)
        vad.destroy()

    def test_silero_vad_custom_params(self):
        """Test that SileroVAD can be initialized with custom parameters."""
        vad = SileroVAD(threshold=0.7, min_silence_duration_ms=200, speech_pad_ms=50)
        self.assertIsNotNone(vad)
        self.assertEqual(vad.threshold, 0.7)
        self.assertEqual(vad.min_silence_duration_ms, 200)
        self.assertEqual(vad.speech_pad_ms, 50)
        vad.destroy()


@pytest.mark.asyncio
async def test_initialize_vad():
    """Test that the initialize_vad function returns a valid SileroVAD instance."""
    vad = await initialize_vad()
    assert vad is not None
    assert isinstance(vad, SileroVAD)
    vad.destroy()


if __name__ == "__main__":
    unittest.main()
