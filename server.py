import asyncio
import json
import logging
import os
import struct
import websockets
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple
from silero_vad import load_silero_vad, get_speech_timestamps
from typing import Tuple
import wave
import requests
import io
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
SAMPLE_RATE = 16000
MAX_MESSAGE_SIZE = 10000  # Maximum size of incoming messages
PING_INTERVAL = 30  # Seconds
PING_TIMEOUT = 60  # Seconds


def load_envs(env_type: str = "dev"):
    if env_type == "dev":
        load_dotenv(".env.dev")
    else:
        load_dotenv(".env")


load_envs(env_type=os.getenv("ENVIRONMENT") or "dev")


class NoSpeechDetected(Exception):
    def __init__(self, message="No Speech Detected"):
        self.message = message
        super().__init__(self.message)


class Segment:
    """Represents a speech segment with start and end times."""

    def __init__(self, speech_start_at: float, speech_end_at: Optional[float] = None):
        self.speech_start_at = speech_start_at
        self.speech_end_at = speech_end_at

    def to_dict(self) -> Dict[str, float]:
        result = {"speech_start_at": self.speech_start_at}
        if self.speech_end_at is not None:
            result["speech_end_at"] = self.speech_end_at
        return result


class SileroVAD:
    """Voice Activity Detection using Silero VAD model."""

    def __init__(
        self,
        threshold: float = 0.5,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
    ):
        self.threshold = threshold
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.sample_rate = SAMPLE_RATE

        # Load the model
        self._load_model()

    def _load_model(self):
        """Load the Silero VAD model using the official library."""
        try:
            self.model = load_silero_vad()
            logger.info("Loaded Silero VAD model")
        except Exception as e:
            logger.error(f"Failed to load Silero VAD model: {e}")
            raise

    def detect(self, audio_samples: np.ndarray) -> List[Segment]:
        """Detect speech segments in audio samples."""
        try:
            # Convert numpy array to torch tensor
            tensor_samples = torch.tensor(audio_samples)

            # Get speech timestamps
            timestamps = get_speech_timestamps(
                tensor_samples,
                self.model,
                threshold=self.threshold,
                min_silence_duration_ms=self.min_silence_duration_ms,
                speech_pad_ms=self.speech_pad_ms,
                return_seconds=True,  # Return timestamps in seconds
            )

            # Convert timestamps to Segment objects
            segments = []
            for ts in timestamps:
                start = ts["start"]
                end = ts.get("end")
                segments.append(Segment(speech_start_at=start, speech_end_at=end))

            return segments

        except Exception as e:
            logger.error(f"Error during VAD detection: {e}")
            return []

    def memory_vad_check(self, raw_audio: bytearray) -> Tuple[bool, List[Segment]]:
        """
        Returns True if speech is still ongoing.
        Returns False if we've reached at least `min_silence_ms` of silence (i.e. no more speech).
        Raises NoSpeechDetected if no speech is found at all.
        """

        # Convert raw PCM (16-bit, mono) to a NumPy array of floats in [-1, 1]
        # (Silero's default usage can also accept int16, but normalizing is a bit safer)
        raw_audio_copy = raw_audio[:]
        audio_float32 = (
            np.frombuffer(raw_audio_copy, dtype=np.int16).astype(np.float32) / 32768.0
        )

        if len(audio_float32) == 0:
            raise NoSpeechDetected("Audio buffer is empty.")

        # Get timestamps
        speech_timestamps = get_speech_timestamps(
            audio=audio_float32,
            model=self.model,
            sampling_rate=SAMPLE_RATE,  # If not specified, defaults to 16000 for some models
            min_silence_duration_ms=self.min_silence_duration_ms,
            min_speech_duration_ms=500,
        )

        if not speech_timestamps:
            # No valid speech at all in this buffer
            del audio_float32
            del raw_audio_copy
            raise NoSpeechDetected()

        # By default, speech_timestamps is a list of dicts with "start" and "end" in samples
        audio_length = len(audio_float32)
        last_speech_end = speech_timestamps[-1]["end"]

        # If the waveform after last detected speech is greater than some threshold, treat it as silence.
        # By default, 700 ms @ 48kHz = 700 * 48 samples but you can adjust for your sample rate.
        # 700 ms at 48kHz = 0.7 * 48000 ~ 33600 samples
        threshold_samples = int(0.7 * SAMPLE_RATE)

        if audio_length - last_speech_end >= threshold_samples:
            # Enough trailing silence → stop
            del audio_float32
            del raw_audio_copy
            return False, speech_timestamps  # speech stopped probably
        del audio_float32
        del raw_audio_copy
        return True, speech_timestamps  # Still speech continuing

    def destroy(self):
        """Clean up resources."""
        # Nothing specific to clean up with the silero-vad library
        pass


class Hub:
    """Manages WebSocket clients and broadcasts messages."""

    def __init__(self):
        self.clients = set()

    async def register(self, client):
        """Register a new client."""
        self.clients.add(client)
        logger.info(f"Client registered. Total clients: {len(self.clients)}")

    async def unregister(self, client):
        """Unregister a client."""
        if client in self.clients:
            self.clients.remove(client)
            logger.info(f"Client unregistered. Total clients: {len(self.clients)}")


class Client:
    """Represents a WebSocket client connection."""

    def __init__(self, websocket, hub: Hub, vad: SileroVAD, language: str):
        self.websocket = websocket
        self.hub = hub
        self.vad = vad
        self.buffer = bytearray()
        self.language = language

    @staticmethod
    def int16_to_float32(sample):
        """Convert int16 audio sample to float32."""
        return float(sample) / 32768.0

    def transcribe(self):
        # Convert bytes to a file-like object
        num_channels = 1
        sample_width = 2
        sample_rate = 16000
        file_obj = io.BytesIO()
        with wave.open(file_obj, "wb") as wav_file:
            wav_file.setnchannels(num_channels)  # 1 = mono, 2 = stereo
            wav_file.setsampwidth(sample_width)  # 2 bytes for 16-bit PCM
            wav_file.setframerate(sample_rate)  # Sample rate (e.g., 16kHz)
            wav_file.writeframes(self.buffer)  # Write raw PCM data

        # Define the filename and content type
        files = {
            "file": (
                "input.wav",
                file_obj,
                "audio/wav",
            )  # Change MIME type accordingly
        }
        BASE_URL = f"http://{os.getenv('TRANSCRIBER_HOST')}:{os.getenv('TRANSCRIBER_PORT')}/transcribe/{self.language}"
        headers = {
            "Accept": "application/json",
        }
        response = requests.post(BASE_URL, headers=headers, files=files)
        return response.json()

    async def handle_messages(self):
        """Process incoming messages from the client."""
        try:
            await self.hub.register(self)

            async for message in self.websocket:
                if isinstance(message, bytes):
                    # Process audio data
                    self.buffer.extend(message)

                    # Process the buffer
                    await self.process_audio()
                else:
                    logger.warning(f"Received non-binary message: {message}")
        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection closed")
        finally:

            # Save buffer to a file
            # Convert buffer to numpy array
            audio_data = np.frombuffer(self.buffer, dtype=np.int16)

            # save to wave
            with wave.open("input.wav", "wb") as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(16000)
                f.writeframes(audio_data)
            await self.hub.unregister(self)

    async def process_audio(self):
        """Process audio data in the buffer and run VAD."""
        if len(self.buffer) < 2:  # Need at least one int16 sample
            return

        logger.info(f"Buffer length: {len(self.buffer)} bytes")

        # Run VAD detection
        try:
            speech_ongoing, segments = self.vad.memory_vad_check(self.buffer)
        except NoSpeechDetected:
            await self.websocket.send(
                json.dumps({"speech": "no_speech", "time_stamps": []})
            )
            return
        transcription = None
        # if speech_ongoing is False, send audio to transcribers
        if not speech_ongoing:
            transcription = self.transcribe()

        # Log segments
        # for segment in segments:
        #     logger.info(f"Speech starts at {segment.speech_start_at:.2f}s")
        #     if segment.speech_end_at is not None:
        #         logger.info(f"Speech ends at {segment.speech_end_at:.2f}s")

        # Send results back to client
        result = {
            "speech": speech_ongoing,
            "time_stamps": segments,
            "transcription": transcription,
        }

        await self.websocket.send(json.dumps(result))


from urllib.parse import parse_qs, urlparse


async def serve_websocket(websocket: websockets.ServerConnection):
    """Handle a WebSocket connection."""
    logger.info(f"New WebSocket connection from {websocket.remote_address}")
    query_params = parse_qs(urlparse(websocket.request.path).query)
    # Create VAD instance
    print("Params: ", query_params)
    language = query_params.get("language", ["uz"])[0]
    vad = SileroVAD(threshold=0.5, min_silence_duration_ms=100, speech_pad_ms=30)

    # Create client
    client = Client(websocket, hub, vad, language)
    # Handle client messages
    await client.handle_messages()


async def main():
    """Start the WebSocket server."""
    global hub
    hub = Hub()

    # Start WebSocket server
    async with websockets.serve(
        serve_websocket,
        "0.0.0.0",
        8555,
        ping_interval=PING_INTERVAL,
        ping_timeout=PING_TIMEOUT,
        max_size=MAX_MESSAGE_SIZE,
        origins=None,  # Allow connections from any origin
    ):
        logger.info("WebSocket server started on port 8555")
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    asyncio.run(main())
