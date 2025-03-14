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
from aiohttp import web
import aiohttp

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




def timer_decorator(func):
    """Decorator to measure the execution time of a function."""
    async def wrapper(*args, **kwargs):
        start_time = asyncio.get_event_loop().time()
        result = await func(*args, **kwargs)
        end_time = asyncio.get_event_loop().time()
        execution_time = end_time - start_time
        logger.info(f"Function {func.__name__} executed in {execution_time:.4f} seconds")
        return result
    return wrapper


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
from pydantic import BaseModel
from google import genai

gemini_api_key = os.environ.get("GOOGLE_API_KEY", "AIzaSyAoBR32ZZsXqXLZLhbUxg3R0Eb3xm1Eqsw")
class LLMTranscription(BaseModel):
    transcription: str    
    def to_dict(self):
        return self.model_dump()

async def  transcribe_with_gemini(audio_file_path, language):
    """
    Transcribe audio using Google's Gemini model
    
    Parameters:
    audio_file_path (str): Path to the audio file
    
    Returns:
    Result: Object containing transcription and usage data
    """
    client = genai.Client(api_key=gemini_api_key)
    myfile = client.files.upload(file=audio_file_path)
    if language == "uz":
        prompt = f'Generate a transcript of the speech. Language is Uzbek. Output in latin characters.'
    elif language == "en":
        prompt = f'Generate a transcript of the speech. Language is English. Output in latin characters.'
    elif language == "ru":
        prompt = f'Generate a transcript of the speech. Language is Russian.'
    else:
        prompt = f'Generate a transcript of the speech. Language is {language}.'

    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=[prompt, myfile],
        config={
            'response_mime_type': 'application/json',
            'response_schema': LLMTranscription,
        },
    )
    print(response.parsed)
    try:
        return response.parsed
    except Exception as e:
        logger.error(f"Error transcribing with Gemini: {e}")
        return None
    



from uuid import uuid4
class Client:
    """Represents a WebSocket client connection."""

    def __init__(self, websocket, hub: Hub, vad: SileroVAD, language: str):
        self.websocket = websocket
        self.hub = hub
        self.vad = vad
        self.buffer = bytearray()
        self.buffer_size = 0
        self.last_buffer_size_used = 0
        self.language = language

    @staticmethod
    def int16_to_float32(sample):
        """Convert int16 audio sample to float32."""
        return float(sample) / 32768.0

    async def transcribe(self):
        # Convert bytes to a file-like object
        num_channels = 1
        sample_width = 2
        sample_rate = 16000
        file_obj = io.BytesIO()

        # Run potentially blocking I/O operations in a thread pool
        loop = asyncio.get_event_loop()

        # Create WAV file
        await loop.run_in_executor(None, lambda: self._create_wav_file(file_obj))

        # Reset file position
        file_obj.seek(0)
        # save file
        idx = uuid4()
        filename = f"{idx}.wav"
        with open(filename, "wb") as f:
            f.write(file_obj.getvalue())

        # BASE_URL = f"http://{os.getenv('TRANSCRIBER_HOST')}:{os.getenv('TRANSCRIBER_PORT')}/{self.language}"

        # # Make HTTP request in a non-blocking way
        # async with aiohttp.ClientSession() as session:
        #     async with session.post(BASE_URL, data=file_obj.getvalue()) as response:
        #         result = await response.json()
        #         return result
        result = await transcribe_with_gemini(filename, self.language)
        os.remove(filename)
        try:
            output_format = {
                "status": "completed",
                "full_text": result.transcription
            }
        except Exception as e:
            logger.error(f"Error parsing result: {e}")
            output_format = {
                "status": "failed",
                "full_text": None
            }
        return output_format
        

    def _create_wav_file(self, file_obj):
        with wave.open(file_obj, "wb") as wav_file:
            wav_file.setnchannels(1)  # 1 = mono, 2 = stereo
            wav_file.setsampwidth(2)  # 2 bytes for 16-bit PCM
            wav_file.setframerate(16000)  # Sample rate (e.g., 16kHz)
            wav_file.writeframes(self.buffer)  # Write raw PCM data

    async def handle_messages(self):
        """Process incoming messages from the client."""
        try:
            await self.hub.register(self)

            async for message in self.websocket:
                try:
                    if isinstance(message, bytes):
                        # Process audio data
                        self.buffer.extend(message)
                        self.buffer_size += len(message)
                        print("Buffer size: ", self.buffer_size)
                        print("Last buffer size used: ", self.last_buffer_size_used)
                        # Process the buffer
                        if self.last_buffer_size_used < self.buffer_size:
                            result = await self.process_audio()
                            if result == "close":
                                self.buffer = bytearray()
                                self.buffer_size = 0
                                self.last_buffer_size_used = 0
                    elif isinstance(message, str):
                        if message == "STOP":
                            transcription = await self.transcribe()
                            await self.websocket.send(json.dumps(
                                {
                                    "speech": False,
                                    "time_stamps": [],
                                    "transcription": transcription,
                                }
                            ))
                            self.buffer = bytearray()
                            self.buffer_size = 0
                            self.last_buffer_size_used = 0
                    else:
                        logger.warning(f"Received non-binary message: {message}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    # Continue processing other messages
        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection closed")
        except Exception as e:
            logger.error(f"Unexpected error in handle_messages: {e}")
        finally:
            try:
                # Save buffer to a file
                audio_data = np.frombuffer(self.buffer, dtype=np.int16)

                # save to wave
                with wave.open("input.wav", "wb") as f:
                    f.setnchannels(1)
                    f.setsampwidth(2)
                    f.setframerate(16000)
                    f.writeframes(audio_data)
            except Exception as e:
                logger.error(f"Error saving audio: {e}")

            await self.hub.unregister(self)

    async def process_audio(self):
        """Process audio data in the buffer and run VAD."""
        if len(self.buffer) < 2:  # Need at least one int16 sample
            return None
        logger.info(f"Buffer length: {len(self.buffer)} bytes")
        # Run VAD detection
        try:
            speech_ongoing, segments = self.vad.memory_vad_check(self.buffer)
        except NoSpeechDetected:
            await self.websocket.send(
                json.dumps(
                    {"speech": "no_speech", "time_stamps": [], "transcription": None}
                )
            )
            return None
        transcription = None
        # if speech_ongoing is False, send audio to transcribers
        self.last_buffer_size_used = self.buffer_size
        if not speech_ongoing:
            transcription = await self.transcribe()

        
        # Send results back to client
        result = {
            "speech": speech_ongoing,
            "time_stamps": segments,
            "transcription": transcription,
        }
        print("Sending result: ", result)
        
        await self.websocket.send(json.dumps(result))
        if transcription is not None:
            print("Closing connection because of speech stopped")
            return "close" # Add return statement to stop processing
        print("Connection closed")

from urllib.parse import parse_qs, urlparse

# Create a single VAD instance at startup instead of per-connection
global_vad = None


async def initialize_vad():
    global global_vad
    global_vad = SileroVAD(threshold=0.5, min_silence_duration_ms=100, speech_pad_ms=30)
    logger.info("Global VAD model initialized")


async def serve_websocket(websocket: websockets.ServerConnection):
    """Handle a WebSocket connection."""
    logger.info(f"New WebSocket connection from {websocket.remote_address}")
    query_params = parse_qs(urlparse(websocket.request.path).query)
    # Use the global VAD instance
    global global_vad
    print("Params: ", query_params)
    language = query_params.get("language", ["uz"])[0]
    # Create client with the global VAD instance
    client = Client(websocket, hub, global_vad, language)
    # Handle client messages
    await client.handle_messages()


async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}


async def http_handler(request):
    """Handle HTTP requests."""
    if request.path == "/health":
        return web.json_response(await health_check())
    return web.Response(status=404)


async def main():
    """Start the WebSocket server."""
    global hub
    hub = Hub()

    # Initialize the VAD model once at startup
    await initialize_vad()

    # Start WebSocket server
    async with websockets.serve(
        serve_websocket,
        "0.0.0.0",
        8444,
        ping_interval=PING_INTERVAL,
        ping_timeout=PING_TIMEOUT,
        max_size=MAX_MESSAGE_SIZE,
        max_queue=64,  # Limit the connection queue
        origins=None,  # Allow connections from any origin
    ):
        logger.info("WebSocket server started on port 8555")
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    asyncio.run(main())
