#!/usr/bin/env python3
import asyncio
import websockets
import wave
import argparse
import time
import json
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("websocket-test-client")


async def send_audio_chunks(websocket, audio_file, chunk_size_ms=100):
    """Send audio chunks from a WAV file to the WebSocket server."""
    try:
        for i in range(2):
            with wave.open(audio_file, "rb") as wav_file:
                # Get WAV file properties
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                frame_rate = wav_file.getframerate()
                n_frames = wav_file.getnframes()

                logger.info(f"Audio file: {audio_file}")
                logger.info(
                    f"Channels: {channels}, Sample width: {sample_width}, Frame rate: {frame_rate}"
                )
                logger.info(
                    f"Total frames: {n_frames}, Duration: {n_frames/frame_rate:.2f} seconds"
                )

                # Calculate frames per chunk based on milliseconds
                frames_per_chunk = int(frame_rate * (chunk_size_ms / 1000))

                # Send audio format information
                # await websocket.send(
                #     json.dumps(
                #         {
                #             "type": "config",
                #             "sample_rate": frame_rate,
                #             "sample_width": sample_width,
                #             "channels": channels,
                #         }
                #     )
                # )

                # Read and send chunks
                frames_sent = 0
                while frames_sent < n_frames:
                    # Read a chunk of audio data
                    chunk_frames = min(frames_per_chunk, n_frames - frames_sent)
                    audio_data = wav_file.readframes(chunk_frames)
                    frames_sent += chunk_frames

                    # Send the chunk
                    await websocket.send(audio_data)
                    logger.info(
                        f"Sent chunk: {len(audio_data)} bytes, {frames_sent}/{n_frames} frames"
                    )

                    # Wait for the next chunk interval
                    await asyncio.sleep(chunk_size_ms / 1000)
                
                # Send end-of-stream marker
            
            await websocket.send(json.dumps({"type": "eos"}))
            logger.info("Finished sending audio data")
            await asyncio.sleep(10)
            print("Sending again")
           
    except Exception as e:
        logger.error(f"Error sending audio: {e}")


async def receive_results(websocket):
    """Receive and display results from the WebSocket server."""
    try:
        while True:
            response = await websocket.recv()
            try:
                # Try to parse as JSON
                result = json.loads(response)
                logger.info(f"Received result: {json.dumps(result, indent=2)}")
            except json.JSONDecodeError:
                # Handle binary or non-JSON responses
                logger.info(f"Received non-JSON response: {len(response)} bytes")
    except websockets.exceptions.ConnectionClosed:
        logger.info("Connection closed")
    except Exception as e:
        logger.error(f"Error receiving results: {e}")


async def main():
    parser = argparse.ArgumentParser(
        description="WebSocket Audio Streaming Test Client"
    )
    parser.add_argument("--host", default="localhost", help="WebSocket server host")
    parser.add_argument("--port", type=int, default=8070, help="WebSocket server port")
    parser.add_argument("--audio", required=True, help="Path to WAV audio file")
    parser.add_argument(
        "--chunk-ms", type=int, default=100, help="Chunk size in milliseconds"
    )
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists() or not audio_path.is_file():
        logger.error(f"Audio file not found: {audio_path}")
        return

    uri = f"ws://{args.host}:{args.port}/ws_vad"
    logger.info(f"Connecting to {uri}")

    try:
        async with websockets.connect(uri) as websocket:
            logger.info("Connected to WebSocket server")

            # Create tasks for sending and receiving
            send_task = asyncio.create_task(
                send_audio_chunks(websocket, str(audio_path), args.chunk_ms)
            )
            receive_task = asyncio.create_task(receive_results(websocket))

            # Wait for sending to complete
            await send_task

            try:
                # Keep connection open until keyboard interrupt
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, closing connection...")
            finally:
                # Clean up receive task
                receive_task.cancel()
                try:
                    await receive_task
                except asyncio.CancelledError:
                    pass

    except Exception as e:
        logger.error(f"Connection error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
