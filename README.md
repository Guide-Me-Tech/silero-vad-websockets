# Silero VAD WebSocket Server

This is a Python implementation of a WebSocket server that uses Silero VAD (Voice Activity Detection) to detect speech in audio streams. The server receives audio data from clients via WebSocket connections, processes it using the Silero VAD model, and sends back the speech detection results.

## Features

- WebSocket server for real-time audio processing
- Voice Activity Detection using the official Silero VAD library
- Support for multiple concurrent clients
- Simple web interface for testing

## Requirements

- Python 3.8+
- WebSockets library
- NumPy
- PyTorch
- Silero VAD library

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/Guide-Me-Tech/silero-vad-websockets.git
   cd https://github.com/Guide-Me-Tech/silero-vad-websockets.git
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

   Note: The first time you run the server, it will automatically download the Silero VAD model.

## Usage

1. Start the WebSocket server:
   ```
   python server.py
   ```

2. Start the HTTP server to serve the web interface:
   ```
   python http_server.py
   ```

3. Open the web interface:
   - Navigate to `http://localhost:8080/home.html` in your browser
   - Click "Connect" to establish a WebSocket connection
   - Click "Start Recording" to begin sending audio data to the server
   - The VAD results will be displayed in real-time

## Running on a Remote Server

If you're running the server on a remote machine:

1. Make sure ports 8070 (WebSocket) and 8080 (HTTP) are open and accessible
2. When connecting from a browser, use the server's IP address or domain name
3. If using HTTPS on your website, you'll need to set up WSS (WebSocket Secure) as well

## API

The server expects audio data in the following format:
- 16-bit PCM
- 16000 Hz sample rate
- Mono channel

The server responds with JSON messages in the following format:
```json
{
  "speech": true,
  "time_stamps": [
    {
      "start": 0.15,
      "end": 1.23
    }
  ]
}
```

## Customization

You can customize the VAD parameters in the `SileroVAD` class:
- `threshold`: Speech detection threshold (default: 0.5)
- `min_silence_duration_ms`: Minimum silence duration in milliseconds (default: 100)
- `speech_pad_ms`: Padding around speech segments in milliseconds (default: 30)

## Troubleshooting

- **WebSocket Connection Issues**: Make sure the WebSocket server is running and the port is accessible. Check browser console for errors.
- **No Audio Detection**: Ensure your microphone is working and properly configured in your browser.
- **Model Loading Errors**: The first time you run the server, it needs internet access to download the model.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Silero VAD](https://github.com/snakers4/silero-vad) - Voice Activity Detection model
- [websockets](https://websockets.readthedocs.io/) - WebSocket implementation for Python 