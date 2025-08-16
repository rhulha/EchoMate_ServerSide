# Voice Activity Detection Web Application

A Python web application that:
- Serves an HTML page using Flask
- Performs Voice Activity Detection (VAD) in the browser using vad.js
- Sends detected speech to the server for transcription
- Uses Moonshine model for speech-to-text transcription

## Requirements

- Python 3.7+
- Flask
- PyTorch
- Transformers (for Moonshine STT model)

## Installation

1. Install required packages:
```
pip install -r requirements.txt
```

2. Run the application:
```
python app.py
```

3. Open your browser and navigate to:
```
http://localhost:8000
```

## How it works

- The web interface uses vad.js to detect speech from the user's microphone
- Voice Activity Detection (VAD) is performed in the browser
- When speech is detected, the audio is sent to the server for transcription
- The server processes the audio using the Moonshine speech-to-text model
- Transcription results are returned to the client

## Features

- Client-side Voice Activity Detection
- Speech-to-text transcription using Moonshine model
- Simple and intuitive UI
- Minimal server overhead by performing VAD in the browser
