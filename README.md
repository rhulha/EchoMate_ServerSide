# EchoMate - Voice Assistant Web Application

EchoMate is a web application that enables speech-to-speech conversations using Moonshine for speech recognition, Kokoro for text-to-speech synthesis, and Silero-VAD for voice activity detection.

# Two Versions of this app

There are two versions of this app.
One is pure client side but you need a PC with a very good graphics card and a browser that supports WebGPU.
The other does most of the work on the server side using Python so that you can use a very simple browser like on a mobile phone.

[pure client side version](https://github.com/rhulha/EchoMate)  
[server side version](https://github.com/rhulha/EchoMate_ServerSide)

This project is the server side version.

## Features

A fully-featured voice assistant web application that:
- Serves an interactive HTML interface using Flask
- Performs Voice Activity Detection (VAD) in the browser using [vad.js](https://github.com/ricky0123/vad)
- Sends detected speech to the server for transcription
- Uses Moonshine model for speech-to-text transcription
- Processes text with an LLM API for conversational responses
- Converts responses to speech using Kokoro text-to-speech
- Supports multiple voice options for personalized experiences

## Requirements

- Python 3.12+
- Flask
- PyTorch
- Transformers (for Moonshine STT model)
- soundfile
- eSpeak NG
- Access to an LLM API service (expected at http://localhost:8080/v1/chat/completions by default)

## Installation

1. Clone the repository:
```
git clone https://github.com/rhulha/EchoMate_ServerSide.git
cd EchoMate_ServerSide
```

2. Install the required Python packages:
```
pip install flask torch transformers soundfile numpy requests
```

3. For Windows users, install eSpeak NG from https://github.com/espeak-ng/espeak-ng/releases

4. Set up your LLM API service (compatible with OpenAI API format) or modify the LLM_API_URL in app.py

5. Run the application:
```
python app.py
```

6. Open your browser and navigate to:
```
http://localhost:8000
```

## How it works

1. The web interface uses vad.js to detect speech from the user's microphone
2. Voice Activity Detection (VAD) is performed in the browser
3. When speech is detected, the audio is sent to the server for transcription
4. The server processes the audio using the Moonshine speech-to-text model
5. The transcription is sent to an LLM API service for generating a response
6. The response is converted to speech using Kokoro text-to-speech with the selected voice
7. The audio response is streamed back to the client and played
8. The conversation history is maintained for context in future exchanges

