from flask import Flask, render_template, request, Response
import requests
import torch
import numpy as np
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from io import BytesIO
import soundfile as sf

import os
if os.name == "nt":  # Check if the OS is Windows
    os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
    os.environ["PHONEMIZER_ESPEAK_PATH"] = r"C:\Program Files\eSpeak NG\espeak-ng.exe"

from models import build_model
from kokoro import generate

SAMPLE_RATE = 16000
LLM_API_URL = "http://localhost:8080/v1/chat/completions"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Loading Kokoro model...")
tts_model = build_model('pth/kokoro-v0_19.pth', device)
VOICE_NAME = 'af_bella'
VOICEPACK = torch.load(f'voices/{VOICE_NAME}.pt', weights_only=True).to(device)
print(f'Loaded voice: {VOICE_NAME}')

app = Flask(__name__, static_url_path='/', static_folder='static')


print("Loading Moonshine model...")
processor = AutoProcessor.from_pretrained("UsefulSensors/moonshine-tiny")
stt_model = AutoModelForSpeechSeq2Seq.from_pretrained("UsefulSensors/moonshine-tiny").to(device)
print("Moonshine model loaded successfully")

@app.route('/')
def index():
    return app.send_static_file('index.html')

def is_llm_api_reachable():
    try:
        response = requests.get(LLM_API_URL.rsplit('/', 1)[0], timeout=1)
        return response.status_code < 500
    except Exception:
        return False
    
@app.route('/process_audio', methods=['POST'])
def process_audio():
    audio_bytes = request.data
    
    if not audio_bytes:
        return Response("No audio data received", status=400)
    
    float_data = np.frombuffer(audio_bytes, dtype=np.float32) # Convert bytes to float32 array
    
    if float_data.dtype != np.float32:
        float_data = float_data.astype(np.float32) # Ensure audio is float32
    
    if np.max(np.abs(float_data)) > 1.0:
        # Normalize if audio level is too high
        float_data = float_data / np.max(np.abs(float_data))
        
    if np.max(np.abs(float_data)) < 0.01:
        print("Audio too quiet, skipping transcription")
        return Response("Audio too quiet", status=400)
        
    # Convert to tensor for Moonshine STT
    # This step might not be needed.
    speech_tensor = torch.FloatTensor(float_data).to(device)

    inputs = processor(speech_tensor,sampling_rate=SAMPLE_RATE,return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = stt_model.generate(**inputs)

    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"Transcription: {transcription}")

    if not is_llm_api_reachable():
        print("LLM API not reachable, returning input text for testing")
        return Response(transcription, status=200, content_type='text/plain')
    
    try:
        payload = {
            "model": "local-model",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": transcription}
            ],
            "temperature": 0.7
        }
        
        response = requests.post(LLM_API_URL, json=payload)
        
        if response.status_code == 200:
            response_json = response.json()
            response_text = response_json['choices'][0]['message']['content']

            audio, _ = generate(tts_model, response_text, VOICEPACK, lang=VOICE_NAME[0])

            if audio is None:
                print("Audio generation failed")
                return Response("Audio generation failed", status=500)
            
            buffer = BytesIO()
            sf.write(buffer, audio, 22050, format='WAV')
            buffer.seek(0)
            return Response(buffer.read(), status=200, content_type='audio/wav')
        
        
        else:
            return f"Error: Received status code {response.status_code} from LLM API"
            
    except Exception as e:
        return f"Error: Could not connect to LLM API: {str(e)}"

    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
