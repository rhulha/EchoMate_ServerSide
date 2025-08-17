from flask import Flask, render_template, request, Response
import requests
import torch
import numpy as np
import json
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
DEFAULT_VOICE = 'af_bella'
# We'll load voices dynamically based on user selection
print(f'Default voice: {DEFAULT_VOICE}')

# Cache for loaded voices to avoid reloading them every time
voice_cache = {}

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
    try:
        if request.content_type and 'multipart/form-data' in request.content_type:
            voice_name = request.form.get('voice', DEFAULT_VOICE)
            system_prompt = request.form.get('system_prompt', "You are a helpful assistant.")
            conversation_history_json = request.form.get('conversation_history', '[]')
            try:
                conversation_history = json.loads(conversation_history_json)
            except:
                conversation_history = []
            
            audio_file = request.files.get('audio')
            if not audio_file:
                return Response("No audio file received", status=400)
            audio_bytes = audio_file.read()
        else:
            return Response("Invalid audio data format", status=400)

        if not audio_bytes:
            return Response("No audio data received", status=400)
        
        print(f"Using voice: {voice_name}, System prompt: {system_prompt}")
        print(f"Conversation history length: {len(conversation_history)}")
        float_data = np.frombuffer(audio_bytes, dtype=np.float32) # Convert bytes to float32 array
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return Response(f"Error processing request: {str(e)}", status=400)
    
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
        # Build messages array starting with system prompt
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history messages
        if conversation_history and len(conversation_history) > 0:
            messages.extend(conversation_history)
        
        # Add current user message
        messages.append({"role": "user", "content": transcription})
        
        payload = {
            "model": "local-model",
            "messages": messages,
            "temperature": 0.7
        }
        
        response = requests.post(LLM_API_URL, json=payload)
        
        if response.status_code == 200:
            response_json = response.json()
            response_text = response_json['choices'][0]['message']['content']

            # Get or load the voice
            if voice_name in voice_cache:
                voicepack = voice_cache[voice_name]
            else:
                try:
                    voicepack = torch.load(f'voices/{voice_name}.pt', weights_only=True).to(device)
                    voice_cache[voice_name] = voicepack
                    print(f'Loaded voice: {voice_name}')
                except Exception as e:
                    print(f"Error loading voice {voice_name}: {str(e)}")
                    # Fall back to default voice
                    voice_name = DEFAULT_VOICE
                    if DEFAULT_VOICE not in voice_cache:
                        voicepack = torch.load(f'voices/{DEFAULT_VOICE}.pt', weights_only=True).to(device)
                        voice_cache[DEFAULT_VOICE] = voicepack
                    else:
                        voicepack = voice_cache[DEFAULT_VOICE]
            
            import re
            # Remove Markdown formatting for better speech
            response_text = re.sub(r'\*\*(.*?)\*\*', r'\1', response_text)  # Bold
            response_text = re.sub(r'\*(.*?)\*', r'\1', response_text)      # Italic
            response_text = re.sub(r'`(.*?)`', r'\1', response_text)        # Code
            response_text = re.sub(r'~~(.*?)~~', r'\1', response_text)      # Strikethrough

            audio, _ = generate(tts_model, response_text, voicepack, lang=voice_name[0])

            if audio is None:
                print("Audio generation failed")
                return Response("Audio generation failed", status=500)
            
            buffer = BytesIO()
            sf.write(buffer, audio, 22050, format='WAV')
            buffer.seek(0)
            
            # Create response with audio data
            response = Response(buffer.read(), status=200, content_type='audio/wav')
            
            # Add the text response in the headers for the client to display
            # URL encode to handle special characters and newlines
            import urllib.parse
            response.headers['X-Response-Text'] = urllib.parse.quote(response_text)
            response.headers['X-User-Text'] = urllib.parse.quote(transcription)
            
            return response
        
        
        else:
            return f"Error: Received status code {response.status_code} from LLM API"
            
    except Exception as e:
        return f"Error: Could not connect to LLM API: {str(e)}"

    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
