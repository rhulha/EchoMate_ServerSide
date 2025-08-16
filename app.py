from flask import Flask, render_template, request, Response
import torch
import numpy as np
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

app = Flask(__name__, static_url_path='/', static_folder='static')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Loading Moonshine model...")
processor = AutoProcessor.from_pretrained("UsefulSensors/moonshine-tiny")
stt_model = AutoModelForSpeechSeq2Seq.from_pretrained("UsefulSensors/moonshine-tiny").to(device)
print("Moonshine model loaded successfully")

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/process_audio', methods=['POST'])
def process_audio():
    audio_bytes = request.data
    
    if not audio_bytes:
        return Response("No audio data received", status=400)
    
    float_data = np.frombuffer(audio_bytes, dtype=np.float32) # Convert bytes to float32 array
    
    sample_rate = 16000  # Expected sample rate for model
    
    if float_data.dtype != np.float32:
        float_data = float_data.astype(np.float32) # Ensure audio is float32
    
    if np.max(np.abs(float_data)) > 1.0:
        # Normalize if audio level is too high
        float_data = float_data / np.max(np.abs(float_data))
        
    if np.max(np.abs(float_data)) < 0.01:
        print("Audio too quiet, skipping transcription")
        return Response("Audio too quiet", status=400)
        
    # Convert to tensor for Moonshine STT
    speech_tensor = torch.FloatTensor(float_data).to(device)
    
    inputs = processor(
        speech_tensor, 
        sampling_rate=sample_rate, 
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        outputs = stt_model.generate(
            **inputs,  # Unpack inputs as keyword arguments
            max_new_tokens=128,
        )
    
    transcription = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    print(f"Transcription: {transcription}")
    
    return Response(transcription, status=200, content_type='text/plain')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
