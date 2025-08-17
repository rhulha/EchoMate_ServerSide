import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/ort.js"
import { MicVAD } from './vad.js';

let vad = null;
let isReady = false;
let audioChunks = [];

function logActivity(message) {
    const timeString = new Date().toLocaleTimeString();
    $("#transcript").append(`<br>${timeString}: ${message}`);
}

function updateStatus(status, message) {
    $("#statusIndicator")
        .removeClass("inactive active listening")
        .addClass(status)
        .text(message);
}

async function sendAudioToServer(audioData) {
    try {
        updateStatus("active", "Processing audio...");
        const selectedVoice = $("#voiceSelect").val();
        const systemPrompt = $("#systemPrompt").val();
        const audioBlob = new Blob([audioData], { type: 'audio/wav' });
        const formData = new FormData();
        formData.append('audio', audioBlob, 'recording.wav');
        formData.append('voice', selectedVoice);
        formData.append('system_prompt', systemPrompt);
        logActivity(`Using voice: ${selectedVoice}`);
        const response = await fetch('/process_audio', {
            method: 'POST',
            body: formData
        });
        
        if (response.ok) {
            const contentType = response.headers.get('Content-Type');
            
            if (contentType && contentType.includes('audio/wav')) {
                const audioBlob = await response.blob();
                const audioUrl = URL.createObjectURL(audioBlob);
                const audioElement = new Audio(audioUrl);
                audioElement.onloadedmetadata = () => {
                    logActivity(`Playing audio response (${Math.round(audioElement.duration)}s)`);
                };
                audioElement.play();
            } else {
                const transcription = await response.text();
                logActivity(`Response: ${transcription}`);
            }
        } else {
            const errorText = await response.text();
            logActivity(`Error: ${errorText}`);
        }
        
        updateStatus("listening", "Listening...");
    } catch (error) {
        console.error("Error sending audio:", error);
        logActivity(`Error: ${error.message}`);
        updateStatus("listening", "Listening...");
    }
}

async function initializeVAD() {
    try {
        vad = await MicVAD.new({
            onSpeechStart: () => {
                console.log("Speech start detected");
                updateStatus("active", "Speech detected!");
                logActivity("Speech started");
                audioChunks = []; // Clear previous audio chunks
            },
            onSpeechEnd: (audio) => {
                console.log("Speech end detected");
                updateStatus("listening", "Processing...");
                logActivity("Speech ended");
                console.log("Audio length:", audio.length);
                
                // Send audio to Flask backend for processing
                sendAudioToServer(audio.buffer);
            },
            onVADMisfire: () => {
                console.log("VAD misfire");
                logActivity("VAD misfire (false positive)");
            }
        });
        
        $("#startButton").prop("disabled", false);
        console.log("VAD initialized successfully");
    } catch (error) {
        console.error("Error initializing VAD:", error);
        logActivity("Error initializing VAD: " + error.message);
    }
}

function setupEventListeners() {
    $("#startButton").on("click", function() {
        if (!isReady && vad) {
            isReady = true;
            vad.start();
            $(this).text("Stop Listening");
            updateStatus("listening", "Listening...");
            logActivity("Started listening");
        } else if (isReady && vad) {
            isReady = false;
            vad.pause();
            $(this).text("Start Listening");
            updateStatus("inactive", "Microphone paused");
            logActivity("Stopped listening");
        }
    });
}

$(document).ready(function() {
    setupEventListeners();
    initializeVAD();
});
