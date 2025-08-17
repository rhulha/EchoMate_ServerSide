import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/ort.js"
import { MicVAD } from './vad.js';

let vad = null;
let isReady = false;
let conversationHistory = [];

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
        // Pause the VAD while processing to avoid picking up system audio
        if (vad && isReady) {
            vad.pause();
            isReady = false;
            $("#startButton").text("Start Listening");
            updateStatus("active", "Processing audio...");
        }
        
        const selectedVoice = $("#voiceSelect").val();
        const systemPrompt = $("#systemPrompt").val();
        const audioBlob = new Blob([audioData], { type: 'audio/wav' });
        const formData = new FormData();
        formData.append('audio', audioBlob, 'recording.wav');
        formData.append('voice', selectedVoice);
        formData.append('system_prompt', systemPrompt);
        
        // Add conversation history to the request
        formData.append('conversation_history', JSON.stringify(conversationHistory));
        
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
                
                // Get the text response from the headers
                const textResponse = response.headers.get('X-Response-Text');
                if (textResponse) {
                    const decodedResponse = decodeURIComponent(textResponse);
                    
                    // Update conversation history with the transcription and response
                    const userTranscription = response.headers.get('X-User-Text');
                    if (userTranscription) {
                        const decodedTranscription = decodeURIComponent(userTranscription);
                        conversationHistory.push({ role: "user", content: decodedTranscription });
                        logActivity(`You said: ${decodedTranscription}`);
                    }
                    
                    logActivity(`Response: ${decodedResponse}`);

                    conversationHistory.push({ role: "assistant", content: decodedResponse });
                    
                    // Display conversation history in the UI
                    updateConversationDisplay();
                }
                
                audioElement.onloadedmetadata = () => {
                    logActivity(`Playing audio response (${Math.round(audioElement.duration)}s)`);
                };
                
                audioElement.onended = () => {
                    resumeListening();
                };
                
                updateStatus("active", "Playing response...");
                audioElement.play();
            } else {
                logActivity(`Error: ${response.statusText}`);
                resumeListening();
            }
        } else {
            const errorText = await response.text();
            logActivity(`Error: ${errorText}`);
            resumeListening();
        }
    } catch (error) {
        console.error("Error sending audio:", error);
        logActivity(`Error: ${error.message}`);
        resumeListening();
    }
}

async function initializeVAD() {
    try {
        vad = await MicVAD.new({
            onSpeechStart: () => {
                console.log("Speech start detected");
                updateStatus("active", "Speech detected!");
                logActivity("Speech started");
            },
            onSpeechEnd: (audio) => {
                console.log("Speech end detected");
                updateStatus("listening", "Processing...");
                logActivity("Speech ended");
                console.log("Audio length:", audio.length);
                sendAudioToServer(audio.buffer);
            },
            onVADMisfire: () => {
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

function resumeListening() {
    if (vad && !isReady) {
        isReady = true;
        vad.start();
        $("#startButton").text("Stop Listening");
        updateStatus("listening", "Listening...");
        logActivity("Resumed listening");
    }
}

function updateConversationDisplay() {
    $("#conversationHistory").empty();
    
    conversationHistory.forEach((message, index) => {
        const messageClass = message.role === "user" ? "user-message" : "assistant-message";
        $("#conversationHistory").append(
            `<div class="message ${messageClass}">
                <div class="message-header">${message.role === "user" ? "You" : "Assistant"}</div>
                <div class="message-content">${message.content}</div>
            </div>`
        );
    });
    
    const conversationDiv = document.getElementById("conversationHistory");
    if (conversationDiv) {
        conversationDiv.scrollTop = conversationDiv.scrollHeight;
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
    
    $("#clearConversationButton").on("click", function() {
        conversationHistory = [];
        updateConversationDisplay();
        logActivity("Conversation history cleared");
    });
}

$(document).ready(function() {
    setupEventListeners();
    initializeVAD();
});
