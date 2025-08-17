import os
import numpy as np
import soundfile as sf
import datetime
from typing import Union

def save_audio_as_wav(
    audio_data: Union[bytes, np.ndarray], 
    filepath: str, 
    sample_rate: int = 16000,
    add_timestamp: bool = False
) -> str:
    """
    Save raw audio data as a WAV file with proper headers.
    
    Args:
        audio_data: Raw audio data, either as bytes or as a numpy array
        filepath: Path to save the WAV file. If add_timestamp is True,
                 a timestamp will be added to the filename
        sample_rate: Sample rate of the audio in Hz (default: 16000)
        add_timestamp: Whether to add a timestamp to the filename (default: False)
    
    Returns:
        The actual filepath where the file was saved
    """
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    if add_timestamp:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        basename, ext = os.path.splitext(filepath)
        filepath = f"{basename}_{timestamp}{ext}"
    
    # Convert bytes to numpy array if necessary
    if isinstance(audio_data, bytes):
        float_data = np.frombuffer(audio_data, dtype=np.float32)
    else:
        float_data = audio_data
    
    # Ensure audio is float32 type
    if float_data.dtype != np.float32:
        float_data = float_data.astype(np.float32)
    
    # Normalize audio if necessary
    if np.max(np.abs(float_data)) > 1.0:
        float_data = float_data / np.max(np.abs(float_data))
    
    sf.write(filepath, float_data, sample_rate, format='WAV')
    return filepath


