import requests
import logging
from typing import Optional
import time
import threading
from elevenlabs.client import ElevenLabs
from elevenlabs.play import play

logger = logging.getLogger(__name__)

class TTSService:
    """
    Handles text-to-speech conversion using ElevenLabs API.
    """
    
    def __init__(self, api_key: str, voice_id: str = "21m00Tcm4TlvDq8ikWAM"):
        """
        Initialize TTS service.
        
        Args:
            api_key: ElevenLabs API key
            voice_id: Voice ID to use for speech synthesis
        """
        self.api_key = api_key

        elevenlabs = ElevenLabs(api_key=api_key)
        self.voice_id = voice_id
        self.base_url = "https://api.elevenlabs.io/v1"
        self.is_speaking = False
        self.speech_lock = threading.Lock()
        
        logger.info("TTS service initialized")
    
    def speak_text(self, text: str) -> bool:
        """
        Convert text to speech and play it.
        
        Args:
            text: Text to convert to speech
            
        Returns:
            True if successful, False otherwise
        """
        if not text or len(text.strip()) == 0:
            return True
        audio = elevenlabs.text_to_speech.convert(
        text="The first move is what sets everything in motion.",
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
        )
        self._play_audio(audio)
        return audio

    
    def _play_audio(self, audio_data: bytes) -> None:
        """
        Play audio data using system audio player.
        
        Args:
            audio_data: MP3 audio data as bytes
        """
        try:
            import tempfile
            import subprocess
            import os
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            # Play using system audio player
            try:
                # Try different audio players
                players = ["mpv", "mplayer", "play", "aplay"]
                for player in players:
                    try:
                        subprocess.run([player, temp_file_path], 
                                     check=True, 
                                     capture_output=True, 
                                     timeout=30)
                        break
                    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                        continue
                else:
                    logger.warning("No suitable audio player found")
                    
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file_path)
                except OSError:
                    pass
                    
        except Exception as e:
            logger.error(f"Audio playback failed: {e}")
    
    def is_busy(self) -> bool:
        """Check if TTS service is currently speaking."""
        return self.is_speaking
