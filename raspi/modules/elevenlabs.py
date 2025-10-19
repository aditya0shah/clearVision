import asyncio
from elevenlabs.client import ElevenLabs as ElevenLabsClient
from elevenlabs.play import play
import logging

logger = logging.getLogger(__name__)

class ElevenLabs:
    def __init__(self, api_key: str):
        self.api_key = api_key
        if api_key:
            self.client = ElevenLabsClient(api_key=api_key)
        else:
            self.client = None
            logger.warning("No ElevenLabs API key provided. Audio generation will be disabled.")

    async def generate_speech(self, text: str):
        """Generate speech from text using ElevenLabs API."""
        if not self.client:
            logger.warning("ElevenLabs client not initialized. Cannot generate speech.")
            return None
            
        try:
            audio = self.client.text_to_speech.convert(
                text=text,
                voice_id="JBFqnCBsd6RMkjVDRZzb",
                model_id="eleven_flash_v2",
                output_format="mp3_44100_128",
            )
            return audio
        except Exception as e:
            logger.error(f"Error generating speech: {e}")
            return None