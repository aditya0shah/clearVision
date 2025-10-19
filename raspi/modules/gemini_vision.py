import os
import logging
from typing import Optional
import google.generativeai as genai
import asyncio


logger = logging.getLogger(__name__)


class GeminiVision:
    """Class for handling Gemini API calls for image description."""
    
    def __init__(self, api_key: str = None):
        """
        Initialize GeminiVision with API key.
        
        Args:
            api_key: Google AI API key. If None, will try to get from environment variable GEMINI_API_KEY
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            logger.warning("No Gemini API key provided. Set GEMINI_API_KEY environment variable or pass api_key parameter.")
            self.model = None
        else:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash')
            logger.info("GeminiVision initialized successfully")
    
    async def describe_image_async(self, image_bytes: bytes) -> Optional[str]:
        """
        Asynchronously describe an image using Gemini API.
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            Description of the image or None if failed
        """
        if not self.model:
            logger.warning("Gemini model not initialized. Skipping image description.")
            return None
            
        try:
            # Convert bytes to PIL Image for Gemini
            from PIL import Image
            import io as io_module
            
            image = Image.open(io_module.BytesIO(image_bytes))
            
            # Generate description
            response = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.model.generate_content([
                    """You are a real-time assistant for a blind person using a head-mounted camera. Speak only when there is an immediate safety concern or directly relevant visual information based on the user’s current action or direction. Ignore all general surroundings and proximity (the user’s haptic device handles that).

                    Report only urgent or context-critical cues, including:



                    Environmental hazards: hot stovetops, fires, sharp tools, broken glass, spills, tripping obstacles (e.g., toys, cords, or uneven surfaces).

                    Traffic and movement dangers: red/green lights, fast-approaching vehicles or cyclists, drop-offs, moving machinery.

                    Warning signs or instructions relevant to movement: “Beware of Dog,” “Wet Floor,” “Pull door ahead,” “Pedestrian crossing ahead,” “Sidewalk closed.”

                    Ignore irrelevant signs or details (e.g., “Speed Limit,” “No Parking,” “Shop Sale”).

                    All alerts must be extremely brief (under two seconds spoken), clear, and actionable, using short phrases with spatial terms (e.g., “Stop. Red light.” / “Hot stove ahead.” / “Toy on floor left.”). Respond only when something urgent or relevant is visible.

                    If nothing important is detected, reply {"type": "N/A"}.


                    If an alert is needed, respond in JSON:

                    { "type": "alert", "msg": "your message here" }
                    """,
                    image
                ])
            )
            
            description = response.text if response.text else "No description available"
            logger.info(f"Gemini description: {description[:100]}...")
            print(f"Gemini description: {description}")
            return description
            
        except Exception as e:
            logger.error(f"Error getting Gemini description: {e}")
            return None
    
    def describe_image_sync(self, image_bytes: bytes) -> Optional[str]:
        """
        Synchronously describe an image using Gemini API.
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            Description of the image or None if failed
        """
        if not self.model:
            logger.warning("Gemini model not initialized. Skipping image description.")
            return None
            
        try:
            # Convert bytes to PIL Image for Gemini
            from PIL import Image
            import io as io_module
            
            image = Image.open(io_module.BytesIO(image_bytes))
            
            # Generate description
            response = self.model.generate_content([
                    """You are a real-time assistant for a blind person using a head-mounted camera. Speak only when there is an immediate safety concern or directly relevant visual information based on the user’s current action or direction. Ignore all general surroundings and proximity (the user’s haptic device handles that).

                    Report only urgent or context-critical cues, including:



                    Environmental hazards: hot stovetops, fires, sharp tools, broken glass, spills, tripping obstacles (e.g., toys, cords, or uneven surfaces).

                    Traffic and movement dangers: red/green lights, fast-approaching vehicles or cyclists, drop-offs, moving machinery.

                    Warning signs or instructions relevant to movement: “Beware of Dog,” “Wet Floor,” “Pull door ahead,” “Pedestrian crossing ahead,” “Sidewalk closed.”

                    Ignore irrelevant signs or details (e.g., “Speed Limit,” “No Parking,” “Shop Sale”).

                    All alerts must be extremely brief (under two seconds spoken), clear, and actionable, using short phrases with spatial terms (e.g., “Stop. Red light.” / “Hot stove ahead.” / “Toy on floor left.”). Respond only when something urgent or relevant is visible.

                    If nothing important is detected, reply {"type": "N/A"}.


                    If an alert is needed, respond in JSON:

                    { "type": "alert", "msg": "your message here" }
                    """,
                    image
            ])
            
            description = response.text if response.text else "No description available"
            logger.info(f"Gemini description: {description[:100]}...")
            return description
            
        except Exception as e:
            logger.error(f"Error getting Gemini description: {e}")
            return None