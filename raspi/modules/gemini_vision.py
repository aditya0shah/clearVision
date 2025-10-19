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
                    "Describe the image very briefly in a couple words.",
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
                "Describe this image in detail. What objects, people, activities, or scenes do you see?",
                image
            ])
            
            description = response.text if response.text else "No description available"
            logger.info(f"Gemini description: {description[:100]}...")
            return description
            
        except Exception as e:
            logger.error(f"Error getting Gemini description: {e}")
            return None