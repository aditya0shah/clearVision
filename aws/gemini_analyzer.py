from google import genai
from google.genai import types
from PIL import Image
import io
import logging
from typing import Optional, Dict, Any
import time

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    """
    You are an AI assistant helping a blind person navigate their environment. 
        Analyze this image and provide a brief, helpful description focusing on:
        
        1. Obstacles or hazards that could be dangerous
        2. Navigation cues (doors, paths, clear areas)
        3. People or important objects they should be aware of
        4. General layout of the space
        
        Keep responses concise (1-2 sentences) and prioritize safety and navigation.
        If there's nothing important to report, simply say "Nothing significant detected.

    """
)

class GeminiAnalyzer:
    """
    Handles image analysis through Google's Gemini API.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize Gemini analyzer.
        
        Args:
            api_key: Google AI API key
        """
        self.api_key = api_key
        # 1. Configure the API key globally
        genai.configure(api_key=api_key)

        try:
            # 2. Initialize the client
            self.client = genai.Client()
            logger.info("Gemini client initialized")
            
            # 3. Create a GenerationConfig with the system prompt
            self.config = types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="text/plain" # Optional: For guaranteed text output
            )
            
            # 4. We will pass the model name directly in the analyze_image method 
            #    instead of trying to initialize a model object here, which is 
            #    the standard pattern.

        except Exception as e:
            logger.error(f"Error initializing client: {e}")
            # Raising the error might be better for an initializer, but keeping the original log
            
        logger.info("Gemini analyzer initialized")
    
    def analyze_image(self, image: Image.Image) -> Optional[str]:
        """
        Analyze image using Gemini API.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            Analysis result as string, or None if failed
        """
        try:
            # We use a simple user prompt that tells the model to execute the system instruction
            user_prompt = "Analyze this image according to your instructions."
            
            # The contents list includes the image part and the text prompt
            contents = [image, user_prompt]
            
            # Generate content, passing the contents and the pre-configured config
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config=self.config  # Use the config containing the system instruction
            )
            
            if response.text:
                logger.debug(f"Gemini analysis: {response.text}")
                return response.text.strip()
            else:
                logger.warning("Gemini returned empty response")
                return None
                
        except Exception as e:
            logger.error(f"Gemini analysis failed: {e}")
            return None
    
    def analyze_from_bytes(self, image_bytes: bytes) -> Optional[str]:
        """
        Analyze image from bytes data.
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            Analysis result as string, or None if failed
        """
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            return self.analyze_image(image)
            
        except Exception as e:
            logger.error(f"Failed to process image bytes: {e}")
            return None
