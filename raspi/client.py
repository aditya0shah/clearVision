import os
import cv2
import requests
import time
import io
import logging
import asyncio
import threading
from typing import Optional, Tuple
import google.generativeai as genai
from modules.gemini_vision import GeminiVision
from modules.elevenlabs import ElevenLabs
from elevenlabs.play import play

# Set environment variable to run OpenCV in headless mode
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



class WebcamClient:
    def __init__(self, server_url: str = "http://localhost:5000", capture_interval: float = 1.0, gemini_api_key: str = None, elevenlabs_api_key: str = None):
        """
        Initialize the webcam client.
        
        Args:
            server_url: URL of the server to send images to
            capture_interval: Time interval between captures in seconds
            gemini_api_key: Google AI API key for Gemini vision
        """
        self.server_url = server_url
        self.capture_interval = capture_interval
        self.cap: Optional[cv2.VideoCapture] = None
        
        # Initialize Modules
        self.gemini_vision = GeminiVision(api_key=gemini_api_key)
        self.elevenlabs = ElevenLabs(api_key=elevenlabs_api_key)
        self.last_vlm_call = 0
        self.vlm_interval = 10.0  # Call Gemini every 5 seconds
        
    def start_capture(self) -> None:
        """Start capturing from webcam and sending to server."""
        try:
            if not self._initialize_webcam():
                return
            
            logger.info(f"Webcam client started. Sending images to {self.server_url}")
            logger.info(f"Capture interval: {self.capture_interval} seconds")
            logger.info("Press 'q' to quit")
            
            frame_count = 0
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Could not read frame from webcam")
                    break
                
                frame_count += 1
                logger.debug(f"Capturing frame {frame_count}")
                
                # Process and send frame
                processed_frame = self._process_frame(frame, frame_count)
                
                # Display processed frame
                cv2.imshow('Webcam Feed with Bounding Boxes', processed_frame)
                
                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                time.sleep(self.capture_interval)
                
        except KeyboardInterrupt:
            logger.info("Stopping webcam client...")
        except Exception as e:
            logger.error(f"Error during capture: {e}")
        finally:
            self.cleanup()
    
    def _initialize_webcam(self) -> bool:
        """Initialize webcam and return success status."""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            logger.error("Could not open webcam")
            return False
        return True
    
    def _process_frame(self, frame, frame_number: int):
        """Process frame and return processed frame with bounding boxes."""
        # Convert frame to JPEG format
        success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            logger.error("Could not encode frame")
            return frame
        
        image_bytes = buffer.tobytes()
        
        # Check if it's time for a Gemini call (every 5 seconds)
        current_time = time.time()
        if current_time - self.last_vlm_call >= self.vlm_interval:
            self.last_vlm_call = current_time
            # Start async Gemini call in background thread
            threading.Thread(
                target=self._call_vlm_async, 
                args=(image_bytes,), 
                daemon=True
            ).start()
        
        return self.send_image_to_server(image_bytes, frame_number, frame)
    
    def _call_vlm_async(self, image_bytes: bytes):
        """Call Gemini API asynchronously in a background thread."""
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Run the async Gemini call
            description = loop.run_until_complete(
                self.gemini_vision.describe_image_async(image_bytes)
            )
            
            if description:
                logger.info(f"Gemini description received: {description[:200]}...")
                
                # Generate and play audio
                audio = loop.run_until_complete(
                    self.elevenlabs.generate_speech(description)
                )
                if audio:
                    try:
                        play(audio)
                        logger.info("Audio played successfully")
                    except Exception as e:
                        logger.error(f"Error playing audio: {e}")
                else:
                    logger.warning("No audio generated")

            else:
                logger.warning("No description received from Gemini")
            
            
            

                
        except Exception as e:
            logger.error(f"Error in async Gemini call: {e}")
        finally:
            loop.close()
    
    def send_image_to_server(self, image_bytes: bytes, frame_number: int, original_frame) -> cv2.Mat:
        """Send image to the server and return processed image with bounding boxes."""
        try:
            files = {
                'image': ('webcam_frame.jpg', io.BytesIO(image_bytes), 'image/jpeg')
            }
            
            response = requests.post(
                f"{self.server_url}/upload",
                files=files,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    logger.debug(f"Frame {frame_number} sent successfully")
                    print(f"Result: {result} and frame number: {frame_number} sent successfully")
                    vision_data = result.get('data', {}).get('vision_processing')
                    if vision_data:
                        return self.draw_bounding_boxes(original_frame.copy(), vision_data)
                    return original_frame
                else:
                    logger.warning(f"Frame {frame_number} failed: {result.get('error', 'Unknown error')}")
                    return original_frame
            else:
                logger.warning(f"Frame {frame_number} failed with status {response.status_code}")
                return original_frame
                
        except requests.exceptions.ConnectionError:
            logger.error(f"Frame {frame_number} failed: Could not connect to server at {self.server_url}")
            return original_frame
        except requests.exceptions.Timeout:
            logger.error(f"Frame {frame_number} failed: Request timeout")
            return original_frame
        except Exception as e:
            logger.error(f"Frame {frame_number} failed: {e}")
            return original_frame
    
    def draw_bounding_boxes(self, image: cv2.Mat, vision_data: dict) -> cv2.Mat:
        """Draw bounding boxes on the image based on vision processing results."""
        try:
            if 'detections' not in vision_data:
                return image

            # Define colors for different object types
            colors = {
                'person': (0, 255, 0),      # Green
                'door': (255, 0, 0),        # Blue
                'chair': (0, 0, 255),       # Red
                'table': (255, 255, 0),     # Cyan
                'default': (255, 0, 255)    # Magenta
            }
            
            for obj in vision_data['detections']:
                box = obj['box']
                xmin, ymin = int(box['xmin']), int(box['ymin'])
                xmax, ymax = int(box['xmax']), int(box['ymax'])
                
                label = obj['label'].replace('.', '')  # Remove trailing period
                score = obj['score']
                color = colors.get(label.lower(), colors['default'])
                
                # Draw bounding box
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
                
                # Draw label with confidence score
                label_text = f"{label}: {score:.2f}"
                self._draw_label(image, label_text, (xmin, ymin), color)
            
            return image
            
        except Exception as e:
            logger.error(f"Error drawing bounding boxes: {e}")
            return image
    
    def _draw_label(self, image: cv2.Mat, text: str, position: Tuple[int, int], color: Tuple[int, int, int]) -> None:
        """Draw a label with background on the image."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        x, y = position
        
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Draw background rectangle
        cv2.rectangle(image, (x, y - text_height - baseline - 5), 
                     (x + text_width, y), color, -1)
        
        # Draw text
        cv2.putText(image, text, (x, y - baseline - 5), 
                   font, font_scale, (255, 255, 255), thickness)
    
    def test_server_connection(self) -> bool:
        """Test if the server is reachable."""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info("Server is reachable")
                return True
            else:
                logger.warning(f"Server returned status {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Cannot reach server: {e}")
            return False
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        logger.info("Webcam client stopped.")

def main() -> None:
    """Main function to run the webcam client."""
    # Configuration
    SERVER_URL = "http://ec2-44-242-141-44.us-west-2.compute.amazonaws.com:80"
    CAPTURE_INTERVAL = 0.1  # More reasonable capture interval
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')  # Get from environment variable
    ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')  # Get from environment variable
    
    # Create and start client
    client = WebcamClient(
        server_url=SERVER_URL, 
        capture_interval=CAPTURE_INTERVAL,
        gemini_api_key=GEMINI_API_KEY,
        elevenlabs_api_key=ELEVENLABS_API_KEY
    )
    
    # Test server connection first
    if not client.test_server_connection():
        logger.error("Please make sure the server is running and accessible.")
        return
    
    # Start capturing
    client.start_capture()

if __name__ == "__main__":
    main()