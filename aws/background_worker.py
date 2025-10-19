import threading
import time
import logging
import queue
from typing import Optional, Callable
from PIL import Image
import io

from gemini_analyzer import GeminiAnalyzer
from tts_service import TTSService
from alert_manager import AlertManager

logger = logging.getLogger(__name__)

class BackgroundWorker:
    """
    Background worker for asynchronous image analysis and alert generation.
    """
    
    def __init__(self, 
                 gemini_api_key: str,
                 elevenlabs_api_key: str,
                 analysis_interval: float = 5.0,
                 enabled: bool = True):
        """
        Initialize background worker.
        
        Args:
            gemini_api_key: Google AI API key for Gemini
            elevenlabs_api_key: ElevenLabs API key for TTS
            analysis_interval: Time between analyses in seconds
            enabled: Whether background processing is enabled
        """
        self.analysis_interval = analysis_interval
        self.enabled = enabled
        
        # Initialize services
        self.gemini_analyzer = GeminiAnalyzer(gemini_api_key)
        self.tts_service = TTSService(elevenlabs_api_key)
        self.alert_manager = AlertManager()
        
        # Thread management
        self.worker_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Store most recent frame
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'alerts_generated': 0,
            'errors': 0,
            'start_time': None
        }
        
        logger.info(f"Background worker initialized (enabled: {enabled}, interval: {analysis_interval}s)")
    
    def start(self) -> None:
        """Start the background worker thread."""
        if not self.enabled:
            logger.info("Background worker is disabled")
            return
        
        if self.worker_thread and self.worker_thread.is_alive():
            logger.warning("Background worker already running")
            return
        
        self.stop_event.clear()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        self.stats['start_time'] = time.time()
        
        logger.info("Background worker started")
    
    def stop(self) -> None:
        """Stop the background worker thread."""
        if self.worker_thread and self.worker_thread.is_alive():
            logger.info("Stopping background worker...")
            self.stop_event.set()
            self.worker_thread.join(timeout=5.0)
            
            if self.worker_thread.is_alive():
                logger.warning("Background worker did not stop gracefully")
            
        logger.info("Background worker stopped")
    
    def update_latest_frame(self, image_bytes: bytes, frame_info: dict = None) -> None:
        """
        Update the most recent frame for background analysis.
        
        Args:
            image_bytes: Image data as bytes
            frame_info: Optional metadata about the frame
        """
        if not self.enabled:
            return
        
        with self.frame_lock:
            self.latest_frame = {
                'image_bytes': image_bytes,
                'frame_info': frame_info or {},
                'timestamp': time.time()
            }
    
    def _worker_loop(self) -> None:
        """Main worker loop that processes the most recent frame."""
        last_analysis_time = 0
        
        logger.info("Background worker loop started")
        
        while not self.stop_event.is_set():
            try:
                current_time = time.time()
                
                # Check if it's time for analysis
                if current_time - last_analysis_time >= self.analysis_interval:
                    # Get the most recent frame
                    with self.frame_lock:
                        frame_data = self.latest_frame
                    
                    if frame_data:
                        self._process_frame(frame_data)
                        last_analysis_time = current_time
                    else:
                        logger.debug("No frame available for analysis")
                else:
                    # Sleep for a short time to avoid busy waiting
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Error in background worker loop: {e}")
                self.stats['errors'] += 1
                time.sleep(1.0)  # Brief pause before retrying
        
        logger.info("Background worker loop ended")
    
    def _process_frame(self, frame_data: dict) -> None:
        """
        Process a single frame through Gemini analysis and TTS.
        
        Args:
            frame_data: Frame data dictionary containing image_bytes and metadata
        """
        try:
            image_bytes = frame_data['image_bytes']
            frame_info = frame_data.get('frame_info', {})
            
            logger.debug(f"Processing frame for analysis (size: {len(image_bytes)} bytes)")
            
            # Analyze with Gemini
            analysis_result = self.gemini_analyzer.analyze_from_bytes(image_bytes)
            
            if analysis_result:
                self.stats['frames_processed'] += 1
                
                # Check if we should generate an alert
                if self.alert_manager.should_alert(analysis_result):
                    # Generate TTS and play
                    success = self.tts_service.speak_text(analysis_result)
                    
                    if success:
                        self.alert_manager.record_alert(analysis_result)
                        self.stats['alerts_generated'] += 1
                        logger.info(f"Alert generated: {analysis_result[:100]}...")
                    else:
                        logger.warning("TTS failed for alert")
                else:
                    logger.debug("Alert filtered out by alert manager")
            else:
                logger.debug("Gemini analysis returned no result")
                
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            self.stats['errors'] += 1
    
    def get_stats(self) -> dict:
        """Get worker statistics."""
        runtime = time.time() - self.stats['start_time'] if self.stats['start_time'] else 0
        
        with self.frame_lock:
            has_latest_frame = self.latest_frame is not None
        
        return {
            **self.stats,
            'runtime_seconds': runtime,
            'has_latest_frame': has_latest_frame,
            'is_running': self.worker_thread.is_alive() if self.worker_thread else False,
            'enabled': self.enabled,
            'alert_manager_stats': self.alert_manager.get_stats()
        }
    
    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable background processing."""
        if self.enabled != enabled:
            self.enabled = enabled
            if enabled:
                self.start()
            else:
                self.stop()
            logger.info(f"Background worker {'enabled' if enabled else 'disabled'}")
    
    def set_analysis_interval(self, interval: float) -> None:
        """Set the analysis interval in seconds."""
        self.analysis_interval = max(1.0, interval)  # Minimum 1 second
        logger.info(f"Analysis interval set to {self.analysis_interval}s")
