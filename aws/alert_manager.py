import logging
from typing import List, Optional
import time
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class AlertManager:
    """
    Manages intelligent alert filtering and deduplication.
    """
    
    def __init__(self, cooldown_seconds: int = 30, max_alerts_per_minute: int = 6):
        """
        Initialize alert manager.
        
        Args:
            cooldown_seconds: Minimum time between similar alerts
            max_alerts_per_minute: Maximum number of alerts per minute
        """
        self.cooldown_seconds = cooldown_seconds
        self.max_alerts_per_minute = max_alerts_per_minute
        
        # Track recent alerts
        self.recent_alerts = []  # List of (timestamp, alert_text) tuples
        self.last_alert_time = None
        
        # Keywords that indicate important alerts
        self.important_keywords = [
            'danger', 'hazard', 'obstacle', 'person', 'people', 'door', 'stairs',
            'wall', 'blocked', 'clear', 'path', 'exit', 'entrance', 'vehicle'
        ]
        
        # Keywords that indicate low-priority alerts
        self.low_priority_keywords = [
            'nothing significant', 'empty', 'quiet', 'normal', 'typical'
        ]
        
        logger.info(f"Alert manager initialized with {cooldown_seconds}s cooldown")
    
    def should_alert(self, alert_text: str) -> bool:
        """
        Determine if an alert should be played based on content and timing.
        
        Args:
            alert_text: Text content of the alert
            
        Returns:
            True if alert should be played, False otherwise
        """
        if not alert_text or len(alert_text.strip()) == 0:
            return False
        
        # Clean and normalize text
        clean_text = alert_text.lower().strip()
        
        # Check for low-priority content
        if any(keyword in clean_text for keyword in self.low_priority_keywords):
            logger.debug("Skipping low-priority alert")
            return False
        
        # Check cooldown period
        if self.last_alert_time:
            time_since_last = time.time() - self.last_alert_time
            if time_since_last < self.cooldown_seconds:
                logger.debug(f"Skipping alert due to cooldown ({time_since_last:.1f}s < {self.cooldown_seconds}s)")
                return False
        
        # Check rate limiting
        if not self._check_rate_limit():
            logger.debug("Skipping alert due to rate limiting")
            return False
        
        # Check for duplicate content
        if self._is_duplicate_content(clean_text):
            logger.debug("Skipping duplicate alert content")
            return False
        
        # Check if content is important enough
        if not self._is_important_content(clean_text):
            logger.debug("Skipping non-important alert content")
            return False
        
        return True
    
    def record_alert(self, alert_text: str) -> None:
        """
        Record that an alert was played.
        
        Args:
            alert_text: Text content of the alert that was played
        """
        current_time = time.time()
        self.last_alert_time = current_time
        
        # Add to recent alerts list
        self.recent_alerts.append((current_time, alert_text.lower().strip()))
        
        # Clean up old alerts (older than 1 minute)
        cutoff_time = current_time - 60
        self.recent_alerts = [(t, text) for t, text in self.recent_alerts if t > cutoff_time]
        
        logger.info(f"Recorded alert: {alert_text[:50]}...")
    
    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        current_time = time.time()
        recent_count = sum(1 for t, _ in self.recent_alerts if t > current_time - 60)
        return recent_count < self.max_alerts_per_minute
    
    def _is_duplicate_content(self, clean_text: str) -> bool:
        """Check if this content is similar to recent alerts."""
        # Simple similarity check - look for exact matches in recent alerts
        for _, recent_text in self.recent_alerts[-5:]:  # Check last 5 alerts
            if self._text_similarity(clean_text, recent_text) > 0.8:
                return True
        return False
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity (0.0 to 1.0)."""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _is_important_content(self, clean_text: str) -> bool:
        """Check if content contains important keywords."""
        return any(keyword in clean_text for keyword in self.important_keywords)
    
    def get_stats(self) -> dict:
        """Get alert statistics."""
        current_time = time.time()
        recent_count = len([t for t, _ in self.recent_alerts if t > current_time - 60])
        
        return {
            'alerts_last_minute': recent_count,
            'max_alerts_per_minute': self.max_alerts_per_minute,
            'cooldown_seconds': self.cooldown_seconds,
            'last_alert_ago': time.time() - self.last_alert_time if self.last_alert_time else None
        }
