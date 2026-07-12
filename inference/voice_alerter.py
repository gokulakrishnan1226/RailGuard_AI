import threading
import queue
import time
from utils.logger import setup_logger

logger = setup_logger("voice_alerter")

class VoiceAlerter:
    """Threaded non-blocking Text-To-Speech engine with alert rate-limiting."""
    
    def __init__(self, rate=150, volume=1.0, enabled=True):
        self.enabled = enabled
        self.speech_queue = queue.Queue()
        self.last_alert_time = {}
        self.cooldown_seconds = 6.0 # Wait 6 seconds between identical alerts
        
        if self.enabled:
            # Start background speaker thread
            self.thread = threading.Thread(target=self._speak_loop, daemon=True)
            self.rate = rate
            self.volume = volume
            self.thread.start()
            logger.info("Threaded Voice Alerter initialized and running.")
        else:
            logger.info("Voice Alerter is disabled in settings.")

    def _speak_loop(self):
        """Worker loop reading text from queue and calling pyttsx3 speaker."""
        try:
            import pyttsx3
            # Initialize engine in the background thread (pyttsx3 prefers run in single thread)
            engine = pyttsx3.init()
            engine.setProperty('rate', self.rate)
            engine.setProperty('volume', self.volume)
        except Exception as e:
            logger.error(f"Failed to initialize pyttsx3 engine: {e}. Speech alerts disabled.")
            return

        while True:
            try:
                text = self.speech_queue.get(timeout=1.0)
                if text is None:
                    break
                
                logger.info(f"Speaking: '{text}'")
                engine.say(text)
                engine.runAndWait()
                # Yield context/sleep shortly
                time.sleep(0.1)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Speech loop exception: {e}")
                time.sleep(1)

    def alert(self, alert_type, text):
        """Pushes speech text to queue if alert cooldown is satisfied."""
        if not self.enabled:
            return
            
        now = time.time()
        last_time = self.last_alert_time.get(alert_type, 0.0)
        
        if now - last_time >= self.cooldown_seconds:
            self.last_alert_time[alert_type] = now
            self.speech_queue.put(text)
        else:
            logger.debug(f"Alert '{alert_type}' is on cooldown. Skipping voice.")

    def stop(self):
        """Stops the speech thread."""
        if self.enabled:
            self.speech_queue.put(None)
