import os
import json
from utils.logger import setup_logger

logger = setup_logger("config")

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json")

def load_config():
    """Loads settings configuration from global config.json file."""
    if not os.path.exists(CONFIG_PATH):
        logger.warning(f"Config file not found at {CONFIG_PATH}. Using default config.")
        return {
            "project_name": "RailGuard AI",
            "database": {"host": "localhost", "user": "root", "password": "", "database": "railguard_db", "port": 3306, "use_mock": True},
            "server": {"host": "127.0.0.1", "port": 8000, "secret_key": "MOCK_KEY", "algorithm": "HS256", "access_token_expire_minutes": 1440},
            "esp32": {"ip": "192.168.1.100", "port": 80, "enabled": false, "vibration_threshold": 7.5},
            "inference": {
                "confidence_thresholds": {"human": 0.5, "animal": 0.5, "obstacle": 0.4, "crack": 0.4},
                "webcam_index": 0, "voice_alerts": True, "voice_rate": 150, "voice_volume": 1.0
            }
        }
    
    try:
        with open(CONFIG_PATH, "r") as f:
            config_data = json.load(f)
            logger.info("Loaded global configurations successfully.")
            return config_data
    except Exception as e:
        logger.error(f"Error reading config.json: {e}")
        return {}

config = load_config()
