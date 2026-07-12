import json
import os
import datetime
# Safe import for mysql.connector with fallback
try:
    import mysql.connector
    from mysql.connector import Error
    MYSQL_AVAILABLE = True
except ImportError:
    mysql = None
    class Error(Exception):
        pass
    MYSQL_AVAILABLE = False

from utils.logger import setup_logger

logger = setup_logger("database")

# Load configuration
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
try:
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
except Exception as e:
    logger.error(f"Failed to load config.json: {e}")
    config = {
        "database": {
            "host": "localhost",
            "user": "root",
            "password": "",
            "database": "railguard_db",
            "port": 3306,
            "use_mock": True
        }
    }


class MockCursor:
    """Mock database cursor for offline testing."""
    def __init__(self, data):
        self.data = data
        self.index = 0

    def execute(self, query, params=None):
        pass

    def fetchall(self):
        return self.data

    def fetchone(self):
        if self.index < len(self.data):
            val = self.data[self.index]
            self.index += 1
            return val
        return None

    def close(self):
        pass

class MockConnection:
    """Mock database connection for offline testing."""
    def __init__(self):
        self.is_mock = True

    def cursor(self, buffered=False, dictionary=False):
        return MockCursor([])

    def commit(self):
        pass

    def rollback(self):
        pass

    def close(self):
        pass

    def is_connected(self):
        return True

class DatabaseManager:
    """Database interface with MySQL and automated in-memory mockup fallback."""
    def __init__(self):
        db_cfg = config["database"]
        self.host = db_cfg.get("host", "localhost")
        self.user = db_cfg.get("user", "root")
        self.password = db_cfg.get("password", "")
        self.database = db_cfg.get("database", "railguard_db")
        self.port = db_cfg.get("port", 3306)
        self.use_mock = db_cfg.get("use_mock", True)
        
        # Force mock database mode if MySQL connector driver is not installed
        if not MYSQL_AVAILABLE:
            self.use_mock = True
        
        # Memory lists to act as in-memory DB tables for mockup
        self.mock_users = [
            {
                "id": 1,
                "username": "admin",
                "password_hash": "$2b$12$R.S2uU61eZgG6z9.qWc2UuU7p60i1nNlV0J0c2K/pM/P7rE.VlyjK", # hashed 'admin123'
                "role": "admin",
                "created_at": datetime.datetime.now()
            }
        ]
        self.mock_alerts = []
        self.mock_telemetry = []
        self.connection = None
        
        # Attempt initialization
        self.connect()
        self.init_tables()

    def connect(self):
        """Attempts to connect to MySQL database, falls back to mock if configured or failed."""
        if self.use_mock:
            logger.info("Database configured to use MOCK database mode.")
            self.connection = MockConnection()
            return

        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                port=self.port,
                connect_timeout=3
            )
            # Create DB if not exists
            cursor = self.connection.cursor()
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.database}")
            cursor.close()
            
            # Reconnect to the database
            self.connection.database = self.database
            logger.info("Successfully connected to MySQL database server.")
        except Error as e:
            logger.warning(f"MySQL Connection failed: {e}. Falling back to in-memory Mock Database.")
            self.connection = MockConnection()
            self.use_mock = True

    def init_tables(self):
        """Initializes tables for live MySQL if connected, otherwise handled in memory."""
        if self.use_mock:
            return

        try:
            cursor = self.connection.cursor()
            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(100) NOT NULL UNIQUE,
                    password_hash VARCHAR(255) NOT NULL,
                    role VARCHAR(50) DEFAULT 'operator',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            # Alerts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    gps_latitude DECIMAL(10, 8) DEFAULT 0.0,
                    gps_longitude DECIMAL(11, 8) DEFAULT 0.0,
                    detection_type VARCHAR(100) NOT NULL,
                    confidence FLOAT NOT NULL,
                    image_path VARCHAR(500) DEFAULT NULL,
                    officer_verified BOOLEAN DEFAULT FALSE,
                    resolved BOOLEAN DEFAULT FALSE
                );
            """)
            # Telemetry table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS esp32_telemetry (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    vibration_level FLOAT NOT NULL,
                    gps_latitude DECIMAL(10, 8) DEFAULT 0.0,
                    gps_longitude DECIMAL(11, 8) DEFAULT 0.0,
                    motor_state VARCHAR(50) DEFAULT 'STOPPED',
                    buzzer_state BOOLEAN DEFAULT FALSE,
                    servo_angle INT DEFAULT 90
                );
            """)
            
            # Default Admin user insert
            cursor.execute("SELECT id FROM users WHERE username = 'admin'")
            if not cursor.fetchone():
                cursor.execute(
                    "INSERT INTO users (username, password_hash, role) VALUES (%s, %s, %s)",
                    ("admin", "$2b$12$R.S2uU61eZgG6z9.qWc2UuU7p60i1nNlV0J0c2K/pM/P7rE.VlyjK", "admin")
                )
            
            self.connection.commit()
            cursor.close()
            logger.info("Live Database tables verified/initialized successfully.")
        except Error as e:
            logger.error(f"Error initializing live database tables: {e}")

    def get_user_by_username(self, username):
        """Fetches user details by username."""
        if self.use_mock:
            for user in self.mock_users:
                if user["username"] == username:
                    return user
            return None

        try:
            self.connect()
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
            user = cursor.fetchone()
            cursor.close()
            return user
        except Error as e:
            logger.error(f"Error fetching user {username}: {e}")
            return None

    def insert_user(self, username, password_hash, role="operator"):
        """Inserts a new user account."""
        if self.use_mock:
            new_id = len(self.mock_users) + 1
            user = {
                "id": new_id,
                "username": username,
                "password_hash": password_hash,
                "role": role,
                "created_at": datetime.datetime.now()
            }
            self.mock_users.append(user)
            return True

        try:
            self.connect()
            cursor = self.connection.cursor()
            cursor.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (%s, %s, %s)",
                (username, password_hash, role)
            )
            self.connection.commit()
            cursor.close()
            return True
        except Error as e:
            logger.error(f"Error inserting user: {e}")
            return False

    def add_alert(self, detection_type, confidence, image_path=None, officer_verified=False, gps_lat=28.6139, gps_lon=77.2090):
        """Creates a security or damage alert record."""
        now = datetime.datetime.now()
        if self.use_mock:
            new_id = len(self.mock_alerts) + 1
            alert = {
                "id": new_id,
                "timestamp": now,
                "gps_latitude": float(gps_lat),
                "gps_longitude": float(gps_lon),
                "detection_type": detection_type,
                "confidence": float(confidence),
                "image_path": image_path,
                "officer_verified": bool(officer_verified),
                "resolved": False
            }
            self.mock_alerts.insert(0, alert)  # Newest first
            logger.info(f"[MOCK DB] Alert Added: {detection_type} ({confidence:.2f}%)")
            return new_id

        try:
            self.connect()
            cursor = self.connection.cursor()
            cursor.execute(
                """INSERT INTO alerts (gps_latitude, gps_longitude, detection_type, confidence, image_path, officer_verified)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (float(gps_lat), float(gps_lon), detection_type, float(confidence), image_path, bool(officer_verified))
            )
            self.connection.commit()
            new_id = cursor.lastrowid
            cursor.close()
            logger.info(f"[MYSQL DB] Alert Added: {detection_type} ({confidence:.2f}%)")
            return new_id
        except Error as e:
            logger.error(f"Error adding alert to DB: {e}")
            return None

    def get_alerts(self, limit=50):
        """Retrieves list of alerts."""
        if self.use_mock:
            # Map datetime objects to ISO strings
            result = []
            for a in self.mock_alerts[:limit]:
                alert_copy = a.copy()
                if isinstance(alert_copy["timestamp"], datetime.datetime):
                    alert_copy["timestamp"] = alert_copy["timestamp"].isoformat()
                result.append(alert_copy)
            return result

        try:
            self.connect()
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute("SELECT * FROM alerts ORDER BY timestamp DESC LIMIT %s", (limit,))
            rows = cursor.fetchall()
            cursor.close()
            # Normalize timestamp to ISO string
            for row in rows:
                if isinstance(row["timestamp"], datetime.datetime):
                    row["timestamp"] = row["timestamp"].isoformat()
            return rows
        except Error as e:
            logger.error(f"Error reading alerts from DB: {e}")
            return []

    def add_telemetry(self, vibration_level, gps_lat=28.6139, gps_lon=77.2090, motor_state="STOPPED", buzzer_state=False, servo_angle=90):
        """Logs track inspection vehicle telemetry data."""
        now = datetime.datetime.now()
        if self.use_mock:
            new_id = len(self.mock_telemetry) + 1
            tel = {
                "id": new_id,
                "timestamp": now,
                "vibration_level": float(vibration_level),
                "gps_latitude": float(gps_lat),
                "gps_longitude": float(gps_lon),
                "motor_state": motor_state,
                "buzzer_state": bool(buzzer_state),
                "servo_angle": int(servo_angle)
            }
            self.mock_telemetry.insert(0, tel)
            return new_id

        try:
            self.connect()
            cursor = self.connection.cursor()
            cursor.execute(
                """INSERT INTO esp32_telemetry (vibration_level, gps_latitude, gps_longitude, motor_state, buzzer_state, servo_angle)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (float(vibration_level), float(gps_lat), float(gps_lon), motor_state, bool(buzzer_state), int(servo_angle))
            )
            self.connection.commit()
            new_id = cursor.lastrowid
            cursor.close()
            return new_id
        except Error as e:
            logger.error(f"Error logging telemetry to DB: {e}")
            return None

    def get_latest_telemetry(self):
        """Gets latest telemetry data."""
        if self.use_mock:
            if self.mock_telemetry:
                tel = self.mock_telemetry[0].copy()
                if isinstance(tel["timestamp"], datetime.datetime):
                    tel["timestamp"] = tel["timestamp"].isoformat()
                return tel
            return {
                "vibration_level": 1.2,
                "gps_latitude": 28.6139,
                "gps_longitude": 77.2090,
                "motor_state": "STOPPED",
                "buzzer_state": False,
                "servo_angle": 90,
                "timestamp": datetime.datetime.now().isoformat()
            }

        try:
            self.connect()
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute("SELECT * FROM esp32_telemetry ORDER BY timestamp DESC LIMIT 1")
            row = cursor.fetchone()
            cursor.close()
            if row:
                if isinstance(row["timestamp"], datetime.datetime):
                    row["timestamp"] = row["timestamp"].isoformat()
                return row
            return None
        except Error as e:
            logger.error(f"Error reading telemetry from DB: {e}")
            return None

# Global Singleton Manager
db_manager = DatabaseManager()
