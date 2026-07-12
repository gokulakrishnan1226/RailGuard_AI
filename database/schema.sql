-- RailGuard AI Database Schema

CREATE DATABASE IF NOT EXISTS railguard_db;
USE railguard_db;

-- Administrative Users Table
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(100) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) DEFAULT 'operator',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Intrusions, Obstacles, and Damage Alerts
CREATE TABLE IF NOT EXISTS alerts (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    gps_latitude DECIMAL(10, 8) DEFAULT 0.0,
    gps_longitude DECIMAL(11, 8) DEFAULT 0.0,
    detection_type VARCHAR(100) NOT NULL, -- e.g., 'Human Intrusion', 'Animal (Elephant)', 'Obstacle (Stone)', 'Rail Crack'
    confidence FLOAT NOT NULL,
    image_path VARCHAR(500) DEFAULT NULL,
    officer_verified BOOLEAN DEFAULT FALSE, -- True if human, verified as authorized officer
    resolved BOOLEAN DEFAULT FALSE
);

-- Real-time Telemetry logs from track patrol vehicle / ESP32
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

-- Insert Default Admin User (hashed password is 'admin123' using bcrypt)
-- Using a standard bcrypt format. Our auth code will handle insert if missing.
INSERT INTO users (username, password_hash, role)
SELECT 'admin', '$2b$12$R.S2uU61eZgG6z9.qWc2UuU7p60i1nNlV0J0c2K/pM/P7rE.VlyjK', 'admin'
WHERE NOT EXISTS (SELECT 1 FROM users WHERE username = 'admin');
