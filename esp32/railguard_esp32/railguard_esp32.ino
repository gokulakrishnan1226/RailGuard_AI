/*
  RailGuard AI - ESP32 WiFi Patroller Hardware Control Firmware
  
  Description:
    Connects to the local patrol WiFi network and hosts an HTTP REST server.
    Listens for JSON commands from the Python FastAPI server to actuate motors,
    LED indicators, alert buzzers, and camera pan servos.
    Periodically measures track vibration levels and returns GPS coordinate telemetry.
*/

#include <WiFi.h>
#include <WebServer.h>
#include <ArduinoJson.h> // Ensure "ArduinoJson" by Benoit Blanchon is installed in Arduino IDE
#include <ESP32Servo.h>  // Ensure "ESP32Servo" by John K. Bennett is installed in Arduino IDE

// WiFi Credentials
const char* ssid = "RailGuard_Patrol_Network";
const char* password = "security_override_2026";

// HTTP Port 80 Web Server
WebServer server(80);

// Hardware Actuator Pin Mappings
const int MOTOR_PWM_PIN = 12;  // Speed control
const int MOTOR_DIR_PIN = 14;  // Direction control
const int LED_RED_PIN = 25;    // Alert indicator
const int LED_GREEN_PIN = 26;  // Safe indicator
const int BUZZER_PIN = 27;     // Siren buzzer
const int SERVO_PIN = 13;      // Camera pan servo
const int VIB_SENSOR_PIN = 34; // Analog vibration sensor input

// Servo Object
Servo panServo;

// Telemetry State Variables
float currentVibration = 1.2;
double currentLatitude = 28.6139;
double currentLongitude = 77.2090;
String currentMotorState = "STOPPED";
bool currentBuzzerState = false;
int currentServoAngle = 90;

void setup() {
  Serial.begin(115200);
  
  // Configure Pins
  pinMode(MOTOR_PWM_PIN, OUTPUT);
  pinMode(MOTOR_DIR_PIN, OUTPUT);
  pinMode(LED_RED_PIN, OUTPUT);
  pinMode(LED_GREEN_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(VIB_SENSOR_PIN, INPUT);

  // Initialize Servo
  panServo.attach(SERVO_PIN);
  panServo.write(currentServoAngle); // Start centered (90 deg)
  
  // Set default LED states (Safe green on, alert off)
  digitalWrite(LED_GREEN_PIN, HIGH);
  digitalWrite(LED_RED_PIN, LOW);
  
  // Connect to WiFi Network
  Serial.print("Connecting to WiFi network: ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);
  
  // Wait for connection with visual indicator blinking
  int timeoutCounter = 0;
  while (WiFi.status() != WL_CONNECTED && timeoutCounter < 30) {
    delay(500);
    Serial.print(".");
    digitalWrite(LED_GREEN_PIN, !digitalRead(LED_GREEN_PIN));
    timeoutCounter++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    digitalWrite(LED_GREEN_PIN, HIGH);
    Serial.println("\nWiFi connected successfully!");
    Serial.print("IP Address allocated: ");
    Serial.println(WiFi.localIP());
  } else {
    // Fail-safe: Flash red LED continuously
    digitalWrite(LED_GREEN_PIN, LOW);
    digitalWrite(LED_RED_PIN, HIGH);
    Serial.println("\nWiFi Connection failed. Running in Offline telemetry mode.");
  }

  // Setup HTTP server endpoints
  server.on("/control", HTTP_POST, handleControlPost);
  server.on("/telemetry", HTTP_GET, handleTelemetryGet);
  server.begin();
  Serial.println("HTTP Web Server started on port 80.");
}

void loop() {
  server.handleClient();
  
  // Read analog vibration levels from sensor pin
  int rawVib = analogRead(VIB_SENSOR_PIN);
  currentVibration = (rawVib / 4095.0) * 10.0; // scale to 0-10 g metric
  
  // Simulate moving GPS Coordinates slightly for patroller movement
  if (currentMotorState == "FORWARD") {
    currentLatitude += 0.00001;
    currentLongitude += 0.000008;
  }
  
  delay(50); // Small delay to yield context
}

// POST endpoint handler for controlling actuators
void handleControlPost() {
  if (server.hasArg("plain") == false) {
    server.send(400, "text/plain", "Missing JSON request body.");
    return;
  }
  
  String body = server.arg("plain");
  StaticJsonDocument<256> doc;
  DeserializationError error = deserializeJson(doc, body);
  
  if (error) {
    server.send(400, "application/json", "{\"status\":\"error\",\"message\":\"JSON parsing failed.\"}");
    return;
  }
  
  // 1. Motor controls: "forward", "stop", "reverse"
  if (doc.containsKey("motor")) {
    String motorCmd = doc["motor"];
    if (motorCmd == "forward") {
      digitalWrite(MOTOR_DIR_PIN, HIGH);
      analogWrite(MOTOR_PWM_PIN, 200); // 78% speed
      currentMotorState = "FORWARD";
    } else if (motorCmd == "reverse") {
      digitalWrite(MOTOR_DIR_PIN, LOW);
      analogWrite(MOTOR_PWM_PIN, 200);
      currentMotorState = "REVERSE";
    } else { // stop
      analogWrite(MOTOR_PWM_PIN, 0);
      currentMotorState = "STOPPED";
    }
  }
  
  // 2. LED controls: "red", "green", "off"
  if (doc.containsKey("led")) {
    String ledCmd = doc["led"];
    if (ledCmd == "red") {
      digitalWrite(LED_RED_PIN, HIGH);
      digitalWrite(LED_GREEN_PIN, LOW);
    } else if (ledCmd == "green") {
      digitalWrite(LED_RED_PIN, LOW);
      digitalWrite(LED_GREEN_PIN, HIGH);
    } else {
      digitalWrite(LED_RED_PIN, LOW);
      digitalWrite(LED_GREEN_PIN, LOW);
    }
  }
  
  // 3. Buzzer / Siren controls (boolean)
  if (doc.containsKey("buzzer")) {
    currentBuzzerState = doc["buzzer"];
    if (currentBuzzerState) {
      // Sound a 1kHz square tone on pin
      tone(BUZZER_PIN, 1000); 
    } else {
      noTone(BUZZER_PIN);
    }
  }
  
  // 4. Servo panning controls (integer angle 0-180)
  if (doc.containsKey("servo")) {
    int targetAngle = doc["servo"];
    if (targetAngle >= 0 && targetAngle <= 180) {
      currentServoAngle = targetAngle;
      panServo.write(currentServoAngle);
    }
  }

  // Respond with success JSON telemetry status
  String responseBody;
  StaticJsonDocument<256> respDoc;
  respDoc["status"] = "success";
  respDoc["vibration"] = currentVibration;
  respDoc["gps_lat"] = currentLatitude;
  respDoc["gps_lon"] = currentLongitude;
  respDoc["motor_state"] = currentMotorState;
  respDoc["buzzer"] = currentBuzzerState;
  respDoc["servo_angle"] = currentServoAngle;
  
  serializeJson(respDoc, responseBody);
  server.send(200, "application/json", responseBody);
}

// GET endpoint handler returning current telemetries
void handleTelemetryGet() {
  String responseBody;
  StaticJsonDocument<256> respDoc;
  respDoc["vibration"] = currentVibration;
  respDoc["gps_lat"] = currentLatitude;
  respDoc["gps_lon"] = currentLongitude;
  respDoc["motor_state"] = currentMotorState;
  respDoc["buzzer"] = currentBuzzerState;
  respDoc["servo_angle"] = currentServoAngle;
  
  serializeJson(respDoc, responseBody);
  server.send(200, "application/json", responseBody);
}
