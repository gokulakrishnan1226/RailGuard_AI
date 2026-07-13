#include <WiFi.h>
#include <HTTPClient.h>
#include <WebServer.h>
#include <ArduinoJson.h>
#include <esp_task_wdt.h>

// --- Configuration ---
const char* ssid = "Infinix";
const char* password = "gokul2006";

const char* backend_ip = "192.168.43.181";
const int backend_port = 8000;
const char* api_key = "RAILGUARD_ESP32_SECRET_KEY";

const String device_id = "ESP32_ROVER_01";

// --- Hardware Pins (Placeholders) ---
const int MOTOR_IN1 = 12;
const int MOTOR_IN2 = 14;
const int MOTOR_IN3 = 27;
const int MOTOR_IN4 = 26;
const int MOTOR_ENA = 25;
const int MOTOR_ENB = 33;
const int LASER_PIN = 4;
const int VIBRATION_PIN = 34; // Analog input
const int TRIG_PIN = 5;
const int ECHO_PIN = 18;
const int SERVO_PIN = 19;
// GPS pins can be configured on Serial1 or Serial2 (e.g., RX=16, TX=17)

// --- State Variables ---
String motor_status = "STOPPED";
bool laser_status = false;
bool gps_status = false; // Mocking true/false for now
bool sensor_status = true;
float vibration_level = 0.0;
unsigned long lastHeartbeatTime = 0;
const unsigned long heartbeatInterval = 5000; // 5 seconds

// --- Web Server ---
WebServer server(80);

// --- Watchdog ---
#define WDT_TIMEOUT 15 // 15 seconds WDT

// --- Function Prototypes ---
void connectWiFi();
void handleStatus();
void handleCommand();
void sendHeartbeat();
void fetchCommands();
void executeCommand(String cmd);
void initHardware();

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("Starting RailGuard ESP32 Firmware...");

  // Init Watchdog
  esp_task_wdt_init(WDT_TIMEOUT, true);
  esp_task_wdt_add(NULL);

  initHardware();
  connectWiFi();

  // Setup Web Server Routes
  server.on("/status", HTTP_GET, handleStatus);
  server.on("/command", HTTP_POST, handleCommand);
  server.begin();
  Serial.println("HTTP Server started on port 80");
}

void loop() {
  server.handleClient();
  
  // Reconnect WiFi if disconnected
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi disconnected. Reconnecting...");
    connectWiFi();
  }

  // Non-blocking Heartbeat & Polling
  if (millis() - lastHeartbeatTime >= heartbeatInterval) {
    lastHeartbeatTime = millis();
    sendHeartbeat();
    fetchCommands();
  }

  // Reset Watchdog Timer
  esp_task_wdt_reset();
}

void initHardware() {
  // Initialize placeholder pins
  pinMode(LASER_PIN, OUTPUT);
  digitalWrite(LASER_PIN, LOW);
  
  pinMode(MOTOR_IN1, OUTPUT);
  pinMode(MOTOR_IN2, OUTPUT);
  pinMode(MOTOR_IN3, OUTPUT);
  pinMode(MOTOR_IN4, OUTPUT);
  
  // More initializations can be added here
}

void connectWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi: ");
  Serial.println(ssid);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("WiFi connected!");
    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("WiFi connection failed.");
  }
}

// --- WebServer Handlers ---

void handleStatus() {
  StaticJsonDocument<256> doc;
  doc["device_id"] = device_id;
  doc["uptime"] = millis() / 1000;
  doc["free_heap"] = ESP.getFreeHeap();
  doc["rssi"] = WiFi.RSSI();
  doc["motor"] = motor_status;
  doc["laser"] = laser_status;
  doc["gps"] = gps_status;
  doc["vibration"] = vibration_level;
  
  String response;
  serializeJson(doc, response);
  server.send(200, "application/json", response);
}

void handleCommand() {
  if (server.hasArg("plain")) {
    String body = server.arg("plain");
    StaticJsonDocument<256> doc;
    DeserializationError error = deserializeJson(doc, body);
    
    if (error) {
      server.send(400, "application/json", "{\"status\":\"error\",\"message\":\"Invalid JSON\"}");
      return;
    }

    String cmd = doc["command"].as<String>();
    executeCommand(cmd);
    
    server.send(200, "application/json", "{\"status\":\"executed\",\"command\":\"" + cmd + "\"}");
  } else {
    server.send(400, "application/json", "{\"status\":\"error\",\"message\":\"Empty body\"}");
  }
}

void executeCommand(String cmd) {
  cmd.toUpperCase();
  Serial.print("Executing command: ");
  Serial.println(cmd);

  if (cmd == "START") {
    motor_status = "RUNNING";
    // digitalWrite(MOTOR_IN1, HIGH); etc.
  } 
  else if (cmd == "STOP") {
    motor_status = "STOPPED";
    // digitalWrite(MOTOR_IN1, LOW); etc.
  }
  else if (cmd == "LEFT") {
    motor_status = "TURNING_LEFT";
  }
  else if (cmd == "RIGHT") {
    motor_status = "TURNING_RIGHT";
  }
  else if (cmd == "LASER_ON") {
    laser_status = true;
    digitalWrite(LASER_PIN, HIGH);
  }
  else if (cmd == "LASER_OFF") {
    laser_status = false;
    digitalWrite(LASER_PIN, LOW);
  }
  else if (cmd == "RESTART") {
    Serial.println("Restarting ESP32...");
    delay(1000);
    ESP.restart();
  }
}

// --- Backend Communication ---

void sendHeartbeat() {
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    String url = String("http://") + backend_ip + ":" + String(backend_port) + "/api/esp32/heartbeat";
    http.begin(url);
    http.addHeader("Content-Type", "application/json");
    http.addHeader("x-api-key", api_key);

    StaticJsonDocument<256> doc;
    doc["device_id"] = device_id;
    doc["ip_address"] = WiFi.localIP().toString();
    doc["wifi_strength"] = WiFi.RSSI();
    doc["heap_memory"] = ESP.getFreeHeap();
    doc["uptime"] = millis() / 1000;
    doc["status"] = "ONLINE";
    doc["battery"] = 100; // Mock battery
    doc["laser"] = laser_status;
    doc["motor"] = motor_status;
    doc["gps"] = gps_status;
    doc["vibration"] = vibration_level;

    String requestBody;
    serializeJson(doc, requestBody);

    int httpResponseCode = http.POST(requestBody);
    if (httpResponseCode > 0) {
      Serial.print("Heartbeat sent, response code: ");
      Serial.println(httpResponseCode);
    } else {
      Serial.print("Error sending heartbeat: ");
      Serial.println(httpResponseCode);
    }
    http.end();
  }
}

void fetchCommands() {
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    String url = String("http://") + backend_ip + ":" + String(backend_port) + "/api/esp32/command?device_id=" + device_id;
    http.begin(url);
    http.addHeader("x-api-key", api_key);

    int httpResponseCode = http.GET();
    if (httpResponseCode == 200) {
      String response = http.getString();
      StaticJsonDocument<256> doc;
      DeserializationError error = deserializeJson(doc, response);
      
      if (!error && doc.containsKey("command")) {
        String cmd = doc["command"].as<String>();
        if (cmd != "" && cmd != "NONE") {
            executeCommand(cmd);
        }
      }
    }
    http.end();
  }
}
