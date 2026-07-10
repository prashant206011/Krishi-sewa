// main.cpp
#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <DHT.h>
#include <ArduinoJson.h>

// -------------------- WIFI --------------------
const char* ssid = "WiFi-Name";
const char* password = "WiFi-Password";

// -------------------- SERVER --------------------
const char* serverUpdate = "http://<SERVER_IP>:8000/update";

// -------------------- pH SENSOR --------------------
#define PH_PIN 35
float voltage = 0;
float phValue = 0;
float neutralVoltage = 0.53;
float acidVoltage    = 1.10;

// -------------------- DHT --------------------
#define DHTPIN 18
#define DHTTYPE DHT11
DHT dht(DHTPIN, DHTTYPE);

// -------------------- MOISTURE --------------------
#define MOISTURE_A0 34
int moistureADC = 0;
int moisturePercent = 0;
int dryValue = 4095;
int wetValue = 1100;

// -------------------- RELAY --------------------
#define RELAY_PIN 23
bool pumpStatus = false;

// -------------------- SETUP --------------------
void setup() {
  Serial.begin(115200);

  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);

  dht.begin();

  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, HIGH); // OFF initially (for Active-Low Relay Modules)

  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nWiFi Connected");
  Serial.println(WiFi.localIP());
}

// -------------------- LOOP --------------------
void loop() {
  // ========== pH ==========
  long phSum = 0;
  for (int i = 0; i < 10; i++) {
    phSum += analogRead(PH_PIN);
    delay(10);
  }
  int phADC = phSum / 10;
  voltage = phADC * (3.3 / 4095.0);
  float slope = (7.0 - 4.0) / (neutralVoltage - acidVoltage);
  phValue = 7.0 + slope * (voltage - neutralVoltage);

  // ========== MOISTURE ==========
  long sum = 0;
  for (int i = 0; i < 20; i++) {
    sum += analogRead(MOISTURE_A0);
    delay(5);
  }
  moistureADC = sum / 20;
  moisturePercent = map(moistureADC, dryValue, wetValue, 0, 100);
  moisturePercent = constrain(moisturePercent, 0, 100);

  // ========== DHT ==========
  float temperature = dht.readTemperature();
  float humidity = dht.readHumidity();

  // Guard against sensor reading errors converting NaN values to base defaults
  if (isnan(temperature)) temperature = 25.0;
  if (isnan(humidity))    humidity = 50.0;

  // ========== DEBUG PRINT ==========
  Serial.println("\n----- SENSOR DATA -----");
  Serial.println("Temp: " + String(temperature) + " °C");
  Serial.println("Humidity: " + String(humidity) + " %");
  Serial.println("Moisture: " + String(moisturePercent) + " %");
  Serial.println("pH: " + String(phValue));

  // ========== PUMP HYSTERESIS LOGIC ==========
  if (!pumpStatus && moisturePercent < 20) {
    pumpStatus = true;
    Serial.println("Action: Soil dry. Triggering Pump ON.");
  }
  if (pumpStatus && moisturePercent >= 30) {
    pumpStatus = false;
    Serial.println("Action: Target moisture reached. Shutting Pump OFF.");
  }

  // Low triggers relay ON, High switches relay OFF (Standard Active-Low Relay Boards)
  digitalWrite(RELAY_PIN, pumpStatus ? LOW : HIGH);

  // ========== SEND DATA TO SERVER ==========
  if (WiFi.status() == WL_CONNECTED) {
    WiFiClient client;  // Fixed: explicitly routes connections through network lanes
    HTTPClient http;
    
    http.begin(client, serverUpdate); 
    http.addHeader("Content-Type", "application/json");

    StaticJsonDocument<256> doc;
    doc["temperature"] = temperature;
    doc["humidity"] = humidity;
    doc["ph"] = phValue;
    doc["moisture"] = moisturePercent;
    doc["pump"] = pumpStatus;

    String json;
    serializeJson(doc, json);

    int code = http.POST(json);
    Serial.println("HTTP POST Response Code: " + String(code));

    http.end();
  } else {
    Serial.println("WiFi Disconnected. Reconnecting...");
    WiFi.begin(ssid, password);
  }

  delay(10000); // Wait 10 seconds per loop
}
