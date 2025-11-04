#include <Wire.h>
#include <ESP8266WiFi.h>
#include <FirebaseESP8266.h>

// WiFi Credentials
#define WIFI_SSID "******"
#define WIFI_PASSWORD "******"

// Firebase Credentials
#define FIREBASE_HOST "******"
#define FIREBASE_AUTH "******"
// Label for this session
#define ACTIVITY_LABEL "test1"  // <-- Change this before uploading for each activity

// MPU6886 address
const int MPU_ADDR = 0x68;

// Firebase objects
FirebaseData fbdo;
FirebaseAuth auth;
FirebaseConfig config;

// Sensor readings
int16_t ax, ay, az;
int16_t gx, gy, gz;

// Timing
unsigned long lastUploadTime = 0;
const unsigned long interval = 500; // 0.5 seconds

void setup() {
  Wire.begin(4, 5);  // SDA = GPIO4 (D2), SCL = GPIO5 (D1)
  Serial.begin(115200);

  // Initialize MPU6886
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B); // PWR_MGMT_1
  Wire.write(0);    // Wake up
  Wire.endTransmission(true);
  Serial.println("MPU6886 Initialized");

  // Connect to WiFi
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi Connected");

  // Firebase setup
  config.host = FIREBASE_HOST;
  config.signer.tokens.legacy_token = FIREBASE_AUTH;
  Firebase.begin(&config, &auth);
  Firebase.reconnectWiFi(true);
  Serial.println("Firebase Ready");
}

void loop() {
  unsigned long currentTime = millis();

  if (currentTime - lastUploadTime >= interval) {
    lastUploadTime = currentTime;

    // Read MPU6886 data
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x3B);  // Start reading from this register
    Wire.endTransmission(false);
    Wire.requestFrom((uint8_t)MPU_ADDR, (size_t)14, true);

    ax = Wire.read() << 8 | Wire.read();
    ay = Wire.read() << 8 | Wire.read();
    az = Wire.read() << 8 | Wire.read();
    Wire.read(); Wire.read(); // Skip temperature
    gx = Wire.read() << 8 | Wire.read();
    gy = Wire.read() << 8 | Wire.read();
    gz = Wire.read() << 8 | Wire.read();

    // Create CSV data string
    String dataString = String(ax) + "," + String(ay) + "," + String(az) + "," +
                        String(gx) + "," + String(gy) + "," + String(gz) + "," +
                        String(currentTime);

    // Firebase path: /IMU/sit/<timestamp>
    String path = "/IMUData/" + String(ACTIVITY_LABEL) + "/" + String(currentTime);

    // Upload to Firebase
    Serial.println("Uploading: " + dataString + " → " + path);
    if (Firebase.setString(fbdo, path, dataString)) {
      Serial.println("Upload Success");
    } else {
      Serial.print("Firebase Error: ");
      Serial.println(fbdo.errorReason());
    }
  }
}