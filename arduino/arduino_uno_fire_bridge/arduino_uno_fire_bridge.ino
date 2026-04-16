#include <DHT.h>

#define DHTPIN 2
#define DHTTYPE DHT11
#define FLAME_PIN 7
#define BUZZ_PIN 8

// 🔥 HIGH SENSITIVITY SETTINGS
#define TEMP_THRESHOLD 1.5
#define HUM_THRESHOLD -1.0
#define ABS_TEMP_THRESHOLD 42.0

const int SAMPLE_SIZE = 20;
const unsigned long READ_INTERVAL_MS = 1000;

const int BUZZER_ACTIVE_STATE = HIGH;
const int BUZZER_IDLE_STATE = LOW;

DHT dht(DHTPIN, DHTTYPE);

float meanT = 0.0, stdDevT = 1.5;
float meanH = 0.0, stdDevH = 3.0;

bool hostBuzzerEnabled = false;
bool localAlarmActive = false;
bool prevAlarmState = false;

int flameIdleState = HIGH;
unsigned long lastReadAt = 0;
String incomingCommand = "";

float tempHistory[SAMPLE_SIZE];
float humHistory[SAMPLE_SIZE];

// ---------- SETUP ----------
void setup() {
  Serial.begin(9600);
  dht.begin();

  pinMode(BUZZ_PIN, OUTPUT);
  pinMode(FLAME_PIN, INPUT_PULLUP);

  digitalWrite(BUZZ_PIN, BUZZER_IDLE_STATE);

  beep(100, 2);

  delay(2000); // stabilize sensor
  calibrateFlameIdleState();
  recalibrateBaseline();
}

// ---------- LOOP ----------
void loop() {
  handleSerialCommands();

  unsigned long now = millis();
  if (now - lastReadAt < READ_INTERVAL_MS) return;
  lastReadAt = now;

  float t = dht.readTemperature();
  float h = dht.readHumidity();

  if (isnan(t) || isnan(h)) {
    Serial.println("LOG sensor_error");
    return;
  }

  // 🔥 FILTERED FLAME DETECTION
  int flameCount = 0;
  for (int i = 0; i < 5; i++) {
    if (digitalRead(FLAME_PIN) != flameIdleState) {
      flameCount++;
    }
    delay(5);
  }
  bool flameDetected = (flameCount >= 3);

  float zT = (t - meanT) / stdDevT;
  float zH = (h - meanH) / stdDevH;

  // 🔥 HIGH SENSITIVITY LOGIC (EARLY DETECTION)
  localAlarmActive = flameDetected 
                     || t > ABS_TEMP_THRESHOLD 
                     || (zT > TEMP_THRESHOLD) 
                     || (zH < HUM_THRESHOLD);

  // 🔔 Beep on NEW alarm trigger (not continuous spam)
  if (localAlarmActive && !prevAlarmState) {
    beep(300, 2);
  }
  prevAlarmState = localAlarmActive;

  updateBuzzer();
  emitSensorPacket(t, h, flameDetected ? 1 : 0, zT, zH, localAlarmActive);
}

// ---------- CALIBRATION ----------
void recalibrateBaseline() {
  Serial.println("LOG calibrating");

  meanT = 0; meanH = 0;
  stdDevT = 0; stdDevH = 0;

  int count = 0;
  while (count < SAMPLE_SIZE) {
    float t = dht.readTemperature();
    float h = dht.readHumidity();

    if (isnan(t) || isnan(h)) {
      delay(1000);
      continue;
    }

    tempHistory[count] = t;
    humHistory[count] = h;

    meanT += t;
    meanH += h;

    count++;
    delay(1000);
  }

  meanT /= SAMPLE_SIZE;
  meanH /= SAMPLE_SIZE;

  for (int i = 0; i < SAMPLE_SIZE; i++) {
    stdDevT += pow(tempHistory[i] - meanT, 2);
    stdDevH += pow(humHistory[i] - meanH, 2);
  }

  stdDevT = sqrt(stdDevT / SAMPLE_SIZE);
  stdDevH = sqrt(stdDevH / SAMPLE_SIZE);

  if (stdDevT < 1.5) stdDevT = 1.5;
  if (stdDevH < 3.0) stdDevH = 3.0;

  Serial.println("LOG calibration_done");
}

// ---------- SERIAL ----------
void handleSerialCommands() {
  while (Serial.available()) {
    char c = Serial.read();

    if (c == '\n' || c == '\r') {
      if (incomingCommand.length() > 0) {
        processCommand(incomingCommand);
        incomingCommand = "";
      }
    } else {
      incomingCommand += c;
    }
  }
}

void processCommand(String cmd) {
  cmd.trim();

  if (cmd == "BUZZER:1") {
    hostBuzzerEnabled = true;
    Serial.println("ACK BUZZER ON");
  }
  else if (cmd == "BUZZER:0") {
    hostBuzzerEnabled = false;
    Serial.println("ACK BUZZER OFF");
  }
  else if (cmd == "CALIBRATE") {
    hostBuzzerEnabled = false;
    localAlarmActive = false;
    updateBuzzer();
    recalibrateBaseline();
  }
}

// ---------- OUTPUT ----------
void emitSensorPacket(float t, float h, int flame, float zT, float zH, bool alarm) {
  Serial.print("TEMP="); Serial.print(t);
  Serial.print(",HUM="); Serial.print(h);
  Serial.print(",FLAME="); Serial.print(flame);
  Serial.print(",ALARM="); Serial.print(alarm ? 1 : 0);
  Serial.print(",ZT="); Serial.print(zT);
  Serial.print(",ZH="); Serial.println(zH);
}

// ---------- BUZZER ----------
void updateBuzzer() {
  digitalWrite(BUZZ_PIN,
    (localAlarmActive || hostBuzzerEnabled) ? BUZZER_ACTIVE_STATE : BUZZER_IDLE_STATE
  );
}

void beep(int duration, int repeat) {
  for (int i = 0; i < repeat; i++) {
    digitalWrite(BUZZ_PIN, HIGH);
    delay(duration);
    digitalWrite(BUZZ_PIN, LOW);
    delay(duration);
  }
}

// ---------- FLAME CALIBRATION ----------
void calibrateFlameIdleState() {
  int highCount = 0, lowCount = 0;

  for (int i = 0; i < 20; i++) {
    if (digitalRead(FLAME_PIN) == HIGH) highCount++;
    else lowCount++;
    delay(20);
  }

  flameIdleState = (highCount >= lowCount) ? HIGH : LOW;

  Serial.print("LOG flame_idle=");
  Serial.println(flameIdleState);
}