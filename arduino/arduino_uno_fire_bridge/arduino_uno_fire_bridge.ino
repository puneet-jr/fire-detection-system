#include <DHT.h>

#define DHTPIN 2
#define DHTTYPE DHT11
#define FLAME_PIN 7
#define BUZZ_PIN 8

#define TEMP_THRESHOLD 2.0
#define HUM_THRESHOLD -2.0

const int SAMPLE_SIZE = 20;
const unsigned long READ_INTERVAL_MS = 1000;
const int BUZZER_ACTIVE_STATE = HIGH;
const int BUZZER_IDLE_STATE = LOW;

DHT dht(DHTPIN, DHTTYPE);

float meanT = 0.0;
float stdDevT = 1.0;
float meanH = 0.0;
float stdDevH = 2.0;

bool hostBuzzerEnabled = false;
bool localAlarmActive = false;
int flameIdleState = HIGH;
unsigned long lastReadAt = 0;
String incomingCommand = "";

float tempHistory[SAMPLE_SIZE];
float humHistory[SAMPLE_SIZE];

void recalibrateBaseline();
void handleSerialCommands();
void processCommand(String command);
void emitSensorPacket(float currentT, float currentH, int flameState, float zT, float zH, bool localAlarm);
void updateBuzzer();
void beepSelfTest(int durationMs, int repeatCount);
void calibrateFlameIdleState();

void setup() {
  Serial.begin(9600);
  dht.begin();
  pinMode(BUZZ_PIN, OUTPUT);
  pinMode(FLAME_PIN, INPUT_PULLUP);
  digitalWrite(BUZZ_PIN, BUZZER_IDLE_STATE);

  beepSelfTest(120, 2);
  calibrateFlameIdleState();

  recalibrateBaseline();
}

void loop() {
  handleSerialCommands();

  unsigned long now = millis();
  if (now - lastReadAt < READ_INTERVAL_MS) {
    return;
  }
  lastReadAt = now;

  float currentT = dht.readTemperature();
  float currentH = dht.readHumidity();
  int rawFlameState = digitalRead(FLAME_PIN);
  bool flameDetected = rawFlameState != flameIdleState;

  if (isnan(currentT) || isnan(currentH)) {
    Serial.println("LOG sensor_read_error");
    return;
  }

  float zT = (currentT - meanT) / stdDevT;
  float zH = (currentH - meanH) / stdDevH;

  localAlarmActive = flameDetected || zT > TEMP_THRESHOLD || (zT > 2.0 && zH < HUM_THRESHOLD);
  updateBuzzer();
  emitSensorPacket(currentT, currentH, flameDetected ? 1 : 0, zT, zH, localAlarmActive);
}

void recalibrateBaseline() {
  Serial.println("LOG calibrating_baseline");

  meanT = 0.0;
  meanH = 0.0;
  stdDevT = 0.0;
  stdDevH = 0.0;

  int collected = 0;
  while (collected < SAMPLE_SIZE) {
    float t = dht.readTemperature();
    float h = dht.readHumidity();

    if (isnan(t) || isnan(h)) {
      delay(1000);
      continue;
    }

    tempHistory[collected] = t;
    humHistory[collected] = h;
    meanT += t;
    meanH += h;
    collected++;
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

  if (stdDevT < 1.0) {
    stdDevT = 1.0;
  }
  if (stdDevH < 2.0) {
    stdDevH = 2.0;
  }

  Serial.println("LOG calibration_complete");
}

void handleSerialCommands() {
  while (Serial.available() > 0) {
    char incoming = Serial.read();
    if (incoming == '\n' || incoming == '\r') {
      if (incomingCommand.length() > 0) {
        processCommand(incomingCommand);
        incomingCommand = "";
      }
    } else {
      incomingCommand += incoming;
    }
  }
}

void processCommand(String command) {
  command.trim();
  if (command.equalsIgnoreCase("BUZZER:1")) {
    hostBuzzerEnabled = true;
    updateBuzzer();
    Serial.println("ACK BUZZER:1");
    return;
  }

  if (command.equalsIgnoreCase("BUZZER:0")) {
    hostBuzzerEnabled = false;
    updateBuzzer();
    Serial.println("ACK BUZZER:0");
    return;
  }

  if (command.equalsIgnoreCase("CALIBRATE")) {
    hostBuzzerEnabled = false;
    localAlarmActive = false;
    updateBuzzer();
    recalibrateBaseline();
    return;
  }
}

void emitSensorPacket(float currentT, float currentH, int flameState, float zT, float zH, bool localAlarm) {
  Serial.print("TEMP=");
  Serial.print(currentT, 2);
  Serial.print(",HUM=");
  Serial.print(currentH, 2);
  Serial.print(",FLAME=");
  Serial.print(flameState);
  if(flameState==1){
    beepSelfTest(1000,1);
  }
  Serial.print(",ALARM=");
  Serial.print(localAlarm ? 1 : 0);
  Serial.print(",ZT=");
  Serial.print(zT, 3);
  Serial.print(",ZH=");
  Serial.println(zH, 3);
}

void updateBuzzer() {
  digitalWrite(BUZZ_PIN, (localAlarmActive || hostBuzzerEnabled) ? BUZZER_ACTIVE_STATE : BUZZER_IDLE_STATE);
}

void beepSelfTest(int durationMs, int repeatCount) {
  for (int i = 0; i < repeatCount; i++) {
    digitalWrite(BUZZ_PIN, BUZZER_ACTIVE_STATE);
    delay(durationMs);
    digitalWrite(BUZZ_PIN, BUZZER_IDLE_STATE);
    delay(durationMs);
  }
}

void calibrateFlameIdleState() {
  int highCount = 0;
  int lowCount = 0;

  for (int i = 0; i < 20; i++) {
    int currentState = digitalRead(FLAME_PIN);
    if (currentState == HIGH) {
      highCount++;
    } else {
      lowCount++;
    }
    delay(20);
  }

  flameIdleState = highCount >= lowCount ? HIGH : LOW;
  Serial.print("LOG flame_idle_state=");
  Serial.println(flameIdleState);
}
