#include <Servo.h>

#define SERVO_PIN 9
#define BUFFERSIZE 255

Servo servo;

void setup() {
  // put your setup code here, to run once:
  servo.attach(SERVO_PIN);
  Serial.begin(9600);
}

void loop() {
  // put your main code here, to run repeatedly:
  char buffer[BUFFERSIZE];

  if (Serial.available()) { //there is a byte here.
    int nbytes = Serial.readBytesUntil('\n', buffer, BUFFERSIZE - 1);
    buffer[nbytes] = 0; //null terminated string
    String command = String(buffer);
    int angle = command.toInt();
    servo.write(angle);
  }
}
