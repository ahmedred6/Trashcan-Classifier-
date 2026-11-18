# Trashcan-Classifier-
This project is about building an AI classifier for a trashcan that filters trash based on type. The AI model uses a pretrained model that is exported to a raspberry pi connected to a camera. 






https://github.com/user-attachments/assets/3969534d-def0-4773-9b95-347f340f88cc




<img width="599" height="448" alt="trainig_status" src="https://github.com/user-attachments/assets/593c7729-be48-4ed2-bc97-59a6538c1219" />
<img width="3000" height="2250" alt="confusion_matrix_normalized" src="https://github.com/user-attachments/assets/35e0bf71-ad7e-45c8-842d-8d7bfefde01b" />


https://github.com/user-attachments/assets/ce684cee-b524-40be-a727-0d45c33e9763

## Hardware

Prototype components
- Raspberry Pi with a Pi Camera Rev 1.3
- A servo motor and a stepper motor for sorting

Wiring used in the test scripts (`HardwareTesting/`, BCM pin numbers)
- Servo signal (50 Hz PWM): GPIO 17
- Stepper IN1 to IN4: GPIO 17, 18, 27, 22

Stepper test script positions (2048 steps per revolution)
- Paper: 0 degrees
- Plastic: 90 degrees
- Metal: 180 degrees
- Other: 270 degrees

Running the motor tests (on the Raspberry Pi)

```
python3 HardwareTesting/StepperMotor.py
python3 HardwareTesting/ServoMotor.py
```

Type paper, plastic, metal or other to move to that position, or q to quit.

Photos and videos of the tests are in `HardwareTestingMedia/`. The chosen mechanical design is in `PossibleDesigns/Selected_Design/`.
