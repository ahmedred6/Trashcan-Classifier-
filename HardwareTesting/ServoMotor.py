import RPi.GPIO as GPIO
import time


SERVO_PIN = 17 
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)

pwm = GPIO.PWM(SERVO_PIN, 50)
pwm.start(0)

def set_angle(angle):
    
    duty = 2 + (angle / 18)
    GPIO.output(SERVO_PIN, True)
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.5)
    GPIO.output(SERVO_PIN, False)
    pwm.ChangeDutyCycle(0)

try:
    current_angle = 0
    set_angle(current_angle)

    while True:
        trash_type = input("Enter trash type (paper/plastic/metal/other or q to quit): ").strip().lower()
        if trash_type == "q":
            break
        target_positions = {
            "paper": 0,
            "plastic": 90,
            "other": 180,
            "metal": 180
        }
        if trash_type in target_positions:
            target = target_positions.get(trash_type)
        else:
            print("Invalid type! Try again.")
            continue

        print(f"Moving servo from {current_angle}° to {target}° for '{trash_type}'")
        set_angle(target)
        current_angle = target

finally:
    pwm.stop()
    GPIO.cleanup()
    print("GPIO cleaned up.")