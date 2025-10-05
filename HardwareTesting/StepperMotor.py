import RPi.GPIO as GPIO
import time

pins = [17, 18, 27, 22]
GPIO.setmode(GPIO.BCM)
for pin in pins:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, 0)

seq = [
    [1,0,0,0],
    [1,1,0,0],
    [0,1,0,0],
    [0,1,1,0],
    [0,0,1,0],
    [0,0,1,1],
    [0,0,0,1],
    [1,0,0,1]
]


STEPS_PER_REV = 2048  
DEG_PER_STEP = 360 / STEPS_PER_REV
current_angle = 0 

def rotate_to(target_angle, delay=0.002):
    global current_angle
    diff = target_angle - current_angle


    if diff > 180:
        diff -= 360
    elif diff < -180:
        diff += 360

    steps = int(abs(diff) / DEG_PER_STEP)
    direction = 1 if diff > 0 else -1

    print(f"Rotating from {current_angle:.1f}° to {target_angle:.1f}° "
          f"({steps} steps, {'CW' if direction==1 else 'CCW'})")

    step_count = len(seq)
    step_index = 0

    for _ in range(steps):
        step_index = (step_index + direction) % step_count
        for pin in range(4):
            GPIO.output(pins[pin], seq[step_index][pin])
        time.sleep(delay)

    current_angle = target_angle % 360  


try:
    while True:
        trash_type = input("Enter trash type (paper/plastic/metal/other or q to quit): ").strip().lower()
        if trash_type == "q":
            break

        target_positions = {
            "paper": 0,
            "plastic": 90,
            "metal": 180,
            "other": 270
        }

        if trash_type not in target_positions:
            print("Invalid type! Try again.")
            continue

        rotate_to(target_positions[trash_type])

finally:
    GPIO.cleanup()
    print("GPIO cleaned up.")