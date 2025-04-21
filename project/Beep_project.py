import time
import datetime
import winsound


def AbeepToPlay():
    frequency = 2000
    duration = 2000
    run = True
    while run:
        x = winsound.Beep(frequency, duration)
        frequency = frequency - 500
        if frequency == 0:
            run = False
            print("done")


target_time = datetime.datetime.now().replace(
    hour=20, minute=0, second=0, microsecond=0)
while True:
    current_time = datetime.datetime.now()
    if current_time >= target_time:
        AbeepToPlay()
