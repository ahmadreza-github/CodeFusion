import sys
from rich import print
import winsound
import datetime
frequency = 2000
duration = 2000
for_loop = True
run = True
to_show_task = []

while run:
    options = [
        "1-add a new task", "2-remove a task", "3-show tasks", "4-exit"
    ]
    for option in options:
        print(option)

    user_choice = int(input("which option?\n>>>"))

    if user_choice == 1:
        add_task = input("What task do you want to add?\n>>>")
        to_show_task.append(add_task)
        print(f"The task '{add_task}' is added to the task list!")
        print("Time to set the alarm:")

        hour = int(input("At what hour: "))
        minute = int(input("At what minute: "))

        target_time = datetime.datetime.now().replace(
            hour=hour, minute=minute, second=0, microsecond=0)
        print(f"The alarm will go off at ({hour}:{minute}) to do '{add_task}'")

        while datetime.datetime.now() < target_time:
            pass
        while for_loop:

            x = winsound.Beep(frequency, duration)
            frequency = frequency - 500
            if frequency == 0:
                print(f"[bold italic]hey, it is time to do: '{
                      add_task}'[/bold italic]")
                for_loop = False
