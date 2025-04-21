import matplotlib.pyplot as plt
import sys
from pathlib import Path
path = Path(r"C:\Users\AHMAD\OneDrive\Desktop\IELTS document\IELTS practice.txt")


def start_function():
    run = True
    while run:
        start = (input("What is start number?\n>>> "))
        if start == "":
            print("Start number can't be empty!")
            continue
        else:
            try:
                start = int(start)
                if start < 0:
                    print("Start number can't be negative!")
                    continue
                if start == 0:
                    print("Start number can't be 0 !")
                    continue

            except ValueError:
                print("Not a valid start number!")
                continue
            run = False
        run = False
    return start


def end_function():
    run = True
    while run:
        end = input("What is ending point?\n>>> ")
        if end == "":
            print("Ending point can't be set to empty!")
            continue

        elif end == "00":
            return float('inf')
        try:
            end = int(end)
            if end == 0:
                print("Ending point can't be set to zero!")
                continue

            elif end < 0:
                print("Ending point can't be set to negative!")
                continue
        except ValueError:
            print("Not a valid end!")
            continue
        except TypeError:
            print("Type error.")
            continue
        run = False
    return end


def provider(parameter, start_number):
    print("________________________________________")
    print("Provided answers:\n")
    for index, answers in enumerate(parameter, start=start_number - 1):
        global content
        content = f"({index + 1}) >>> ({answers})"
        print(content)
    print("________________________________________")


def save_file(parameter, start_number, path):
    with open(path, "a") as f:
        for index, answers in enumerate(parameter, start=start_number - 1):
            content = f"({index + 1}) >>> ({answers})"
            f.write(content + "\n")
        print(">> File created successfully <<")
    print("  ^^^^^^^^^^^^^^^^^^^^^^^^^^")


def plot(input1, input2, third_parameter):
    labels = ["wrong answers", "right answers", "unanswered"]
    colors = ["red", "blue", 'yellow']
    ex = .1, .1, .1
    plt.pie([input1, input2, third_parameter], labels=labels,
            autopct='%.2f %%', colors=colors, explode=ex)
    plt.title("Your approximate performance on this test")
    plt.show()
