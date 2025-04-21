# a project that enables you to manage your system, pathes and some simple task briefly and efficiently using some short commands
import json
import clipboard as clp
import pyperclip as pyclp
import sys
import os
import subprocess as sbp
from pathlib import Path
import keyboard as kd
import time
import requests
os.system('cls')

# Define paths for different categories
pathes = {
    "ai": r"C:\Users\AHMAD\OneDrive\Desktop\Perplexity",
    "chrome": r"C:\Users\AHMAD\OneDrive\Desktop\Google Chrome",
    "course": r"C:\Users\AHMAD\OneDrive\Desktop\ongoing courses",
    "general": r"C:\Users\AHMAD\OneDrive\Desktop\program(github)\comprehensive\general",
    "project": r"C:\Users\AHMAD\OneDrive\Desktop\program(github)\comprehensive\project",
    "whole": r"C:\Users\AHMAD\OneDrive\Desktop\program(github)\comprehensive",
    "mainfold": r"C:\Users\AHMAD\OneDrive\Desktop\program(github)",
    "lib": r"C:\Users\AHMAD\OneDrive\Desktop\program(github)\libraries",
    "dict": r"C:\Users\Public\Desktop\TDictionary.lnk",
    "music": r"C:\Users\AHMAD\OneDrive\Desktop\PERSONAL FOLDER\my music",
    "daily": r"C:\Users\AHMAD\OneDrive\Desktop\Daily task.txt",
    "relax": r"C:\Users\AHMAD\OneDrive\Desktop\relax.py",
    "access": r"C:\Users\AHMAD\OneDrive\Desktop\quick_access.py",
    'unifold': r"C:\Users\AHMAD\OneDrive\Desktop\university",
}

options = ["1-add a new task", "2-show the tasks", "3-open the file", "4-exit"]

file_path = Path(r"C:\Users\AHMAD\OneDrive\Desktop\MyTasks.txt")
uni_task = r"C:\Users\AHMAD\OneDrive\Desktop\university\university tasks.txt"

run = True

# ...........................................
# define below


# checking if a path exists.
def check_path_existence():
    is_running = True
    while is_running:
        path = input(
            "Enter a path here to check it (e to exit):\n>>> ").strip().lower()
        path = path.replace("'", "").replace('"', "")
        if os.path.exists(path):
            print(f"\nFound the path: {path}\n")
            start_input = input("start path (y/n)? ").lower().strip()
            if start_input == "y":
                os.startfile(path)
            else:
                pass
        elif path == "e":
            is_running = False
        else:
            print("\nCouldn't find the path provided!\n")


# showing the name of present course path
def show_path():
    with open("course_path.json", "r") as f:
        content = json.load(f)
        path = content.get("path")
        str_path = (path)
    new = str_path[48:-1]
    print(new)

# changing the present course path and starting a new one


def change_course_path():
    import json
    import sys
    file = "course_path.json"
    while True:
        new_path = input("What is the path to the new course:\n>>> ").strip()

        if new_path == "":
            print("Can't remain empty")
            continue
        # inside the parentheses, we want the condition to be true, but when it comes out of the parentheses, it transfers to false and "False" is what we need to check if extensions are not equal to True
        # take a look at its logic below
        elif not (new_path.endswith(".mp4") or new_path.endswith(".mkv")):
            print("Provide a path to a video file: (.mp4/.mkv)")
            continue
        # "new_path.endswith(".mp4")" or "new_path.endswith(".mkv"):"" This part checks if the path ends with either .mp4 or .mkv.
        # "not (...)": The not operator then negates the result of that entire expression.

        elif not os.path.exists(new_path):
            print("Provided path doesn't exist at all!")
            continue

        else:
            be_written = {"path": f"{new_path}"}
            with open(file, "w") as f:
                json.dump(be_written, f)
                print("Path to new course changed successfully.")
                start_input = input(
                    "Start newly added course (y/n)? ").lower().strip()
                if start_input == "y":
                    os.startfile(new_path)
                    break
                else:
                    sys.exit()

# starting and writing the present course


def start_present_course():
    import json
    import os
    file = 'course_path.json'
    from pathlib import Path
    if os.path.exists(file):
        try:
            with open(file, "r") as f:
                content = json.load(f)
                main_path = content.get("path")
                main_path = Path(main_path)
                os.startfile(main_path)
                sys.exit()
        except FileNotFoundError:
            print("The file is deleted or lost!")
            change_course_path()

    else:
        with open(file, "w") as f:
            while True:
                provided_path = input("Provide a path here:\n>>> ").strip()
                if provided_path == "":
                    print("Can't remain empty.")
                    continue
                elif not (provided_path.endswith(".mp4") or provided_path.endswith('.mkv')):
                    print("Provide a path to a video file: (.mp4/.mkv)")
                    continue
                elif not os.path.exists(provided_path):
                    print("Provided path doesn't exist at all!")
                    continue
                else:
                    be_written = {"path": f"{provided_path}"}
                    json.dump(be_written, f)
                    os.startfile(provided_path)
                    break

# getting the last index


def get_last_index(path):
    if path.exists():
        with open(path, "r") as f:
            lines = f.readlines()
            if lines:
                try:
                    return int(lines[-1].split("-")[0])
                except (ValueError, IndexError):
                    return 0
    return 0


# arranging numbers in a text file
def arrange_numbers():
    idx = 1
    new_lines = []

    with open(file_path, "r") as f:
        content = f.readlines()

        for line in content:
            main_lines = line.split("-")[:1]
            modified_line = line

            for chars in main_lines:
                if chars.isdigit():
                    modified_line = modified_line.replace(chars, str(idx), 1)
                    idx += 1

            new_lines.append(modified_line)

    with open(file_path, "w") as f:
        f.writelines(new_lines)


# task file related
def tasks():
    while run:
        print("Choose an option between: (1, 4):")
        for option in options:
            print(option)

        user_choice = input(">>>").strip()

        if user_choice == "1":
            arrange_numbers()
            last_index = get_last_index(file_path)
            while True:
                user_task = input(
                    "What should be written inside the file? (q to Quit)\n\n>>> ")
                if user_task.lower() == "q":
                    break
                else:
                    with open(file_path, "a") as f:
                        content = f"{last_index + 1}- {user_task}\n"
                        f.write(content)
                        last_index += 1
                        print(f"Task '{user_task}' added.")

        elif user_choice == "2":
            if file_path.exists():
                with open(file_path, "r") as f:
                    tasks = f.read()
                    print(
                        "________________________________________________________________")
                    print("Current Tasks:")
                    print(tasks)
                    print(
                        "________________________________________________________________")
            else:
                print("No tasks found.")

        elif user_choice == "3":
            if file_path.exists():
                os.system(f"start /max notepad {file_path}")
            else:
                print("File does not exist.")

        elif user_choice == "4":
            sys.exit()
        else:
            print("Invalid choice. Try again.")


# university text file related:
def university_tasks():
    uni_path = Path(
        r"C:\Users\AHMAD\OneDrive\Desktop\university\university tasks.txt")

    def add_task():
        arrange_numbers()
        last_index = get_last_index(uni_path)
        import os
        os.system('cls')
        with open(uni_path, 'a') as f:
            is_running = True
            while is_running:
                os.system('cls')
                add = input(f"What is your new assignment?\n>>> ")
                if add == "close":
                    is_running = False
                elif add == "close and open":
                    os.system(f"start /max notepad {uni_path}")
                    quit()
                elif add == "open":
                    os.system(f"start /max notepad {uni_path}")
                else:
                    content = f"{last_index + 1}- {add}\n"
                    last_index += 1
                    f.write(content)

    options = "1- add", "2- open", "3- show", "4- exit"
    for option in options:
        print(option)

    opt = input("What is your choice?\n>>> ")
    if opt == '1':
        add_task()
    elif opt == "2":
        os.system(f"start /max notepad {uni_path}")
    elif opt == "3":
        with open(uni_path, "r") as f:
            content = f.read()
            print(
                "Here are your tasks:\n__________________________________________________________")
            print(content)
            print("__________________________________________________________")
    elif opt == "4":
        quit()


def main_shortcuts():
    os.system('cls')

    def open_env(related_path):
        vsc_path = r"C:\Program Files\Microsoft VS Code\Code.exe"
        open_with_vsc = {
            "general": r"C:\Users\AHMAD\OneDrive\Desktop\program(github)\comprehensive\general",
            "project": r"C:\Users\AHMAD\OneDrive\Desktop\program(github)\comprehensive\project",
            "libraries": r"C:\Users\AHMAD\OneDrive\Desktop\program(github)\libraries",
        }
        if related_path in open_with_vsc:
            sbp.Popen([vsc_path, open_with_vsc[related_path]])
    print('(Ctrl + Alt + p = project)\n(Ctrl + Alt + g = general)\n(Ctrl + Alt + l = libraries)')
    kd.add_hotkey("Ctrl + Alt + p", lambda: open_env("project"))
    kd.add_hotkey("Ctrl + Alt + g", lambda: open_env('general'))
    kd.add_hotkey("Ctrl + Alt + l", lambda: open_env('libraries'))
    kd.wait("esc")
    os.system("cls")

# this starts the vpn first and after 2 minutes, starts youtube.


def start_youtube():
    def check_connection():
        try:
            requests.get("https://www.google.com", timeout=5)
            return True
        except requests.ConnectionError or requests.exceptions.ReadTimeout:
            return False
    if check_connection():
        os.startfile(r"C:\Users\AHMAD\OneDrive\Desktop\fg800p.exe")
        dealy = 120
        for x in range(dealy, 0, -1):
            second = x % 60
            minute = int(x / 60) % 60
            print(f"\rYouTube will be opened by: {
                  minute:02}:{second:02}", end='')
            time.sleep(1)
        os.startfile(r"C:\Users\AHMAD\OneDrive\Desktop\YouTube.lnk")
        os.system("cls")
    else:
        print("Connect your device to a network first!")

# passing you all the keys and values inside "pathes" by index:


def pass_pathes():
    idx = 1
    for key, value in pathes.items():
        content = f"{idx}-{key}: {value}\n"
        idx += 1
        print(content)


# define above
# ...........................................
# MAIN BODY of the code:
run = True
if len(sys.argv) == 2:
    command = sys.argv[1]

    if command in pathes:
        os.startfile(pathes[command])

    elif command == "task":
        tasks()

    elif command == "clear":
        content = "import os\nos.system('cls')"
        if content == pyclp.paste():
            print(">{ ALREADY IN CLIPBOARD }<")
            pass
        else:
            pyclp.copy(content)

    elif command == "in-cmd":
        path = input("Provide a path to a python script:\n>>> ")
        import subprocess as sbp
        sbp.run(f'start cmd /k "python {path}"', shell=True)

    elif command == "present":
        start_present_course()

    elif command == "change-present":
        change_course_path()

    elif command == "show-present":
        show_path()

    elif command == "check-path":
        check_path_existence()

    elif command == "close-all":
        close = 'taskkill/f /fi "STATUS eq RUNNING" '
        os.system(f"start cmd /k {close}")
    elif command == "uni":
        university_tasks()
    elif command == "paper":
        os.system(
            f"start /max notepad {r"C:\Users\AHMAD\OneDrive\Desktop\recent project"}")
    elif command == "shorts":
        main_shortcuts()
    elif command == "start_youtube":
        start_youtube()
    elif command == "pass":
        pyclp.copy("3x9aUPId")
    elif command == 'pathes':
        pass_pathes()
    else:
        print(f"No name as '{command}' found in pathes")
else:
    print("Enter exactly two commands.")
# ...........................................
# MAIN BODY of the code:
