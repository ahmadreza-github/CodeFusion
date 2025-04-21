from functions import plot , start_function , end_function , provider , save_file 
from pathlib import Path
import sys
import os
from rich import print
path1 = Path(
    r"C:\Users\AHMAD\OneDrive\Desktop\IELTS document\general information.rtf")
path2 = Path(
    r"C:\Users\AHMAD\OneDrive\Desktop\IELTS document\writing tasks.rtf")
saving_path = Path(
    r"C:\Users\AHMAD\OneDrive\Desktop\IELTS document\IELTS practice.txt")
options = ["reading", "writing", "listening"]

list_of_empties = []
be_saved = []


def test_taker():

    run = True
    while run:
        for option in options:
            print(option)
        global section_input
        section_input = input("Which part are you working on? ")
        section_input = section_input.strip()
        if section_input not in options:
            print("\ncheck your answer again!\n")
            continue

        if section_input == "reading":
            start = start_function()
            end = end_function()
            amount = start
            if end <= start:
                print("[red]what you think of me is you![/red]\U0001f4a9")
                continue
            else:
                print(f"From ({start}) to ({end}), Let's go.")
                run = True
                while run and amount < end:
                    ans_input = input(
                        f'what is the answer of ({amount})?\n>>> ')
                    ans_input = ans_input.strip() 
                    # this is how you can prevent amount from increasing
                    if ans_input == "*":
                        print('skipped')

                    if ans_input == "result":
                        provider(be_saved, start)
                        continue
                    # when ans_input is equal to something special
                    if ans_input == "*":
                        list_of_empties.append("(skipped)")
                        be_saved.append("skipped")
                    # here is how you can set a step back with a symbol:
                    elif ans_input == "<":
                        if be_saved:
                            removed_value = be_saved.pop()
                            print(f"Removed value: {removed_value}")
                            amount -= 2
                        else:
                            print('There is no value to remove!')
                    else:
                        be_saved.append(ans_input)

                    if ans_input == "":
                        print(
                            "The answer can't be blank, answer it or skip it using '*' symbol")
                        continue
                    if ans_input == "done":
                        be_saved.pop()
                        provider(be_saved, start)
                        save_file(be_saved, start, saving_path)
                        sys.exit()

                    amount += 1
                if amount == end:
                    run = False
                    provider(be_saved, start)
                    save_file(be_saved, start, saving_path)

            wrong_answers = int(
                input("How many wrong answers have you passed? "))
            right_answers = int(
                input("How many right answers have you passed? "))
            plot(wrong_answers, right_answers, len(list_of_empties))

        elif section_input == "writing":
            print("[red]For the writing part, you'll be directed to another file\nin order to insert info or write a text.[red]")
            run = True
            while run:
                c = input("Press 'c' to continue or 'e' to exit: ")
                if c.lower().strip() == "e":
                    run = False
                elif c == "c":
                    choose_input = int(input(
                        "What are you going to do?\n1-add general information\n2-write a text \n>>>"))

                    if choose_input == 1:
                        print("Directing to the file...")
                        os.startfile(path1)
                        sys.exit()
                    elif choose_input == 2:
                        print("Directing to the file...")
                        os.startfile(path2)
                        sys.exit()
                    else:
                        print("Invalid choice. Please try again.")
                        continue
                else:
                    print("Check your answer again!")
                    continue

        elif section_input == "reading":
            test_taker()


test_taker()


# if end_function == "00":
#     nonestop ending
