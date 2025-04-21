from rich import print
from rich import Console
start = int(input("the beginning is: "))
end_input = input("the end is: ")

if end_input == "*":
    end = None  # set end to None initially
else:
    end = int(end_input)

print(f"the beginning is: {start}")
test_number = start
results = []
options = [1, 2, 3, 4]
run = True

while run:
    answer = input(f"what is the answer of {test_number}?\n>>>")

    if answer.lower() == "result":
        for idx, ans in enumerate(results):
            if ans is not None:
                print(f"The answer of {start+idx}: {ans}")
            else:
                print(f"The answer of: {start+idx} was skipped.")
        run = False
    else:
        if answer.lower() == "empty":
            results.append(None)  # Placeholder for skipped answer
        else:
            try:
                answer = int(answer)
                if answer not in options:
                    print("Choose a correct option")
                    continue
                results.append(answer)
            except ValueError:
                print(
                    "Invalid input. Please enter a number between 1 and 4 or 'result' to display results.")

        test_number += 1

        if test_number == end:
            x = input("Do you want to see the results? (y/n): ")
            if x.lower() == "y":
                for idx, ans in enumerate(results):
                    if ans is not None:
                        print(f"The answer of {start+idx}: {ans}")
                    else:
                        print(f"The answer of {start+idx} was skipped.")
            run = False

    if end is None and answer == "*":
        new_end_input = input(
            "Enter a new end number or '*' to continue:\n>>>")
        if new_end_input == "*":
            continue
        else:
            end = int(new_end_input)
