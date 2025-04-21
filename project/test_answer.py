from rich import print

start = int(input("The beginning is: "))
end_input = input("The end is:")

if end_input == "*":
    end = None
else:
    end = int(end_input)

print(f"The beginning is: {start}")
test_number = start
results = []
options = [1, 2, 3, 4]

while True:
    answer = input(f"What is the answer of {test_number}?\n>>>").lower().strip()

    if not answer:
        print("Invalid input. Please enter a number between 1 and 4 or 'result' to display results or 'empty' to skip.")
        continue

    if answer == "result":
        for idx, ans in enumerate(results):
            if ans is not None:
                print(f"The answer of {start+idx}: {ans}")
            else:
                print(f"The answer of {start+idx} was skipped.")
        break
    elif answer == "empty":
        results.append(None)
    elif answer == "*":
        if end is None:
            end_input = input(
                "Enter a new end number or '*' to continue:\n>>>")
            if end_input == "*":
                continue
            else:
                end = int(end_input)
        else:
            print("Invalid input. Please enter a number between 1 and 4 or 'result' to display results or 'empty' to skip.")
    else:
        try:
            answer = int(answer)
            if answer not in options:
                print("Choose a correct option")
                continue
            results.append(answer)
        except ValueError:
            print("Invalid input. Please enter a number between 1 and 4 or 'result' to display results or 'empty' to skip.")

    test_number += 1

    if end is not None and test_number > end:
        break

if end is not None:
    for idx, ans in enumerate(results):
        if ans is not None:
            print(f"The answer of {start+idx}: {ans}")
        else:
            print(f"The answer of {start+idx} was skipped.")


