# hi, take a look at this code and tell me how to do something that:
# i put a symbol like (!) in front of a test number like: 4! and the program will show that symbol in front of that number when representing results
# perplexity
from rich import print
start = int(input(f"what is the start number? "))
end = input(f"what is the end number? ")

if end == "*":
    end = None
    print("[purple]The end is set to limitless[/purple]")
else:
    end = int(end)

print(f"[bold italic][yellow]From ({start}) to ({
      end}), let's go![/bold italic][/yellow]")

options = ["1", "2", "3", "4", "#", "*", "result"]

result = []
run = True

current_number = start
while run:
    ans_inp = input(f"what is the answer of test number {
                    current_number}?\n>>> ")
    if ans_inp == "#":
        print("skipped")
        result.append(None)
    elif ans_inp != "":
        if ans_inp in options:
            if ans_inp != "result":
                result.append(ans_inp)
        else:
            print("Enter a valid answer!")
            continue
    else:
        print("[underline italic]No answer entered![/underline italic]")
        continue

    current_number += 1
    if ans_inp == "result":
        run = False
    elif end is not None and current_number > end:
        run = False
c = input("You want the results?(y/n)\n>>>")
if c == "y":

    for idx, ans in enumerate(result, start):
        if ans is not None:
            print(f"the answer of test number ({idx}) is: {ans}\n")
        else:
            print(f"the answer of test number ({idx}) was skipped.\n")
else:
    run = False
