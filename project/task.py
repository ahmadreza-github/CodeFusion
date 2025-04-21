options = ["1-Add a new task", "2-Delete a task", "3-Show the tasks"]
nu_opt = ["1", "2", "3"]
run = True
results = []

while run:
    for option in options:
        print(option)
    oper = input('Enter the operation:\n>>> ')

    if oper == "1":
        first_op = input("What is the task you desire to add?\n>>> ")
        if first_op != "":
            results.append(first_op)
            print(f"The task '{first_op}' is added")
        else:
            print("The task box is still empty, add something")

    if oper not in nu_opt:
        print("Enter just a number between 1 and 3!\n")

    if oper == "3":
        for idx, present in enumerate(results, start=1):
            print(f"The task number {idx} is: {present}")

    if oper == "2":
        del_inp = input("Enter the number of the task that you want to delete:\n>>> ")
        if del_inp.isdigit():  # Check if input is a valid number
            # Adjust index since user input starts from 1
            del_idx = int(del_inp) - 1
            if 0 <= del_idx < len(results):  # Check if index is within range
                # Remove task at specified index
                deleted_task = results.pop(del_idx)
                print(f"Task '{deleted_task}' deleted successfully")
            else:
                print("Invalid task number. Please enter a valid task number.")
        else:
            print("Invalid input. Please enter a valid task number.")
