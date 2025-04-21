import random
options = [1,  2, 3, 4]
run = True
while run:
    try:

        user_choice = int(input("Enter your choice:\n"))
        pc_choice = random.choice(options)
        print("pc choice is: ", pc_choice)

        if user_choice not in options:
            print("Invalid choice!")
        if pc_choice == user_choice:
            print("you won!")
            print("same number")
    except ValueError:
        print("Enter a correct value!")
