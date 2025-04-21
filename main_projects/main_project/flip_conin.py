# a project to evaluate you luck:
import random as rnd
play_time = 10
saved = []
run = True
index = 1
choices = ["obversef(head)" , "reverse (tale)"]

head_count = 0
tail_count = 0

while run:
    print(f"Index number is: {index}")
    user_opt = input("FILP? (y/n)?" )  
    choice = user_opt.lower().strip() 

    if choice == "n":
        run = False

    elif choice == "y" and choice.isalpha():
        random_choice = rnd.choice(choices) 
        print(f"You got a {random_choice} this time!\n")
        saved.append(random_choice)
        index += 1 
        if user_opt == "obversef(head)":
            head_count += 1
        else:
            tail_count += 1 
        if index == 10:
            print("Done, Here are results")
            run = False
            for index , saved in enumerate(saved , start = 1):
                print(f"you totaly got:{index}: {saved}") 
            if head_count > tail_count:
                print(f"Fially head won!")

            elif tail_count > head_count:
                print(f"Fially tail won!")
            else:
                print("This is a tie!")
                print("run the program and try again!")

    else:
        print("an invalid option was used!")
        continue 