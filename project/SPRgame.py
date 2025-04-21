import random
from rich import print

while True:
    pc_score = 0
    user_score = 0
    run = True

    while run:
        options = ["scissors", "paper", "rock"]
        user_choice = input(
            "Enter your choice between options: (scissors, paper, rock):\n")
        print(f" [underline bold] Your choice is: {
              user_choice} [/underline bold]")

        pc_choice = random.choice(options)
        print(f" [underline bold] PC choice: {pc_choice} [/underline bold]")

        if user_choice not in options:
            print(f"Enter a correct choice among: {options}")
        else:
            if pc_choice == user_choice:
                print(f" [underline bold] Equal => Try again [/underline bold]")

            elif pc_choice == "scissors":
                if user_choice == "paper":
                    pc_score += 1
                else:
                    user_score += 1

            elif pc_choice == "paper":
                if user_choice == "rock":
                    pc_score += 1
                else:
                    user_score += 1

            elif pc_choice == "rock":
                if user_choice == "scissors":
                    pc_score += 1
                else:
                    user_score += 1

            if pc_score == 3:
                print("[underline bold] PC won [/underline bold]")
                print("You screwed it up! \U0001F4A9  \U0001F923 ")

                run_again = input(
                    "Would you like to play again? (y/n): ").lower()
                if run_again != "y":
                    run = False
                else:
                    break  # Exit the inner loop and restart the game

            elif user_score == 3:
                print("[underline bold] User won [/underline bold]")
                print("Would you like to play again?")

                run_again = input(
                    "Would you like to play again? (y/n): ").lower()
                if run_again != "y":
                    run = False
                else:
                    break  # Exit the inner loop and restart the game

        print(f"[underline bold] User score is: {
              user_score} [/underline bold]")
        print(f"[underline bold] PC score is: {pc_score} [/underline bold]")
