# modeling some simple bank operation:
# this script is python-course-driven
import os
os.system("cls")


def show_deposit():
    print(f"Your balance is ${balance:.2f}")


def deposit():
    amount = float(input("Enter an amount to be deposited\n>>> "))
    if amount <= 0:
        print("that is not a valid amount")
        return 0  # if you don't return 0 and just skip it, you get TypeError: cause you didn't determine the behaviour of program

    else:
        return amount


def withdrawl():
    amount = float(input("How much do you want to withdraw?\n>>> "))
    if amount > balance:
        print(f"Not enough funds to withdraw ${amount}")
        return 0

    elif amount <= 0:
        print("Undrawable amount")
        return 0

    else:
        # if none of above conditions occur, the amount is entered correctly, so we return it as valid.
        return amount


balance = 0

choices = ("***************************", "Welcome to banking program:",
           "***************************", "1-Show balance", "2-Deposit", "3-Withdraw", "4-Exit")

is_running = True

while is_running:
    for items in choices:
        print(items)

    choice = input("\nChoose one:\n>>> ")

    if choice == "1":
        show_deposit()

    elif choice == "2":
        # a such a nice strategy: deposit is counted by refering to deposit() and then added to balance
        balance += deposit()

    elif choice == "3":
        balance -= withdrawl()

    elif choice == "4":
        is_running = False

    else:
        print("Not a valid input.")
