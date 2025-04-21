# A simple project wich generates you a customized password
import string
import random as rnd
run = True
while run:
    c_numbers = input("How many nubmers must be used? ")
    capital_letters = input("How many capital letters must be used? ")
    small_letters = input("How many small letters must be used? ")

    if small_letters.isdigit() and capital_letters.isdigit() and c_numbers.isdigit():
        capital_letters = int(capital_letters)
        small_letters = int(small_letters)
        numbers = int(c_numbers)

        break
    else:
        print("not valid")
        continue

capitals = string.ascii_uppercase
smalls = string.ascii_lowercase
numbers = string.digits

c = rnd.sample(capitals, k=capital_letters)
s = rnd.sample(smalls, k=small_letters)
n = rnd.sample(numbers, k=capital_letters)
final = c + s + n
rnd.shuffle(final)
final_pass = "".join(final)
print("\n" + "---->", final_pass + "\n")
print("{} small letters are used , {} capital letters are used , {} numbers are used.".format(
    small_letters, capital_letters, c_numbers))
