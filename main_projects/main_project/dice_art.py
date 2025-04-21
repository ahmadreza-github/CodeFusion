# this game is created by watching a python course:

import random 

dice_art = {
    1: ("┌─────────┐",
        "│         │",
        "│    ●    │",
        "│         │",
        "└─────────┘"),
    
    2: ("┌─────────┐",
        "│         │",
        "│   ● ●   │",
        "│         │",
        "└─────────┘"),
    
    3: ("┌─────────┐",
        "│      ●  │",
        "│    ●    │",
        "│  ●      │",
        "└─────────┘"),

    4: ("┌─────────┐",
        "│  ●   ●  │",
        "│         │",
        "│  ●   ●  │",
        "└─────────┘"),
        
    5: ("┌─────────┐",
        "│  ●   ●  │",
        "│    ●    │",
        "│  ●   ●  │",
        "└─────────┘"),

    6: ("┌─────────┐",
        "│ ●  ●  ● │",
        "│         │",
        "│ ●  ●  ● │",
        "└─────────┘")
}

dice = []
total = 0
number_of_dice = int(input("How many dice?\n>>> "))

for die in range(number_of_dice): # iot generate a random number of dice based on user's input (number_of_dice)
    dice.append(random.randint(1 , 6))

# for die in range(number_of_dice) # horizontal dice
#     for line in dice_art.get(dice[die]):
#         print(line)

for line in range(5):
    for die in dice:
        print(dice_art.get(die)[line] , end = " ")
    print()

for die in dice: # all values in "die" will append and add to total
    total += die
print(f"Total: {total}")