# a script which converts:

# kilogram to pounds
# pound to kilograms
# Fahrenheit to Celcius
# Celcius to Fahrenheit
# Hours and minutes to seconds
# Seconds to hours and minutes


def converter():
    run = True
    while run:
        choice_input = input(
            "1-kilogram to pounds\n2-pound to kilograms\n3-Fahrenheit to Celcius\n4-Celcius to Fahrenheit\n5-Hours and minutes to seconds\n6-Seconds to hours and minutes\n(1 / 2 / 3 / 4 / 5 / 6): ").strip()

        if choice_input == "1":
            kilo_to_lbs = int(input("Enter your weight in kilograms?\n>>>"))

            answer = round(kilo_to_lbs * 2.20462262, ndigits=2)
            print(f"\n-----> You are ({answer}) pounds <-----\n")

        elif choice_input == "2": 
            lbs_to_kilos = int(input("Enter your weight in pounds?\n>>> "))
            answer = round(lbs_to_kilos * 0.454, ndigits=2)
            print(f"\n-----> You are ({answer}) kilos <-----\n")

        elif choice_input == "3":
            fahrenheit_to_celsius = int( 
                input('Enter the temperature in fahrenheit\n>>>'))
            answer = (fahrenheit_to_celsius - 32) * 5 / 9  # 5/9 is equal to 0.5555555555555556
            print(
                f"\n-----> ({fahrenheit_to_celsius}) fahrenheit is ({answer}) celcius <-----\n")

        elif choice_input == "4":
            cel_to_fahren = int(
                input('Enter the temperature in celcius\n>>> '))
            answer = (cel_to_fahren * 9/5) + 32
            print(
                f"\n-----> ({cel_to_fahren}) celcius is ({answer}) fahrenheit <-----\n")
        elif choice_input == "5":
            hour = int(input("How many hours?\n>>> "))
            minutes = input("How many minutes?\n>>> ")
            if minutes == "":
                minutes = 0
            else:
                minutes = int(minutes)

            answer_hour = hour * 3600 

            answer_minute = minutes * 60  

            final_answer = answer_hour + answer_minute
            print(f"\n\n---> ({hour:02}:{minutes:02}) has ({final_answer}) seconds.\n\n")

        elif choice_input == "6":
            mytime = int(input("Enter the time in seconds: "))
            mytime += 1  # mytime = mytime + 1 : cause python's behaviour
            for x in range(mytime):
                second = x % 60
                minutes = int(x / 60) % 60
                hours = int(x / 3600)
            print(f"\n\n---> ({hours:02}:{minutes:02}:{second:02})\n\n")
        else:
            print("Invalid entry")
converter() 






