import sys

run = True
while run:
    try:
        def calculator():
            while True:
                num1 = int(input("Enter the first number:\n"))
                op = input("Enter the operator: (+, -, *, /)\n")
                num2 = int(input("Enter the second number:\n"))
                options = ["+", "-", "*", "/"]
                if op not in options:
                    print("This is a value error")
                    sys.exit()  # Exit the program if an invalid operator is entered
                else:
                    print("The result is:")
                    if op == "+":
                        print(num1 + num2)
                    elif op == "-":
                        print(num1 - num2)
                    elif op == "*":
                        print(num1 * num2)
                    elif op == "/":
                        if num2 != 0:
                            print(num1 / num2)
                        else:
                            print("Cannot divide by zero")
                            return  # Exit the calculator function
        calculator()
    except ValueError:
        print("Enter the correct value")
    except Exception as e:
        print(e)
    run = False  # Exit the while loop
