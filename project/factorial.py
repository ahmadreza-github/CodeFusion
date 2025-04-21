run = True
while run:
    try:
        def factorial(n):
            if n == 0:
                return 1
            else:
                return n * factorial(n-1)

        user_input = input(
            "Enter a number to calculate its factorial (or type 'Exit' to quit): ")

        if user_input.lower() == "exit":
            run = False
            print("Exited successfully")
        else:
            num = int(user_input)

            # Check if the number is negative
            if num < 0:
                print("Factorial is not defined for negative numbers.")

            elif num == 0:
                print("The factorial of 0 is 1.")
            else:
                result = factorial(num)
                print(f"The factorial of {num} is {result}.")
    except ValueError:
        print("Enter a correct value!")
