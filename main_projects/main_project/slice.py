def nu(number):
    numbers = number[::-1]  # reverses the list
    grouped_by_three = []  # append nums by each three numbers
    for i in range(0, len(numbers), 3):
        # consider 123456789: i = 0: "123" | i = 1: "456" | i = 2: "789"
        grouped_by_three.append(numbers[i: i + 3])
        # reverse back
    result = ','.join(grouped_by_three)[::-1]
    return result


print(nu('10000000'))
