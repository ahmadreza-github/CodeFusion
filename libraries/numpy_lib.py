import numpy as np
c = np.array([1, 2, 3, 4, 5])
d = np.array([6, 7, 8])
e = np.concatenate((c, d))


# broadcast:
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = a ** b  # a * b | a + b | a - b and ...
# or
d = np.array([10])  # 10 will be multiplied by all elements in (a)
n = d * a
# or
new = b + 7  # all elements inside (b) will be added to (7)

# slice from a list
c = np.array([1, 2, 3, 4, 5, 6, 7, 8])
b = c[2:5]

# sum, min, mix, sorted of an array
c = np.array([1, 2, 3, 23, 4, 5, 6, 7, 8, 29])
sum_c = np.sum(c)
max_c = np.max(c)
min_c = np.min(c)
sorted_c = np.sort(c)


x2 = np.arange(0, 7, 0.5)  # a list from 0 to 7 that has gone forward by 5.0


array = np.array([[1, 2], [3, 4]])  # a two dimensional array
array2 = np.array([8, 1, 5, 8])  # an only dimentional array
# change the size, pay attention to multiplies of both
array_2 = array2.reshape((4, 1))

print(array.shape)  # provides you with the dimention of an array

a = np.array([1, 2, 3, 4, 5, 6])
c = a.reshape(2, 3)  # usage of reshape
print(c.shape)

# generate a random matrix:
g = np.random.randint(1, 100, size=(3, 4))  # size = (column , line)

c = a[a > 4]  # (c) is equal to all elements bigger than 4 inside (a)

# vstack and hstack: vstack: vertical stack , hstack = horizontal stack
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.vstack((a, b))
v = np.hstack((a, b))  # by tuple

n = np.array([5, 7, 12])

print("..................")
mean = (n.mean())
std = (n.std())
norm_a = (n - mean) / std
print(norm_a)
