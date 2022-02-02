from math import ceil


import math
l = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
l[::2] = ["A"] * math.ceil(len(l)/2)
print(l)
