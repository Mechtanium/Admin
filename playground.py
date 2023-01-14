import math
import os
import sys


def run_junk():
    print("\n\n", flush=True)
    os.write(2, bytearray("Hello World from C\n", encoding="UTF-8", errors="e"))


def round_to_n(x, n):
    x = x if x % 10 != 5 else x + 1
    n = n if x > 9 else n - 1
    return x if x == 0 else round(x, -int(math.floor(math.log10(abs(x)))) + (n - 1))


print(round_to_n(73, 1))
run_junk()
