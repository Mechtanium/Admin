import math
import os
import sys
from local_utils import *
import asyncio
import csv
import pandas as pd

URL = "https://docs.google.com/spreadsheets/d/1ZQbeOeCaiLMidenqmwq7wC-ni7rdtUYQXH1XER6XyyQ/edit#gid=0"
csv_url = URL.replace('/edit#gid=', '/export?format=csv&gid=')


def get_data():
    return pd.read_csv(csv_url)


async def load_data():
    with open("input/files_2.csv") as file:
        reader = csv.reader(file)
        for row in reader:
            await asyncio.sleep(1)
            print(row)


def round_to_n(x, n):
    x = x if x % 10 != 5 else x + 1
    n = n if x > 9 else n - 1
    return x if x == 0 else round(x, -int(math.floor(math.log10(abs(x)))) + (n - 1))


def run_junk():
    # print(round_to_n(73, 1))
    # print("\n\n", flush=True)
    # os.write(2, bytearray("Hello World from C\n", encoding="UTF-8", errors="e"))
    # asyncio.run(load_data())
    print(from_sec(83213))


run_junk()
