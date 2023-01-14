import math
import re
import numpy
import pandas

l2 = "2L"
l1 = "1L"
s2 = "2S"
s1 = "1S"
date_time_col = "Date Time (GMT+01:00)"
time_col = "Time (GMT+01:00)"
dur_col = "Daylight duration (SEC)"
date_col = "Date"
id_col = "id"
well_col = "Well index"
blind_col = "THP BLIND (PSI)"
temp_col = "TEMP (°F)"
flp_col = "FLP (PSI)"
ro_col = "THP R/O (PSI)"
man_col = "Manifold Pressure (PSI)"
out_folder = "output/"
well_key = "wellhead"
flow_key = "flowstation"


def round_to_n(x, n):
    x = x if x % 10 != 5 else x + 1
    n = n if x > 9 else n - 1
    return x if x == 0 else round(x, -int(math.floor(math.log10(abs(x)))) + (n - 1))


def to_sec(h, m, s):
    return (int(h) * 60 * 60) + (int(m) * 60) + int(s)


def column_matcher(title):
    if re.search("#", string=title) is not None:
        found = id_col
    elif re.search(".*(Date|DATE).*(Time|TIME).*GMT.*", string=title) is not None:
        found = date_time_col
    elif re.search("THP.*R/O.*(PSI|units)", string=title) is not None:
        found = ro_col
    elif re.search(".*TEMP.*(F|units)", string=title) is not None:
        found = temp_col
    elif re.search(".*FLP.*(PSI|units)", string=title) is not None:
        found = flp_col
    elif re.search("THP.*BLIND.*(PSI|units)", string=title) is not None:
        found = blind_col
    elif re.search("THP.*(PSI|units)", string=title) is not None:
        found = blind_col
    elif re.search(".*1S.*PSI.*", string=title) is not None:
        found = s1
    elif re.search(".*2S.*PSI.*", string=title) is not None:
        found = s2
    elif re.search(".*1L.*PSI.*", string=title) is not None:
        found = l1
    elif re.search(".*2L.*PSI.*", string=title) is not None:
        found = l2
    else:
        found = False

    return found


def file_matcher(name: str):
    if re.search("\\d+-\\d+-\\d+.*flow.*man.*", string=name.lower()) is not None:
        flowstation = True
    else:
        flowstation = False

    return flowstation


def file_matcher2(name: str):
    if re.search(".*1s.*", string=name.lower()) is not None:
        well = s1
    elif re.search(".*1l.*", string=name.lower()) is not None:
        well = l1
    elif re.search(".*2s.*", string=name.lower()) is not None:
        well = s2
    else:
        well = l2

    return well


def restructure(data, count, duration, times, dates):
    for datetime in data[date_time_col]:
        try:
            date_time = re.sub("\\.0(?=\\s)", "", datetime)
            datetime_array = date_time.split()
            date = datetime_array[0].split("/")

            time_array = datetime_array[1].split(":")

            if datetime_array[2] == "PM" and time_array[0] != "12":
                hour = int(time_array[0]) + 12
            elif datetime_array[2] == "AM" and time_array[0] == "12":
                hour = int(time_array[0]) - 12
            else:
                hour = time_array[0]

            minutes = time_array[1]
            sec = round_to_n(int(time_array[2]), 1)

            if sec == 60:
                sec = "00"
                minutes = int(minutes) + 1

            if minutes == 60:
                minutes = "00"
                hour = int(hour) + 1

            if hour == 24:
                hour = "00"
                date[1] = int(date[1]) + 1

            duration.append(to_sec(hour, minutes, sec))
            times.append(f"{hour}:{minutes}:{sec}")
            dates.append(f"{date[1]}/{date[0]}/{date[2]}")
            date_time = f"{date[1]}/{date[0]}/{date[2]} {datetime_array[1]} {datetime_array[2]}"

            data.loc[count, date_time_col] = date_time
            count += 1
        except IndexError:
            print(f"\n\n{datetime}", flush=True)
            raise

    data.insert(1, dur_col, numpy.array(duration), True)
    data.insert(2, time_col, numpy.array(times), True)
    data.insert(3, date_col, numpy.array(dates), True)
    return data.drop(axis=1, columns="index", errors='ignore')


def try_key(temp, key):
    try:
        temp[f"{key}"]
    except KeyError:
        temp[f"{key}"] = dict()


def find_data(index, wlhd):
    for w in wlhd:
        if index == w[0]:
            return w[1]

    return None


def split_join(flowstation: pandas.DataFrame, wellhead: pandas.DataFrame, offset):
    joined = []
    info = [s1, l1, s2, l2]
    for i, o in zip(info, offset):
        # print(f'\n\nNow working on {i} column\n')
        data = flowstation.drop(flowstation.columns.difference([i, 'Daylight duration (SEC)']),
                                axis=1)
        data.rename(columns={i: man_col}, inplace=True)
        data.insert(2, well_col, [i for _ in range(data.shape[0])], True)

        # print(f"{data.shape[0]} rows before drop and merge")
        data_well = find_data(i, wellhead)
        if data_well is not None:
            data_well.drop_duplicates(inplace=True, subset=[time_col])
            data = data.merge(data_well, how='inner', on=[dur_col])

            # print(f"{data.shape[0]} rows after drop and merge")
            # offset the rows by the required amount 'o'
            data_y = data.drop(data.columns.difference([ro_col, id_col]), axis=1, errors="ignore").iloc[o:]
            data_x = data.drop(columns=[ro_col], axis=1, errors="ignore").iloc[:(data.shape[0] - 1 - o)]
            data_y.reset_index(inplace=True)
            data_x.reset_index(inplace=True)
            data_y.drop(columns=["index"], axis=1, inplace=True)
            data_x.drop(columns=["index"], axis=1, inplace=True)
            data = data_y.merge(data_x, how='inner', on=[id_col])
            joined.append((i, data))

    return joined


class WellDataPoint:

    def __init__(self, thp, day_sec, man_pres, temp, _l1=0, _s1=1, _l2=0, _s2=0):
        self.thp = thp
        self.day_sec = day_sec
        self.man_pres = man_pres
        self.temp = temp
        self.l1 = _l1
        self.s1 = _s1
        self.l2 = _l2
        self.s2 = _s2

    def __str__(self):
        day_sec, deli, i, man_pres, temp, well, well_titles = self.fields()
        return f"""\033[1;31mTesting data\033[0m
{day_sec:>20}{deli:3}{self.day_sec} seconds
{man_pres:>20}{deli:3}{self.man_pres} psi
{temp:>20}{deli:3}{self.temp} °F
{well:>20}{deli:3}{well_titles[i]}
"""

    def fields(self):
        deli = ' '
        day_sec = "Day duration:"
        man_pres = "Manifold Pressure:"
        temp = "Temperature:"
        well = "Well Name:"
        wells = [self.l1, self.l2, self.s1, self.s2]
        well_titles = ["Awoba NW 1L", "Awoba NW 2L", "Awoba NW 1S", "Awoba NW 2S"]  # List of well titles
        i = 0
        # Find the well with dummy value 1
        while not (wells[i]):  # not(0) yields true and not(anything else) yields false
            i += 1
        return day_sec, deli, i, man_pres, temp, well, well_titles

    def __plain__(self):
        day_sec, deli, i, man_pres, temp, well, well_titles = self.fields()
        space = '40'
        d_space = '3'
        return f"""Testing data
{day_sec:>{space}}{deli:{d_space}}{self.day_sec} seconds
{man_pres:>{space}}{deli:{d_space}}{self.man_pres} psi
{temp:>{space}}{deli:{d_space}}{self.temp} °F
{well:>{space}}{deli:{d_space}}{well_titles[i]}
"""

    def __repr__(self):
        return f"Practice([{self.day_sec}, {self.man_pres}, {self.temp}, {self.l1}, {self.s1}, {self.l2}, {self.s2}])"

    def get_x(self):
        return [self.day_sec, self.man_pres, self.temp, self.l1, self.s1, self.l2, self.s2]

    def get_y(self):
        return self.thp


def oversample_balance(data: pandas.DataFrame):
    # get buckets for control column
    data = data.astype(float, errors='ignore')
    mx = data[ro_col].max(axis=0, skipna=True)
    mn = data[ro_col].min(axis=0, skipna=True)
    rng = mx - mn
    bucket = rng / 10

    # shuffle data into buckets
    max_count = 0
    counter = mn
    temp = []
    results = []

    while counter < mx:

        sub_data = data[data[ro_col].between(counter, counter + bucket, inclusive='right')]
        if sub_data.shape[0] > 0:
            temp.append(sub_data)

        max_count = max_count if sub_data.shape[0] < max_count else sub_data.shape[0]

        counter += bucket

    for r in temp:
        counter = 0
        pumped_data = r
        print(r.shape, "\n", r.head())
        # add elements of r to pumped_data
        while pumped_data.shape[0] < max_count:
            new_row = r.iloc[[counter % r.shape[0]]]

            pumped_data = pandas.concat([pumped_data, new_row], ignore_index=True)

        # add final results to results series
        results.append(pumped_data)

    return pandas.concat(results, ignore_index=True)
