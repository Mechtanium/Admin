# 'dataset' holds the input data for this script
import re
import math
import pandas
from local_utils import date_time_col, time_col, man_col, dur_col, round_to_n, to_sec
import json
import os
import re
import pandas
from local_utils import date_time_col, dur_col, date_col, s1, s2, l1, l2, well_key, flow_key
from local_utils import restructure, try_key, column_matcher, file_matcher, file_matcher2, split_join as spj
id_col = "index"


def split_join(dt, info):
    joined = []
    for i in info:
        print(f'\n\nNow working on {i["data-column"]} column\n')
        data = dt.drop(axis=1, columns=i["non-columns"])
        data.rename(columns={i["data-column"]: man_col}, inplace=True)
        data.insert(2, "Well index", [i["well-index"] for x in range(data.shape[0])], True)
        print(f"{data.shape[0]} rows before drop and merge")
        data.drop_duplicates(inplace=True, subset=[time_col])
        i["dataset"].drop_duplicates(inplace=True, subset=[time_col])
        print(data.head())
        print(i["dataset"].head())
        data = data.merge(i["dataset"], how='left', on=[time_col, dur_col])
        print(data.head())
        print(f"{data.shape[0]} rows after drop and merge")
        joined.append(data)

    return pandas.concat(joined, ignore_index=True)


def get_22_data():
    dataset = pandas.read_csv("input/flowstation.csv")
    dataset_1S = pandas.read_csv("input/1S.csv")
    dataset_1L = pandas.read_csv("input/1L.csv")
    dataset_2S = pandas.read_csv("input/2S.csv")
    dataset_2L = pandas.read_csv("input/2L.csv")

    print(dataset.head())

    for dat in [dataset, dataset_1S, dataset_1L, dataset_2L, dataset_2S]:
        count = 0
        duration = []
        times = []
        dat.dropna(axis=0, how="any", inplace=True)
        dat.reset_index(inplace=True)
        print(".", end="\t")

        for datetime in dat[date_time_col]:
            date_time = re.sub("\.0(?=\\s)", "", datetime)
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
            date_time = f"{date[1]}/{date[0]}/{date[2]} {datetime_array[1]} {datetime_array[2]}"

            dat.loc[count, date_time_col] = date_time
            count += 1

        dat.insert(1, dur_col, duration, True)
        dat.insert(2, time_col, times, True)

        dat.drop(axis=1, columns=["#", date_time_col], inplace=True, errors="ignore")

    info_1S = {
        "non-columns": ["index", "NW1L, PSI (LGR S/N: 20705686)", "NW2L, PSI (LGR S/N: 20705686)",
                        "NW2S, PSI (LGR S/N: 20705686)"],
        "data-column": 'NW1S, PSI (LGR S/N: 20705686)',
        "well-index": '1S',
        "dataset": dataset_1S
    }

    info_1L = {
        "non-columns": ["index", "NW1S, PSI (LGR S/N: 20705686)", "NW2L, PSI (LGR S/N: 20705686)",
                        "NW2S, PSI (LGR S/N: 20705686)"],
        "data-column": 'NW1L, PSI (LGR S/N: 20705686)',
        "well-index": '1L',
        "dataset": dataset_1L
    }

    info_2S = {
        "non-columns": ["index", "NW1S, PSI (LGR S/N: 20705686)", "NW1L, PSI (LGR S/N: 20705686)",
                        "NW2L, PSI (LGR S/N: 20705686)"],
        "data-column": 'NW2S, PSI (LGR S/N: 20705686)',
        "well-index": '2S',
        "dataset": dataset_2S
    }

    info_2L = {
        "non-columns": ["index", "NW1S, PSI (LGR S/N: 20705686)", "NW1L, PSI (LGR S/N: 20705686)",
                        "NW2S, PSI (LGR S/N: 20705686)"],
        "data-column": 'NW2L, PSI (LGR S/N: 20705686)',
        "well-index": '2L',
        "dataset": dataset_2L
    }

    dataset = split_join(dataset, [info_1S, info_1L, info_2S, info_2L])
    dataset.drop(axis=1, columns=[id_col], inplace=True, errors="ignore")
    dataset.insert(0, id_col, [x for x in range(dataset.shape[0])], True)

    return dataset.drop(axis=1, columns="level_0")


def get_all_data():
    # get the list of data files and load each into a dataframe, put the dataframes in a list called files.
    alldata = pandas.read_csv("input/files.csv")
    files = []
    i = 0
    for filepath in alldata["File Path"]:
        i += 1
        try:
            files.append((pandas.read_csv(filepath)))
        except UnicodeDecodeError:
            os.write(2, bytearray(f"Failed {i} - {filepath}\n", encoding="UTF-8", errors="e"))
            raise
    print(f"{len(files)} of {alldata.shape[0]} loaded")
    # for each dataframe in files, standardize the column names and remove columns with too few values
    truth = 0
    total = 0
    cut_off = 0.4
    temp = []
    for file, name in zip(files, alldata["Name"]):
        for col in file.columns:
            total += 1
            result = column_matcher(col)
            if not result:
                file.drop(axis=1, columns=[col], inplace=True)
            else:
                file.rename(columns={col: result}, inplace=True)
                if file[result].isna().sum() / file.shape[0] > cut_off:
                    truth += 1
                    file.drop(axis=1, columns=[result], inplace=True)
        file.dropna(axis=0, how="any", inplace=True)
        temp.append(file)
    files = temp
    print(f"{truth}/{total} columns dropped due to insufficient data at {cut_off * 100:.0f}% cut-off")
    temp = dict()
    # restructure data (extract and correct time, remove unnecessary column and add new once)
    # remove null rows, drop first row in case of gibberish headers and group dataframes by day in dict
    for index, dat, name, path in zip(range(len(files)), files, alldata["Name"], alldata["File Path"]):
        try:
            count = 0
            duration = []
            times = []
            dates = []
            dat.dropna(axis=0, how="any", inplace=True)
            dat.drop(0, axis=0, inplace=True, errors="ignore")
            dat.reset_index(inplace=True)
            print(f"•{index + 1:^4}•", end="  ")
            dat = restructure(dat, count, duration, times, dates)

            dat.drop(axis=1, columns=date_time_col, inplace=True, errors="ignore")
            print(f"{str(sorted([x for x in dat.columns])):<123}  •")
            # dat.to_csv(f"output/temp/{name}", index=False)

            if file_matcher(name):
                # flowstation file
                key = re.split("(?<=\\d{2})(-|_|\\s)(?=flow.*)", string=name.lower(), maxsplit=2)[0][:8]
                try_key(temp, key)

                temp[key][flow_key] = dat

            else:
                # wellhead file
                key = re.split("(?<=\\d{2})(\\s-|-|_|\\s)?(?=(t|\\d[ls]).*)", string=name.lower(), maxsplit=2)[0][:8]
                try_key(temp, key)

                try:
                    temp[key][well_key]
                except KeyError:
                    temp[key][well_key] = []

                temp[key][well_key].append((file_matcher2(name.lower()), dat))

        except KeyError:
            print(f"\n\n{path}", flush=True)
            print("Columns:", dat.columns, end="\n\n", flush=True)

    return temp


def get_conversion_factors():
    # get conversion_factors.csv from input in a dataframe
    return pandas.read_csv("input/conversion_factors.csv", encoding="UTF-8")
    # return dataframe
    pass


def offset_wells(agg_data, how=None):
    # Prepare aggregated data for each well from all days
    if how is None:
        how = [0, 0, 0, 0]

    data_1S = []
    data_2S = []
    data_1L = []
    data_2L = []
    for key in agg_data.keys():
        try:
            temp = spj(agg_data[key][flow_key], agg_data[key][well_key], how)  # returns list of (well id, data) tuples
            for t in temp:

                if t[0] == s1:
                    data_1S.append(t[1])
                elif t[0] == s2:
                    data_2S.append(t[1])
                elif t[0] == l1:
                    data_1L.append(t[1])
                elif t[0] == l2:
                    data_2L.append(t[1])
        except KeyError:
            pass
    data_1L = pandas.concat(data_1L, ignore_index=True)
    data_2L = pandas.concat(data_2L, ignore_index=True)
    data_1S = pandas.concat(data_1S, ignore_index=True)
    data_2S = pandas.concat(data_2S, ignore_index=True)

    data_s = []
    for name, data in zip([l1, l2, s1, s2], [data_1L, data_2L, data_1S, data_2S]):
        data.drop_duplicates(subset=[dur_col, date_col], inplace=True)
        data_s.append(data)
        # data.to_csv(f"output/{name}.csv", index=False)

    return pandas.concat(data_s, ignore_index=True)
