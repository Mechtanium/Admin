# 'dataset' holds the input data for this script
import pandas
from datastore import get_22_data, split_join

date_time_col = "Date Time (GMT+01:00)"
time_col = "Time (GMT+01:00)"
dur_col = "Daylight duration (SEC)"
id_col = "index"


data = get_22_data()
data.drop(axis=1, columns=["THP BLIND (PSI)"], inplace=True)
data.dropna(axis=0, inplace=True, how="any")
data.reset_index(inplace=True)
data.drop(axis=1, columns="level_0", inplace=True)
dummies = pandas.get_dummies(data["Well index"])
data = pandas.concat([data, dummies], axis=1).reindex(data.index)
data.drop(columns=["Well index", "index"], axis=1, inplace=True)
# data.to_csv("output/data.csv", index_label="id")
