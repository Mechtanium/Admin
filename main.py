# 'dataset' holds the input data for this script
import os.path
import pickle as pkl

import gradio as gr
import pandas
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import export_graphviz

import datastore
from local_utils import WellDataPoint, well_col, time_col, ro_col, blind_col, flp_col, date_col, id_col, dur_col
from local_utils import man_col, temp_col, s1, s2, l1, l2, to_sec, out_folder, oversample_balance

N_EST = 10

null_mode = 'Null'

day_mode = '22-11-2020'
all_mode = 'All'
data_mode = all_mode

train_mode = 'Train'
test_mode = 'Test'
mode = train_mode

model_file = "rf-AWNW"
scaler_file = "ss-AWNW"


def clean_prepare_train(data_i, train_size=0.015, test_size=0.005):
    data_i.drop(axis=1, columns=[blind_col], inplace=True)
    data_i.dropna(axis=0, inplace=True, how="any")
    data_i.reset_index(inplace=True)
    data_i.drop(axis=1, columns=["level_0", id_col, "index"], inplace=True, errors="ignore")
    # data.to_csv("output/data.csv", index_label=id_col) if mode == all_mode else \
    #     data.to_csv("output/data_22.csv", index_label=id_col)
    dummies = pandas.get_dummies(data_i[well_col])
    data_i = pandas.concat([data_i, dummies], axis=1).reindex(data_i.index)
    data_i.drop(columns=[well_col], axis=1, inplace=True)
    data_i.drop(axis=1, columns=[flp_col, time_col, date_col], inplace=True, errors='ignore')
    y = data_i[ro_col]
    x_i = data_i.drop(axis=1, columns=[ro_col])
    # x.to_csv("output/x.csv", index_label=id_col)
    # y.to_csv("output/y.csv", index_label=id_col)
    # print(y.head())
    # print(f"{y.shape[0]} rows")
    # print(x.head())
    print(f"\n{x_i.shape[0]} rows")
    scaler_i = StandardScaler(copy=False)
    scaler_i.fit(x_i)
    x_fit = scaler_i.transform(x_i)
    x_train, x_test, y_train, y_test = \
        train_test_split(x_fit, y, random_state=30, train_size=train_size, test_size=test_size)
    model_i = RandomForestRegressor(n_estimators=N_EST, random_state=30)
    model_i.fit(x_train, y_train)
    y_pred = model_i.predict(x_test)
    score_i = r2_score(y_test, y_pred)
    x_test, y_test, y_pred = (pandas.DataFrame(x_test).reset_index(),
                              pandas.DataFrame(y_test).reset_index(),
                              pandas.DataFrame(y_pred).reset_index()
                              )
    data_run = pandas.concat([x_test, y_test, y_pred], axis=1).drop("index", axis=1)

    return model_i, scaler_i, score_i, x_i, data_run


def report_on(model_i, scaler_i, score_i, x_i):
    print(f"""
        \033[1;31mAI generalization stats\033[0m
        Model performance (rms score): \033[0;35m{score_i * 100:.2f}%\033[0m
        """)

    tests = [WellDataPoint(thp=661.84, day_sec=54100, man_pres=143.93, temp=93.9, _l1=0, _s1=1, _l2=0, _s2=0),
             WellDataPoint(thp=1118.456152, day_sec=86050, man_pres=166.063, temp=79.70630396, _l1=1, _s1=0, _l2=0,
                           _s2=0),
             WellDataPoint(thp=609.08, day_sec=42600, man_pres=137.2, temp=95.477, _l1=0, _s1=0, _l2=0, _s2=1),
             WellDataPoint(thp=1118.07, day_sec=49400, man_pres=146.44, temp=98.5, _l1=0, _s1=0, _l2=1, _s2=0)]

    for test in tests:
        print(f"\n{test}")
        try:
            test_x = pandas.DataFrame([test.get_x()], columns=x_i.columns)
            y_vis_pred = model_i.predict(scaler_i.transform(test_x))
            print(f"Real: \033[0;35m{test.get_y():.2f} psi\033[0m vs. "
                  f"Prediction: \033[0;35m{y_vis_pred[0]:.2f} psi\033[0m", flush=True)
        except ValueError:
            print(x_i.columns, flush=True)


def train(mode, best=(25, 10, 54, 0, 0)):
    if mode == day_mode:
        data = datastore.get_22_data()
        model, scaler, score, x, results = clean_prepare_train(data, train_size=0.75, test_size=0.25)
        write_state_files(model, scaler)
        results.to_csv(f"{out_folder}POWER_BI_DATA_DAY.csv", index_label=id_col)
        report_on(model, scaler, score, x)
    else:
        # get data payload
        data_dict = datastore.get_all_data()

        # search for the best offset combination model
        # best = find_best(data_dict, model_search, best)
        print(f"\033[1;31mFinal offsets\033[0m\n{s1}: {best[0]}, {l1}: {best[1]}, {s2}: {best[2]}, {l2}: {best[3]}")
        data = datastore.offset_wells(data_dict, [x for x in best[:4]])

        # remove unnecessary id columns
        data.drop(data.columns.difference([ro_col, dur_col, man_col, well_col, time_col, date_col, blind_col, flp_col,
                                           temp_col]), inplace=True, errors='ignore')

        # dump it
        data.to_csv(f"{out_folder}data_opt.csv", index_label=id_col)

        # balance it by oversampling
        data = oversample_balance(data)
        data.to_csv(f"{out_folder}data_opt_balanced.csv", index_label=id_col)

        # create model
        model, scaler, score, x, results = clean_prepare_train(data, train_size=0.75, test_size=0.25)
        write_state_files(model, scaler)
        results.to_csv(f"{out_folder}POWER_BI_DATA.csv", index_label=id_col)
        report_on(model, scaler, score, x)

        # try with no offsetting
        # data = datastore.offset_wells(data_dict, [0, 0, 0, 0])
        # data.drop(id_col, inplace=True, errors='ignore')
        # data.to_csv(f"{out_folder}data_base.csv", index_label=id_col)
        # model, scaler, score, x, _ = clean_prepare_train(data, train_size=0.75, test_size=0.25)
        # report_on(model, scaler, score, x)

        return model


def print_graph(model : RandomForestRegressor, x):
    for est, idx in zip(model.estimators_, len(model.estimators_)):
        file = f'tree_{idx}.dot'
        export_graphviz(model, out_file=file, feature_names=x.columns,
                        class_names=['extreme', 'moderate', 'vulnerable', 'non-vulnerable'],
                        rounded=True, proportion=False, precision=4, filled=True)


def write_state_files(model, scaler):
    pkl.dump(model, open(f"{model_file}.mdl", "wb"))
    pkl.dump(scaler, open(f"{scaler_file}.sts", "wb"))


def read_state_files(mdl, scl):
    mdl = pkl.load(open(f"{mdl}.mdl", "rb"))
    scl = pkl.load(open(f"{scl}.sts", "rb"))
    return mdl, scl


def model_search(dt_dict, s_1, l_1, s_2, l_2, current_best):
    dt = datastore.offset_wells(dt_dict, [s_1, l_1, s_2, l_2])
    _, _, scr, _, _ = clean_prepare_train(dt, train_size=0.75, test_size=0.25)
    scores_i = (s_1, l_1, s_2, l_2, scr)
    print(f"s1: {s_1}, l1: {l_1}, s2: {s_2}, l2: {l_2}, \033[0;35mscore: {scr * 100}\033[0m vs. "
          f"\033[1;31mbest: {current_best[4] * 100}\033[0m")
    return scores_i if scr > current_best[4] else current_best


def find_best(data_dict, model_search, best):
    for i in range(60):
        best = model_search(data_dict, i, best[1], best[2], best[3], best)
    for j in range(60):
        best = model_search(data_dict, best[0], j, best[2], best[3], best)
    for k in range(60):
        best = model_search(data_dict, best[0], best[1], k, best[3], best)
    for n in range(180):
        best = model_search(data_dict, best[0], best[1], best[2], n, best)
    return best


def change_well_to_dummy(wl):
    _l1, _l2, _s1, _s2 = 0, 0, 0, 0

    if wl == parse_well_id(l1):
        _l1 = 1
    elif wl == parse_well_id(s1):
        _s1 = 1
    elif wl == parse_well_id(l2):
        _l2 = 1
    elif wl == parse_well_id(s2):
        _s2 = 1

    return _l1, _l2, _s1, _s2


def app(hours, mins, secs, man_pres, temp, well, thp=None, regen=False):
    global test_x, y_vis_pred

    dur_sec = to_sec(hours, mins, secs)

    if regen or not (os.path.exists(f"{model_file}.mdl") and os.path.exists(f"{scaler_file}.sts")):
        train(data_mode)

    mdl, scl = read_state_files(model_file, scaler_file)

    thp = 0 if thp is None else thp

    _l1, _l2, _s1, _s2 = change_well_to_dummy(well)

    test = WellDataPoint(thp=thp, day_sec=dur_sec, man_pres=man_pres, temp=temp, _l1=_l1, _s1=_s1, _l2=_l2, _s2=_s2)
    columns = ['Daylight duration (SEC)', 'Manifold Pressure (PSI)', 'TEMP (°F)', '1L', '1S', '2L', '2S']
    try:
        test_x = pandas.DataFrame([test.get_x()], columns=columns)
        y_vis_pred = mdl.predict(scl.transform(test_x))
        print(f"Real: \033[0;35m{test.get_y():.2f} psi\033[0m vs. "
              f"Prediction: \033[0;35m{y_vis_pred[0]:.2f} psi\033[0m")
    except ValueError:
        print(test, flush=True)

    return f"{test.__plain__()}\nReal: {test.get_y():.2f} psi vs. Prediction: {y_vis_pred[0]:.2f} psi"


def parse_well_id(well_id):
    return f"Awoba NW {well_id}"


def parse_well_id_2(well_id):
    return f"Abura {well_id}"


def i_app(wl, pres):
    # match well to conversion factor
    factor = factors.loc[factors["Well"] == wl[6:]]["Conversion factor"]
    factor.reindex(axis=0)

    # return math result
    return pres - [f for f in factor][0]


# get conversion factors
factors = datastore.get_conversion_factors()

if mode == train_mode:
    app(23, 59, 40, 143.96, 79.523, parse_well_id(s2))
    app(17, 2, 0, 144.41, 97.278, parse_well_id(l1), regen=True)
else:
    with gr.Blocks() as demo:

        with gr.Tab("AI Approach"):
            hours = gr.Number(label="Hours (24-hour format)", value=23)
            mins = gr.Number(label="Minutes", value=59)
            secs = gr.Number(label="Seconds", value=40)
            man_pres = gr.Number(label=man_col, value=143.96)
            temp = gr.Number(label=temp_col, value=79.523)
            well = gr.Radio(
                [parse_well_id(w) for w in [l1, s1, l2, s2]],
                value=parse_well_id(s2),
                label="Select a well"
            )
            thp = gr.Number(label=ro_col, value=641.98)
            greet_btn = gr.Button("Simulate")
            greet_btn.style(full_width=True)
            output = gr.Textbox(label="Results")
            greet_btn.click(fn=app, inputs=[hours, mins, secs, man_pres, temp, well, thp], outputs=output)
        with gr.Tab("Isaac's Approach"):

            # build interface to take in well selection and manifold pressure
            i_man_pres = gr.Number(label=man_col, value=143.96)
            i_well = gr.Radio(
                [parse_well_id_2(w) for w in factors["Well"]],
                label="Select a well"
            )
            i_greet_btn = gr.Button("Simulate")
            i_greet_btn.style(full_width=True)
            i_output = gr.Textbox(label="Results")

            # call i_app function with params on button click
            i_greet_btn.click(fn=i_app, inputs=[i_well, i_man_pres], outputs=i_output)

    demo.launch()
