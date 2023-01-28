# 'dataset' holds the input data for this script
import os.path

import gradio as gr
import numpy
import pandas
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, mean_squared_error, \
    mean_squared_log_error, median_absolute_error, mean_absolute_percentage_error, r2_score, mean_poisson_deviance, \
    mean_gamma_deviance, mean_tweedie_deviance, d2_tweedie_score, mean_pinball_loss, d2_pinball_score, \
    d2_absolute_error_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import datastore
from local_utils import *

MAX_DEPTH = 20
N_EST = 10

mode = {"app": test_mode, "data": all_mode, "regen": False}


def clean_prepare_train(data_i, train_size=0.015, test_size=0.005):
    # drop sparse column THP BLIND then drop empty rows for all remaining columns
    data_i.drop(axis=1, columns=[blind_col], inplace=True)
    data_i.dropna(axis=0, inplace=True, how="any")
    data_i.reset_index(inplace=True)

    # change well_id to dummies
    dummies = pandas.get_dummies(data_i[well_col])
    data_i = pandas.concat([data_i, dummies], axis=1).reindex(data_i.index)
    data_i.drop(columns=[well_col], axis=1, inplace=True)

    # remove useless columns
    data_i = keep_useful_cols(data_i, [ro_col, dur_col, man_col, blind_col, temp_col] + dummies.columns.tolist())

    # get x and y
    y = data_i[ro_col]
    x_i = data_i.drop(axis=1, columns=[ro_col])

    # verify data row count
    print(f"\n{x_i.shape[0]} rows")

    # fit scaler
    scaler_i = StandardScaler(copy=False)
    scaler_i.fit(x_i)
    x_fit = pandas.DataFrame(scaler_i.transform(x_i), columns=x_i.columns)

    # data split
    x_train, x_test, y_train, y_test = \
        train_test_split(x_fit, y, random_state=30, train_size=train_size, test_size=test_size)

    # model
    model_i = RandomForestRegressor(n_estimators=N_EST, random_state=30, max_depth=MAX_DEPTH)
    model_i.fit(x_train, y_train)
    # print([est.get_depth() for est in model_i.estimators_])

    # testing
    y_pred = model_i.predict(x_test)
    score_i = r2_score(y_test, y_pred)
    # print("explained_variance_score:", explained_variance_score(y_test, y_pred))
    # print("max_error:", max_error(y_test, y_pred))
    # print("mean_absolute_error:", mean_absolute_error(y_test, y_pred))
    # print("mean_squared_error:", mean_squared_error(y_test, y_pred))
    # print("mean_squared_log_error:", mean_squared_log_error(y_test, y_pred))
    # print("median_absolute_error:", median_absolute_error(y_test, y_pred))
    # print("mean_absolute_percentage_error:", mean_absolute_percentage_error(y_test, y_pred))
    # print("r2_score:", r2_score(y_test, y_pred))
    # print("mean_poisson_deviance:", mean_poisson_deviance(y_test, y_pred))
    # print("mean_gamma_deviance:", mean_gamma_deviance(y_test, y_pred))
    # print("mean_tweedie_deviance:", mean_tweedie_deviance(y_test, y_pred))
    # print("d2_tweedie_score:", d2_tweedie_score(y_test, y_pred))
    # print("mean_pinball_loss:", mean_pinball_loss(y_test, y_pred))
    # print("d2_pinball_score:", d2_pinball_score(y_test, y_pred))
    # print("d2_absolute_error_score:", d2_absolute_error_score(y_test, y_pred))

    # create power_bi data payload
    x_test, y_test, y_pred = (pandas.DataFrame(x_test).reset_index(),
                              pandas.DataFrame(y_test).reset_index(),
                              pandas.DataFrame(y_pred, columns=[sim_col]).reset_index())
    data_run = pandas.concat([x_test, y_test, y_pred], axis=1).drop("index", axis=1)

    return model_i, scaler_i, score_i, x_i, data_run


def report_on(model_i, scaler_i, score_i, x_i):
    print(f"""
        \033[1;31mAI generalization stats\033[0m
        Model performance (rms score): \033[0;35m{score_i * 100:.2f}%\033[0m
        """)

    tests = [WellDataPoint(thp=661.84, day_sec=54100, man_pres=143.93, temp=93.9, _l1=0, _s1=1, _l2=0, _s2=0),
             WellDataPoint(thp=1118.456, day_sec=86050, man_pres=166.063, temp=79.706, _l1=1, _s1=0, _l2=0, _s2=0),
             WellDataPoint(thp=609.08, day_sec=42600, man_pres=137.2, temp=95.477, _l1=0, _s1=0, _l2=0, _s2=1),
             WellDataPoint(thp=1118.07, day_sec=49400, man_pres=146.44, temp=98.5, _l1=0, _s1=0, _l2=1, _s2=0)]

    for test in tests:
        print(f"\n{test}")
        try:
            test_x = pandas.DataFrame(scaler_i.transform(pandas.DataFrame([test.get_x()], columns=x_i.columns)),
                                      columns=x_i.columns)
            y_vis_pred = model_i.predict(test_x)
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
        if not os.path.exists(f"{out_folder}data_opt_balanced.csv"):
            data_dict = datastore.get_all_data()

            # search for the best offset combination model
            # best = find_best(data_dict, model_search, best)
            print(f"\033[1;31mFinal offsets\033[0m\n{s1}: {best[0]}, {l1}: {best[1]}, {s2}: {best[2]}, {l2}: {best[3]}")
            data = datastore.offset_wells(data_dict, [x for x in best[:4]])

            # remove unnecessary id columns
            data = keep_useful_cols(data)

            # balance it by oversampling
            data = oversample_balance(data)

            # dump it
            data.to_csv(f"{out_folder}data_opt_balanced.csv", index_label=id_col)
        else:
            data = pandas.read_csv(f"{out_folder}data_opt_balanced.csv")

        # create model
        model, scaler, score, x, results = clean_prepare_train(keep_useful_cols(data), train_size=0.75, test_size=0.25)
        write_state_files(model, scaler)
        results.to_csv(f"{out_folder}POWER_BI_DATA.csv", index_label=id_col)
        report_on(model, scaler, score, x)

        return model


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


def app(hours, mins, secs, man_pres, temp, well, thp=None, regen=False, full_text_reply=True):
    global test_x, y_vis_pred

    dur_sec = to_sec(hours, mins, secs)

    if regen or not (os.path.exists(f"{model_file}.mdl") and os.path.exists(f"{scaler_file}.sts")):
        train(mode['data'])

    mdl, scl = read_state_files(model_file, scaler_file)

    thp = 0 if thp is None else thp

    _l1, _l2, _s1, _s2 = change_well_to_dummy(well)

    test = WellDataPoint(thp=thp, day_sec=dur_sec, man_pres=man_pres, temp=temp, _l1=_l1, _s1=_s1, _l2=_l2, _s2=_s2)
    columns = ['Daylight duration (SEC)', 'Manifold Pressure (PSI)', 'TEMP (°F)', '1L', '1S', '2L', '2S']
    try:
        test_x = pandas.DataFrame(scl.transform(pandas.DataFrame([test.get_x()], columns=columns)), columns=columns)
        y_vis_pred = mdl.predict(test_x)
        print(f"Real: \033[0;35m{test.get_y():.2f} psi\033[0m vs. "
              f"Prediction: \033[0;35m{y_vis_pred[0]:.2f} psi\033[0m")
    except ValueError:
        print(test, flush=True)
        raise

    return f"{test.__plain__()}\nReal: {test.get_y():.2f} psi vs. Prediction: {y_vis_pred[0]:.2f} psi" if \
        full_text_reply else y_vis_pred


def i_app(wl, pres):
    # match well to factors
    factor = factors.loc[factors["Well"] == wl[6:]]

    # retrieve conversion and flow factor
    c_factor = factor["Conversion Factor"]
    f_factor = factor["Flow Factor"]

    # return math result
    return f"""\
Testing data
    Manifold pressure: {pres} psi
    Well: {wl}
    
Flowing tubing head pressure: {pres + [f for f in c_factor][0]:.2f} psi
Q-liquid: {pres * [f for f in f_factor][0]:.2f} bbl/day"""


scroll_data = pandas.read_csv(f"{out_folder}data_opt_balanced.csv")  # pandas.DataFrame()
n_real = 0
n_sim = 0
mn = 0
mx = 0
_, _, _, _, results = clean_prepare_train(scroll_data, train_size=0.50, test_size=0.50)
state_var = False
results.insert(0, id_col, numpy.array(range(results.shape[0])), False)

# randomize data rows and reset index
scroll_data = scroll_data.sample(frac=1)
scroll_data.drop([id_col, "index"], axis=1, inplace=True, errors="ignore")
scroll_data.insert(0, id_col, numpy.array(range(scroll_data.shape[0])), False)
y_range = min(scroll_data[ro_col]), max(scroll_data[ro_col])
# async def load_data():
#     global state_var
#     if not state_var:
#         state_var = True
#         global scroll_data
#         data = pandas.read_csv(f"{out_folder}data_opt_balanced.csv")
#         model, scaler, score, x, results = clean_prepare_train(keep_useful_cols(data), train_size=0.50, test_size=0.50)
#         i = 0
#
#         while i < results.shape[0]:
#             await asyncio.sleep(1)
#             i += 1
#             new_row = results.iloc[[i]]
#             print(new_row)
#             scroll_data = pandas.concat([scroll_data, new_row], ignore_index=True)
#             if scroll_data.shape[0] > 100:
#                 scroll_data.drop(0, axis=0, inplace=True)
#                 print(scroll_data.shape)


# URL = "https://docs.google.com/spreadsheets/d/1ZQbeOeCaiLMidenqmwq7wC-ni7rdtUYQXH1XER6XyyQ/edit#gid=0"
# csv_url = URL.replace('/edit#gid=', '/export?format=csv&gid=')
#
#
# def get_data():
#     return pandas.read_csv(csv_url)


def get_real_data() -> pandas.DataFrame:
    global results
    global mn
    global mx
    mx += 1
    mn = 0 if mx - 50 < 0 else mx - 50
    sl = results.iloc[mn:mx]
    sl.insert(0, time_col, numpy.array([from_sec(int(r)) for r in sl[id_col].tolist()]), False)
    return sl  # scroll_data


def get_sim_data() -> pandas.DataFrame:
    global results
    sl = results.iloc[mn:mx]
    sl.insert(0, time_col, numpy.array([from_sec(r) for r in sl[id_col].tolist()]), False)
    return sl  # scroll_data


x_real = 0
x_pres = 0
x_ql = 0


def get_x_real_data() -> pandas.DataFrame:
    global results
    sl = scroll_data.iloc[mn:mx]
    sl = sl.drop(time_col, axis=1, errors="ignore")
    sl.insert(0, time_col, numpy.array([from_sec(int(r)) for r in sl[id_col].tolist()]), False)
    return sl  # scroll_data


def get_x_sim_pres_data() -> pandas.DataFrame:
    global results
    sl = scroll_data.iloc[mn:mx]
    sl = sl.drop(sim_col, axis=1, errors="ignore")
    sl = sl.drop(time_col, axis=1, errors="ignore")
    sl.insert(0, time_col, numpy.array([from_sec(int(r)) for r in sl[id_col].tolist()]), False)
    sl.insert(0, sim_col, numpy.array([calc_excel(r)[0] for r in sl[man_col].tolist()]), False)
    return sl  # scroll_data


def get_x_sim_ql_data() -> pandas.DataFrame:
    global results
    sl = scroll_data.iloc[mn:mx]
    sl = sl.drop(time_col, axis=1, errors="ignore")
    sl.insert(0, time_col, numpy.array([from_sec(int(r)) for r in sl[id_col].tolist()]), False)
    sl.insert(0, ql_col, numpy.array([calc_excel(r)[1] for r in sl[man_col].tolist()]), False)
    return sl  # scroll_data


# get conversion factors
factors = datastore.get_conversion_factors()

if mode['app'] == train_mode:
    app(23, 59, 40, 143.96, 79.523, parse_well_id(s2))
    app(17, 2, 0, 144.41, 97.278, parse_well_id(l1), regen=mode['regen'])
else:
    with gr.Blocks() as demo:

        with gr.Tab("AI approach"):
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

        with gr.Tab("Excel approach"):
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

        with gr.Tab("Dashboard"):
            # pull data into line plot
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### AI real vs. calculated")
                    gr.LinePlot(value=get_real_data, y=ro_col, x=time_col, label="Awoba 2L", title="Real Tubing Head Pressure",
                                y_title=ro_col, x_title=time_col, every=1, height=150, width=600)
                    gr.LinePlot(value=get_sim_data, y=sim_col, x=time_col, label="Awoba 2L", title="Calculated Tubing Head Pressure",
                                y_title=sim_col, x_title=time_col, every=1, height=150, width=600)
                with gr.Column():
                    gr.Markdown("### Excel real vs. calculated")
                    gr.LinePlot(value=get_x_real_data, y=ro_col, x=time_col, label="Abura 2S", title="Real Tubing Head Pressure",
                                y_title=ro_col, x_title=time_col, every=1, height=150, width=600, y_lim=y_range)
                    gr.LinePlot(value=get_x_sim_pres_data, y=sim_col, x=time_col, label="Abura 2S", title="Calculated Tubing Head Pressure",
                                y_title=sim_col, x_title=time_col, every=1, height=150, width=600, y_lim=y_range)
                    gr.LinePlot(value=get_x_sim_ql_data, y=ql_col, x=time_col, label="Abura 2S", title="Calculated Production",
                                y_title=ql_col, x_title=time_col, every=1, height=150, width=600)

            # with gr.Column():
            #     with gr.Row():
            #         gr.LinePlot(value=get_real_data, y=ro_col, x=id_col, label="Real Tubing Head Pressure",
            #                     y_title=ro_col, x_title=time_col, every=1, height=80, width=600)
            #         gr.LinePlot(value=get_sim_data, y=sim_col, x=id_col, label="Calculated Tubing Head Pressure",
            #                     y_title=sim_col, x_title=time_col, every=1, height=80, width=600)
            #     with gr.Row():
            #         gr.LinePlot(value=get_real_data, y=ro_col, x=id_col, label="Real Tubing Head Pressure",
            #                     y_title=ro_col, x_title=time_col, every=1, height=80, width=600)
            #         gr.LinePlot(value=get_sim_data, y=sim_col, x=id_col, label="Calculated Tubing Head Pressure",
            #                     y_title=sim_col, x_title=time_col, every=1, height=80, width=600)

    demo.launch(enable_queue=True)

