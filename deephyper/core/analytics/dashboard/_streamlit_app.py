import json
import os
import tempfile

import copy
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd
import streamlit as st
from deephyper.core.analytics.dashboard._pyplot import plot_single_line, plot_single_line_improvement
from deephyper.core.analytics.dashboard._results_processors import (
    DefaultProcessor,
    PercUtilProcessor,
    ProfileProcessor,
    SearchProcessor,
)
from tinydb import Query, TinyDB


def _worker_utilization(profile, num_workers):
    # compute worker utilization
    t0 = profile.iloc[0].timestamp
    t_max = profile.iloc[-1].timestamp
    T_max = (t_max - t0) * num_workers

    cum = 0
    for i in range(len(profile.timestamp) - 1):
        cum += (
            profile.timestamp.iloc[i + 1] - profile.timestamp.iloc[i]
        ) * profile.n_jobs_running.iloc[i]
    perc_util = cum / T_max

    return perc_util * 100


def _files_selection(uploaded_file):
    # Can be used wherever a "file-like" object is accepted:
    df = pd.read_csv(uploaded_file)

    # assuming it's a search csv
    if (
        "objective" in df.columns
        and "elapsed_sec" in df.columns
        and "duration" in df.columns
        and "id" in df.columns
    ):
        df["iteration"] = df.index

        st.subheader("Scatter Plot")
        st.sidebar.header("Scatter Plot")

        line_plot_option_x = st.sidebar.selectbox(
            "Choose the X-axis",
            tuple(df.columns),
            index=list(df.columns).index("iteration"),
        )
        line_plot_option_y = st.sidebar.selectbox(
            "Choose the Y-axis",
            tuple(df.columns),
            index=list(df.columns).index("objective"),
        )

        outlier_threshold = st.sidebar.slider(
            "Outlier Threshold", min_value=0.0, max_value=5.0, value=3.0
        )

        min_float = np.finfo(np.float32).min
        has_failed = (
            (abs(df[line_plot_option_y] - min_float) < 1e-3)
            | (df[line_plot_option_y] < min_float)
        )
        n_failures = sum(has_failed.astype(int))

        if n_failures > 0:
            st.warning(f"**{n_failures}** failure{'s' if n_failures > 1 else ''} detected!")

        df = df[~has_failed]
        df = df[(np.abs(stats.zscore(df[line_plot_option_y])) < outlier_threshold)]

        min_y = st.sidebar.number_input("Min Y: ", value=df[line_plot_option_y].min())
        max_y = st.sidebar.number_input("Max Y: ", value=df[line_plot_option_y].max())

        df = df[(min_y < df[line_plot_option_y]) & (df[line_plot_option_y] < max_y)]

        plot_single_line(df, line_plot_option_x, line_plot_option_y)

        st.subheader("Line Plot")
        plot_single_line_improvement(df, line_plot_option_x, line_plot_option_y)

        st.subheader("Top-K Configurations")
        st.sidebar.header("Top-K Configurations")

        k = st.sidebar.number_input("Number of displayed headers: ", min_value=1, max_value=len(df), value=5)
        df = df.sort_values(by=["objective"], ascending=False, ignore_index=True)

        subdf = df.iloc[:k]
        st.dataframe(subdf)

    # assuming it is a profile csv
    elif "timestamp" in df.columns and "n_jobs_running" in df.columns:
        df.set_index(df.columns[0])

        st.header("Worker Utilization")
        num_workers = st.number_input("Number of Workers", value=df.n_jobs_running.max())

        perc_ut = _worker_utilization(df, num_workers)

        fig = plt.figure()
        plt.pie(
            [perc_ut, 100 - perc_ut],
            explode=(0.1, 0),
            labels=["Used", "Not Used"],
            autopct="%1.1f%%",
            shadow=True,
            startangle=90,
        )
        st.pyplot(fig)
        plt.close()

        t0 = df.iloc[0].timestamp
        df["timestamp"] = df.timestamp - t0

        t0 = float(df.iloc[0].timestamp)
        t_max = float(df.iloc[-1].timestamp)

        t0, t_max = st.slider(
            "Time Range", min_value=t0, max_value=t_max, value=(t0, t_max)
        )

        df = df[(df.timestamp >= t0) & (df.timestamp <= t_max)]
        fig = plt.figure()
        plt.step(df.timestamp, df.n_jobs_running, where="post")
        plt.xlabel("Time (sec.)")
        plt.ylabel("Number of Used Workers")
        st.pyplot(fig)
        plt.close()

    else:
        st.write("Sorry but this type of CSV is not supported!")


def _database_selection(uploaded_file):
    temp_db = tempfile.NamedTemporaryFile(mode="w+")
    json.dump(json.load(uploaded_file), temp_db)
    temp_db.read()
    db = TinyDB(temp_db.name)

    # Get the selectable caracteristics from the database
    selection = db.all()
    criterias = {}
    not_criterias = ["description", "results", "date", "random_state"]
    for select in selection:
        criterias = _get_criterias(select, criterias, not_criterias)

    # Show the possible criterias
    with st.sidebar.expander("Select benchmark criterias :"):
        choices, differences = _select_choices(criterias, 1)

        # Select the corresponding runs
        query = _generate_query(choices, Query())
        selection = db.search(query)
        select_size = len(selection)
        st.markdown("------")
        if select_size != 0:
            st.info(
                "{nb} benchmark{s} found.".format(
                    nb=select_size, s="s" if select_size > 1 else ""
                )
            )
        show_all = st.checkbox("Show all runs of a same config", False)
        st.markdown("------")
        comp_param = st.selectbox(
            "The criteria for comparison ?",
            differences.keys()
        )

    # Sort the selection and bring together identical runs
    headers, conf_sizes, conf_list = _regroup_identicals(selection, show_all)

    # Process and fuse the results
    menu = st.sidebar.expander("Display Parameters :")
    results_processors = {
        "profile": ProfileProcessor(menu),
        "search": SearchProcessor(),
    }
    default_processor = DefaultProcessor(menu)

    for idx, config in enumerate(conf_list):
        for key, val_list in config["results"].items():
            processor = results_processors.get(key, default_processor)

            # Preprocess the results
            new_val_list = []
            for val in val_list:
                val = processor.verify(idx, key, val)
                new_val = processor.preprocess(val)
                new_val_list.append(new_val)

            # Fuse the results of each configuration
            fused_val = processor.fuse(new_val_list)
            config["results"][key] = fused_val

    # Extract results list
    res_list = list(map(lambda x: x.get("results"), conf_list))

    # Rearrange to regroup common results together
    report = _regroup_results(res_list)
    
    # Show headers info
    get_config_name = get_get_config_name(differences)
    configs_names = list(map(get_config_name, headers, range(1, len(headers)+1)))

    if len(headers) == 0:
        st.warning("0 benchmark found.")
    else:
        st.header("Selected configs")
    colors = plt.get_cmap("gnuplot")(np.linspace(0.1, 0.80, len(headers)))
    for i in range(len(headers)):
        with st.expander(configs_names[i]):
            st.info(f"Number of benchmarks comprised : {conf_sizes[i]}")
            _show_bench_info(headers[i], 1, colors[i])

    # Show the results
    if len(headers) != 0:
        st.header("Results")
        
    raw_data = {}
    for key, val_list in report.items():
        processor = results_processors.get(key, default_processor)
        display = processor.display(st, key, val_list, configs_names, colors)
        if not display:
            raw_data[key] = val_list
    if len(headers) != 0:
        default_processor.display_raw(st, raw_data, configs_names)
    
    if comp_param is not None:
        # Regroup for comparison
        comp_headers, indxd_res_list = _regroup_similar(conf_list, comp_param)
        
        # Order according to the comparison parameter
        param_values = differences[comp_param]
        fun = partial(_order_indxd_results, param_values)
        ordered_res_list = list(map(fun, indxd_res_list))

        # Rearrange to regroup common results together
        comp_report = _regroup_results(ordered_res_list)

        # Show the comparison
        st.header("Comparison")
        comp_names = list(map(get_config_name, comp_headers, range(1, len(headers)+1)))
        for key, val_list in comp_report.items():
            processor = results_processors.get(key, default_processor)
            display = processor.compare(st, key, val_list, comp_param, param_values, comp_names)
        


def _get_criterias(data, old_criterias, not_criterias):
    criterias = copy.deepcopy(old_criterias)
    for key, val in data.items():
        if key not in not_criterias:
            if type(val) == list:
                val = tuple(val)
            if key not in old_criterias:
                if type(val) == dict:
                    criterias[key] = {}
                else:
                    criterias[key] = {val}
            if type(val) == dict:
                criterias[key] = _get_criterias(data[key], criterias[key], not_criterias)
            else:
                criterias[key].add(val)
    return criterias


def _select_choices(criterias, depth):
    choices = {}
    differences = {}
    for key, val in criterias.items():
        if type(val) == dict:
            if depth == 1:
                st.markdown("------")
            h = depth + 1 if depth < 6 else 6
            st.markdown(f"{(h)*'#'} {key} :")
            choices[key], new_differences = _select_choices(criterias[key], depth + 1)
            differences = {**differences, **new_differences}
        else:
            val = list(val)
            val.sort()
            if len(val) == 1:
                choice = val
                st.markdown(f"{key} : {choice[0]}")
            else:
                choice = st.multiselect(label=key, options=val)
            if len(choice) == 0:
                choice = val
            if type(choice[0]) == tuple:
                choice = list(map(lambda x: list(x), choice))
            if len(choice) != 1:
                differences[key] = choice
            choices[key] = choice
    return choices, differences


def _generate_query(choices, query):
    sentence = query.noop()
    for key, val in choices.items():
        if type(val) == dict:
            test = _generate_query(choices[key], query[key])
        elif len(val) != 0:
            test = query[key].one_of(val)
        else:
            test = query.noop()
        sentence = (~(query[key].exists()) | test) & sentence
    return sentence


def _regroup_identicals(select, show_all):
    selection = copy.deepcopy(select)

    def _cleanup_info(info):
        info["summary"].pop("date")
        info["summary"].pop("description")
        try:
            info["parameters"].pop("random_state")
        except:
            pass

    headers = []
    conf_sizes = []
    conf_list = []
    for s in selection:
        if show_all:
            pass
        else:
            _cleanup_info(s)
        current_conf = s.copy()
        current_conf.pop("results")
        if current_conf in headers and not show_all:
            index = headers.index(current_conf)
            conf_sizes[index] += 1
            for key, val in s["results"].items():
                conf_list[index]["results"][key].append(val)
        else:
            headers.append(current_conf)
            conf_sizes.append(1)
            for key, val in s["results"].items():
                s["results"][key] = [val]
            conf_list.append(s)
    return headers, conf_sizes, conf_list


def _regroup_similar(select, comp_param):
    selection = copy.deepcopy(select)
    def _get_param_val(s, comp_param):
        param_val = None
        for key, val in s.items():
            if type(val) == dict:
                if param_val is None:
                    param_val = _get_param_val(s[key], comp_param)
            else:
                if key == comp_param:
                    param_val = s[key]
                    s[key] = {}
        return param_val

    comp_headers = []
    indxd_res_list = []
    for s in selection:
        param_val = _get_param_val(s, comp_param)
        if param_val is not None:
            results = s.pop("results")
            current_conf = s
            indxd_res = {
                "param_val": param_val,
                "results": results
            }
            if current_conf in comp_headers:
                index = comp_headers.index(current_conf)
                for key, val in indxd_res["results"].items():
                    indxd_res_list[index]["results"][key].append(val)
                indxd_res_list[index]["param_values"].append(param_val)
            else:
                comp_headers.append(current_conf)
                for key, val in indxd_res["results"].items():
                    indxd_res["results"][key] = [val]
                indxd_res["param_values"] = [param_val]
                indxd_res_list.append(indxd_res)
    return comp_headers, indxd_res_list


def _order_indxd_results(param_values, indexed_res):
    indxd_res = copy.deepcopy(indexed_res)
    param_index = indxd_res["param_values"]
    results = indxd_res["results"]
    for key, val_list in results.items():
        new_val_list = []
        for param_val in param_values:
            try:
                idx = param_index.index(param_val)
                new_val = val_list[idx]
            except:
                new_val = None
            new_val_list.append(new_val)
        results[key] = new_val_list
    return results


def get_get_config_name(differences):
    def _get_config_diff(config, differences):
        diff=[]
        for key, val in config.items():
            if type(val) != dict:
                if key in differences.keys():
                    if type(val) == str:
                        diff.append(val)
                    else:
                        diff.append(f"{key}: {val}")
            else:
                diff += _get_config_diff(config[key], differences)
        return diff

    def get_config_name(config, idx):
        diff = _get_config_diff(config, differences)
        name = " - ".join(diff) if len(diff) != 0 else f"config {idx}"
        return name
    
    return get_config_name


def _show_bench_info(configuration, depth, c):
    for key, val in configuration.items():
        h = depth + 2 if depth < 4 else 6
        if type(val) == dict:
            if depth == 1:
                st.markdown("------")
                color = f"rgb({c[0]*255}, {c[1]*255}, {c[2]*255})"
                st.markdown(
                    f"<h3 style='color:{color}'>{key} :</h3>", unsafe_allow_html=True
                )
            else:
                st.markdown(f"{h*'#'} {key}:")
            _show_bench_info(configuration[key], depth + 1, c)
        else:
            st.markdown(f"**{key}:** {val}")


def _regroup_results(res_list):
    report = {}
    for results in res_list:
        for key in report.keys():
            if key in results.keys():
                report[key].append(results[key])
            else:
                report[key].append(None)
        for key, val in results.items():
            if key not in report.keys():
                report[key] = [val]
    return report


def main():
    st.title("DeepHyper Dashboard")

    with st.sidebar.expander("Source Selection"):
        st.markdown("The file should be a `csv` or a `json`.")
        upload_type = st.radio(
            "Selection Type:", ("Upload a Local File", "Enter a File Path")
        )

        if upload_type == "Upload a Local File":
            uploaded_file = st.file_uploader("")
        else:
            path = st.text_input("Enter the file path")
            if path is not None and len(path) > 0:
                if not (os.path.exists(path)):
                    st.warning("File not found!")
                else:
                    uploaded_file = open(path, "r")
            else:
                uploaded_file = None

    if uploaded_file is not None:
        ext = uploaded_file.name.split(".")[-1]

        boards = {"csv": _files_selection, "json": _database_selection}

        def default(x):
            return st.sidebar.warning(
                f"File should either be a `csv` or a `json`, not '{ext}'."
            )

        boards.get(ext, default)(uploaded_file)

        uploaded_file.close()


if __name__ == "__main__":
    main()
