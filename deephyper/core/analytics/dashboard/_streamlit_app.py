import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from tinydb import TinyDB, Query
import json
import tempfile
import numpy as np

from deephyper.core.analytics.dashboard._pyplot import plot_single_line


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


def _files_selection():
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file")

    if uploaded_file is not None:

        # Can be used wherever a "file-like" object is accepted:
        df = pd.read_csv(uploaded_file, index_col=0)

        # assuming it's a search csv
        if (
            "objective" in df.columns
            and "elapsed_sec" in df.columns
            and "duration" in df.columns
            and "id" in df.columns
        ):
            df["iteration"] = df.index

            st.header("Line Plot")
            st.sidebar.header("Line Plot")
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

            plot_single_line(df, line_plot_option_x, line_plot_option_y)

        # assuming it is a profile csv
        elif "timestamp" in df.columns and "n_jobs_running" in df.columns:
            st.header("Worker Utilization")
            num_workers = st.number_input(
                "Number of Workers", value=df.n_jobs_running.max()
            )

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

def _database_selection():
    uploaded_file = st.sidebar.file_uploader("Choose a Database file")

    if uploaded_file is not None:
        temp_db = tempfile.NamedTemporaryFile(mode="r+")
        json.dump(json.load(uploaded_file), temp_db)
        temp_db.read()
        db = TinyDB(temp_db.name)

        # Get the selectable caracteristics from the database
        selection = db.all()
        criterias = {}
        not_criterias = ["description", "results"]
        for select in selection:
            _get_criterias(select, criterias, not_criterias)

        # Show the possible criterias
        choices = {}
        with st.sidebar.expander("Select benchmark criterias :"):
            _select_choices(criterias, choices, 1)

        # Select the corresponding runs
        run = Query()
        run = _generate_query(choices, run)
        selection = db.search(run)
        select_size = len(selection)
        if select_size == 0:
            st.sidebar.warning("0 benchmark found.")
        else:
            st.sidebar.info("{nb} benchmark{s} found.".format(nb=select_size, s='s' if select_size > 1 else ''))
        
        # Sort selection and bring together identical runs
        configurations = []
        bench_per_conf = []
        results_list = []
        _regroup_identicals(selection, configurations, bench_per_conf, results_list)

        # Show configurations info
        st.header("Selected configurations")
        for i in range(len(configurations)):
            with st.expander(f"Configuration {i+1}"):
                st.info(f"Number of benchmarks comprised : {bench_per_conf[i]}")
                _show_bench_info(configurations[i], 1)

        # Preprocess the results
        def to_max(l):
            r = [l[0]]
            for e in l[1:]:
                r.append(max(r[-1], e))
            return r

        roll_val = st.sidebar.slider('Roll value', 1, 50, 25)
        for results in results_list:
            for key, val_list in results.items():
                if key in ["profile", "search"]:
                    new_val_list = []
                    for val in val_list:
                        if key == "profile":
                            profile = pd.DataFrame({"n_jobs_running" : val["n_jobs_running"]}, index=val["timestamp"])
                            profile.index -= profile.index[0]
                            new_base = np.arange(0, val["timestamp"][-1], 0.1)
                            profile = profile.reindex(profile.index.union(new_base)).interpolate('values').loc[new_base]
                            profile = profile.rolling(roll_val).mean()
                            new_val_list.append(profile.to_dict())
                        elif key == "search":
                            objective = val["objective"]
                            search = pd.DataFrame({"objective" : to_max(objective)})
                            new_val_list.append(search.to_dict())
                    results[key] = new_val_list

        # Mean the results of each configuration
        for results in results_list:
            for key, val_list in results.items():
                if key in ["profile", "search"]:
                    df_list = list(map(lambda x: pd.DataFrame(x), val_list))
                    df_concat = pd.concat(df_list)
                    by_row_index = df_concat.groupby(df_concat.index)
                    df = by_row_index.mean()
                    results[key] = df.to_dict()
                elif type(val_list[0]) in [int, float]:
                    avrg = sum(val_list)/len(val_list)
                    results[key] = avrg
                else:
                    results.pop(key)


        # Show the results
        st.header("Results")

        for i in range(len(configurations)):
            st.subheader(f"Configuration {i+1} :")
            for key, val in results_list[i].items():
                if key in ["profile", "search"]:
                    st.markdown(f"**{key} :**")
                    fig = plt.figure()
                    df = pd.DataFrame.from_dict(val)
                    plt.plot(df)
                    st.pyplot(fig)
                else:
                    st.markdown(f"**{key} :** {val}")


def _get_criterias(data, criterias, not_criterias):
    for key, val in data.items():
        if key not in not_criterias:
            if type(val) == list:
                val = tuple(val)
            if key not in criterias:
                if type(val) == dict:
                    criterias[key] = {}
                else:
                    criterias[key] = {val}
            if type(val) == dict :
                _get_criterias(data[key], criterias[key], not_criterias)
            else:
                criterias[key].add(val)

def _select_choices(criterias, choices, depth):
    for key, val in criterias.items():
        if type(val) == dict :
            if depth == 1:
                st.markdown("------")
            h = depth+1 if depth < 6 else 6
            st.markdown(f"{(h)*'#'} {key} :")
            choices[key] = {}
            _select_choices(criterias[key], choices[key], depth+1)
        else:
            choice = st.multiselect(
                label=key,
                options=val
            )
            if len(choice) != 0 and type(choice[0]) == tuple:
                choice = list(map(lambda x: list(x), choice))
            choices[key] = choice

def _generate_query(choices, query):
    sentence = query.noop()
    for key, val in choices.items():
        if type(val) == dict :
            test = _generate_query(choices[key], query[key])
        elif len(val) != 0:
            test = query[key].one_of(val)
        else:
            test = query.noop()
        sentence = (~ (query[key].exists()) | test) & sentence
    return sentence

def _show_bench_info(configuration, depth):
    for key, val in configuration.items():
        h = depth+2 if depth < 4 else 6
        if type(val) == dict :
            if depth == 1:
                st.markdown("------")
            st.markdown(f"{h*'#'} {key}:")
            _show_bench_info(configuration[key], depth+1)
        else:
            st.markdown(f"**{key}:** {val}")

def _regroup_identicals(selection, configurations, bench_per_conf, results_list):
    def _cleanup_info(info):
        #info["summary"].pop("date")
        return 0

    for s in selection:
        results = s.pop("results")
        _cleanup_info(s)
        try:
            index = configurations.index(s)
            bench_per_conf[index] += 1
            for key, val in results.items():
                results_list[index][key].append(val)
        except:
            configurations.append(s)
            bench_per_conf.append(1)
            for key, val in results.items():
                results[key] = [val]
            results_list.append(results)

def main():
    st.title("DeepHyper Dashboard")

    dashboard_type = st.sidebar.radio(
        "Where are your data stored?", ("Files", "Database")
    )
    if dashboard_type == "Files":
        _files_selection()
    else:
        _database_selection()


if __name__ == "__main__":
    main()
