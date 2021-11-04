import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from tinydb import TinyDB, Query
import json

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
        data = json.load(uploaded_file)
        with open("db.json", 'w') as outfile:
            json.dump(data, outfile)
        db = TinyDB("db.json")

        # Récup les caractéristiques de la db que l’on peut sélectionner
        selection = db.all()
        criterias = {}
        not_criterias = ["description", "results"]
        for select in selection:
            _get_criterias(select, criterias, not_criterias)

        # Afficher les choix possibles
        choices = {}
        with st.sidebar.expander("Select benchmark criterias :"):
            _select_criterias(criterias, choices)

        # Faire la sélection
        run = Query()
        run = _test_criterias(choices, run)
        selection = db.search(run)
        
        # Print les caractéristiques de la selection
        # Moyenner les valeurs de la sélection
        # Moyenner les courbes de la sélection
        # Afficher les valeurs
        # Afficher les courbes

        st.header("Benchmark selection")
        # Headers
        columns = st.columns(len(selection))
        for i in range(len(selection)):
            with columns[i]:
                st.subheader(f"Benchmark {i+1}")
                bench = selection[i]
                for part, elements in bench.items():
                    if part != "results":
                        st.markdown(f"- **{part} :**")
                        for key, value in elements.items():
                            if key != "stable":
                                st.markdown(f"*{key}* : {value}")

        st.header("Results")

        # results
        for result in selection[0]["results"]:
            if type(selection[0]["results"][result]) != dict:
                st.markdown(f"**{result} :**")
                for i in range(len(selection)):
                    value = selection[i]["results"][result]
                    st.markdown(f"**- benchmark {i+1}** : {value}")

        # 1st figure: number of jobs vs time
        fig = plt.figure()
        for i in range(len(selection)):
            profile = pd.DataFrame.from_dict(selection[i]["results"]["profile"])
            plt.step(profile.timestamp, profile.n_jobs_running, where="post", label=f"benchmark {i+1}")
        plt.legend()
        plt.title("profile")
        st.pyplot(fig)
        plt.close()

        # 2nd figure: objective vs iter
        def to_max(l):
            r = [l[0]]
            for e in l[1:]:
                r.append(max(r[-1], e))
            return r

        fig = plt.figure()
        for i in range(len(selection)):
            plt.plot(to_max(pd.DataFrame.from_dict(selection[i]["results"]["search"]).objective), label=f"benchmark {i+1}")
        plt.legend()
        plt.title("search")
        st.pyplot(fig)
        plt.close()

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

def _select_criterias(criterias, choices):
    for key, val in criterias.items():
        if type(val) == dict :
            st.markdown("------")
            st.write(f"{key} :")
            choices[key] = {}
            _select_criterias(criterias[key], choices[key])
        else:
            choice = st.multiselect(
                label=key,
                options=val
            )
            if len(choice) != 0 and type(choice[0]) == tuple:
                choice = list(map(lambda x: list(x), choice))
            choices[key] = choice

def _test_criterias(choices, query):
    sentence = query.noop()
    for key, val in choices.items():
        if type(val) == dict :
            test = _test_criterias(choices[key], query[key])
        elif len(val) != 0:
            test = query[key].one_of(val)
        else:
            test = query.noop()
        sentence = (~ (query[key].exists()) | test) & sentence
    return sentence

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
