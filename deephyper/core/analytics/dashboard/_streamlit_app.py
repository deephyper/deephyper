import os


import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd
import streamlit as st
from deephyper.core.analytics.dashboard._pyplot import (
    plot_single_line,
    plot_single_line_improvement,
)

from deephyper.core.analytics.dashboard._views import Graphs, Table


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
        has_failed = (abs(df[line_plot_option_y] - min_float) < 1e-3) | (
            df[line_plot_option_y] < min_float
        )
        n_failures = sum(has_failed.astype(int))

        if n_failures > 0:
            st.warning(
                f"**{n_failures}** failure{'s' if n_failures > 1 else ''} detected!"
            )

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

        k = st.sidebar.number_input(
            "Number of displayed headers: ", min_value=1, max_value=len(df), value=5
        )
        df = df.sort_values(by=["objective"], ascending=False, ignore_index=True)

        subdf = df.iloc[:k]
        st.dataframe(subdf)

    # assuming it is a profile csv
    elif "timestamp" in df.columns and "n_jobs_running" in df.columns:
        df.set_index(df.columns[0])

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


def main():
    st.title("DeepHyper Dashboard")

    with st.sidebar.expander("Source Selection"):
        st.markdown("The file should be a `csv`.")
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

        boards = {"csv": _files_selection}

        def default(x):
            return st.sidebar.warning(f"File should be a `csv`, not '{ext}'.")

        boards.get(ext, default)(uploaded_file)

        uploaded_file.close()


if __name__ == "__main__":
    main()
