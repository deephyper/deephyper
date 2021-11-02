import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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
