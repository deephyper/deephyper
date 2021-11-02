import streamlit as st
import pandas as pd

from deephyper.core.analytics.dashboard._pyplot import plot_single_line


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
            st.write("Selected a profile CSV!")
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
