import abc
import os
import sys

import altair as alt
import pandas as pd
import streamlit as st
from deephyper.core.analytics import DBManager


from st_aggrid import AgGrid, GridOptionsBuilder, ColumnsAutoSizeMode


class View(abc.ABC):
    @abc.abstractmethod
    def show(self):
        ...


class Dashboard(View):
    def __init__(self, database_path="~/.deephyper/db.json") -> None:
        super().__init__()
        self.database_path = database_path

    def show(self):

        st.title("DeepHyper Dashboard")

        st.sidebar.header("Settings")

        source_selection = SourceSelection(self.database_path)
        with st.sidebar.expander("Experiments Source"):
            source_selection.show()

        if not (source_selection.dbm):
            return

        experiment_selection = ExperimentSelection(source_selection.dbm)
        experiment_selection.show()

        if len(experiment_selection.data) == 0:
            st.warning("No experiments selected!")
        else:
            if len(experiment_selection.data) == 1:
                # There is only 1 experiment to visualize
                charts = {
                    "Table": CSVView,
                    "Scatter": ScatterPlotView,
                    "Search Trajectory": SearchTrajectoryPlotView,
                    "Utilization": ProfilePlotView,
                }
                default_charts = ["Table", "Scatter"]
                exp = experiment_selection.data[0]
                exp["data"]["search"]["results"] = pd.DataFrame(
                    exp["data"]["search"]["results"]
                ).reset_index()
            else:
                # There are multiple experiments to compare
                charts = {
                    "Table": CSVView,
                    "Scatter": ScatterPlotView,
                    "Search Trajectory": SearchTrajectoryPlotView,
                    "Utilization": ProfilePlotView,
                }
                default_charts = ["Search Trajectory"]
                exp = experiment_selection.data
                for i in range(len(exp)):
                    exp[i]["data"]["search"]["results"] = pd.DataFrame(
                        exp[i]["data"]["search"]["results"]
                    ).reset_index()

            with st.sidebar.expander("Charts Selection"):
                selected_charts = st.multiselect(
                    "Charts", options=charts.keys(), default=default_charts
                )

            st.sidebar.markdown("""---""")
            st.sidebar.header("Options")

            for name in selected_charts:
                chart = charts[name]
                chart(exp).show()


class SourceSelection(View):
    """Select the source database."""

    def __init__(self, path) -> None:
        super().__init__()
        self.path = path
        self.dbm = None

    def show(self):
        st.markdown("The database should be a `json` file.")

        self.path = st.text_input("Enter the database path", value=self.path)
        if self.path is not None and len(self.path) > 0:
            if not (os.path.exists(self.path)):
                st.warning("File not found!")
            else:
                self.dbm = DBManager(path=self.path)


class ExperimentSelection(View):
    def __init__(self, dbm):
        self.dbm = dbm
        self.selection = {}
        self.data = []
        self.selection_mode = "single"

    def show(self):

        # main column
        st.header("Available Experiments")

        self.selection_mode = st.radio(
            "Selection Mode", ["single", "multiple"], index=0, horizontal=True
        )

        with st.spinner(text="In progress..."):

            data_summary = []
            for exp_data in self.dbm.list():

                data_summary.append(
                    {
                        "id": exp_data["id"],
                        "date_created": exp_data["metadata"]["add_date"],
                        "label": exp_data["metadata"]["label"],
                        "num_workers": exp_data["metadata"]["search"]["num_workers"],
                        "num_evaluations": len(
                            next(iter(exp_data["data"]["search"]["results"].values()))
                        ),
                    }
                )

            if len(data_summary) > 0:
                df = pd.DataFrame(data=data_summary)

                # Full example
                # https://github.com/PablocFonseca/streamlit-aggrid-examples/blob/main/main_example.py
                options_builder = GridOptionsBuilder.from_dataframe(df)
                options_builder.configure_selection(
                    self.selection_mode, use_checkbox=True
                )
                options_builder.configure_column(
                    "date_created",
                    type=["dateColumnFilter", "customDateTimeFormat"],
                    custom_format_string="yyyy-MM-dd HH:mm",
                    pivot=True,
                )
                options_builder.configure_column("label", editable=True)
                grid_options = options_builder.build()

                grid_response = AgGrid(
                    df,
                    grid_options,
                    theme="streamlit",
                    data_return_mode="filtered",
                    width="100%",
                    fit_columns_on_grid_load=False,
                    columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
                )

                df = grid_response["data"]
                self.data = [
                    self.dbm.get(exp_id=int(row["id"]))
                    for row in grid_response["selected_rows"]
                ]
                for exp_data, row in zip(self.data, grid_response["selected_rows"]):
                    exp_data["metadata"]["label"] = row["label"]


class DataView(View):
    """For experiments displaying data (e.g., plots and tables)"""

    def __init__(self, name, data):
        super().__init__()
        self.name = name
        self.data = data
        self.is_single = not (isinstance(data, list))


class CSVView(DataView):
    def __init__(self, data) -> None:
        super().__init__("Table", data)

    def show(self):
        st.header(self.name)

        if self.is_single:
            st.dataframe(self.data["data"]["search"]["results"])
        else:
            for i in range(min(len(self.data), 5)):
                with st.expander(
                    f"[{self.data[i]['id']}] {self.data[i]['metadata']['label']}"
                ):
                    st.dataframe(self.data[i]["data"]["search"]["results"])


class ScatterPlotView(DataView):
    def __init__(self, data):
        super().__init__("Scatter Plot", data)

    def show(self):

        if self.is_single:
            df = self.data["data"]["search"]["results"]
        else:
            df = []
            for i in range(len(self.data)):
                df_i = self.data[i]["data"]["search"]["results"]
                df_i["label"] = self.data[i]["metadata"]["label"]
                df.append(df_i)
            df = pd.concat(df, axis=0)

        columns = list(df.columns)

        # Options for the plot
        with st.sidebar.expander(self.name):
            if "timestamp_end" in columns:
                x_idx = columns.index("timestamp_end")
            elif "timestamp_gather" in columns:
                x_idx = columns.index("timestamp_gather")
            else:
                x_idx = columns.index("index")

            x_axis = st.selectbox(
                label="X-axis",
                options=columns,
                index=x_idx,
                key=f"{self.name}:x-axis:selectbox",
            )
            col1, col2 = st.columns(2)
            x_axis_max = col1.number_input(
                "X-axis Max",
                value=df[x_axis].max(),
                key=f"{self.name}:x-axis max:number_input",
            )
            x_axis_min = col2.number_input(
                "X-axis Min",
                value=df[x_axis].min(),
                key=f"{self.name}:x-axis min:number_input",
            )
            domain_x = [x_axis_min, x_axis_max]

            y_axis = st.selectbox(
                label="Y-axis",
                options=columns,
                index=columns.index("objective"),
                key=f"{self.name}:y-axis:selectbox",
            )
            col1, col2 = st.columns(2)
            y_axis_max = col1.number_input(
                "Y-axis Max",
                value=df[y_axis].max(),
                key=f"{self.name}:y-axis max:number_input",
            )
            y_axis_min = col2.number_input(
                "Y-axis Min",
                value=df[y_axis].min(),
                key=f"{self.name}:y-axis min:number_input",
            )
            domain_y = [y_axis_min, y_axis_max]

            color_var = st.selectbox(
                label="Color",
                options=columns,
                index=columns.index("objective")
                if self.is_single
                else columns.index("label"),
            )

        st.header(self.name)

        c = (
            alt.Chart(df)
            .mark_circle(size=60)
            .encode(
                x=alt.X(
                    x_axis,
                    title=x_axis.replace("_", " ").title(),
                    scale=alt.Scale(domain=domain_x),
                ),
                y=alt.Y(
                    y_axis,
                    title=y_axis.replace("_", " ").title(),
                    scale=alt.Scale(domain=domain_y),
                ),
                color=color_var,
                tooltip=columns,
            )
            .interactive()
        )

        st.altair_chart(c, use_container_width=True)


class SearchTrajectoryPlotView(DataView):
    def __init__(self, data):
        super().__init__("Search Trajectory", data)

    def show(self):

        if self.is_single:
            df = self.data["data"]["search"]["results"]
            columns = list(df.columns)

        else:
            df = []
            for i in range(len(self.data)):
                df_i = self.data[i]["data"]["search"]["results"]
                df_i["label"] = self.data[i]["metadata"]["label"]
                df.append(df_i)
            df = pd.concat(df, axis=0)

            columns = list(df.columns)
            columns.remove("label")

        # Options for the plot
        with st.sidebar.expander(self.name):
            if "timestamp_end" in columns:
                x_idx = columns.index("timestamp_end")
            elif "timestamp_gather" in columns:
                x_idx = columns.index("timestamp_gather")
            else:
                x_idx = columns.index("index")

            x_axis = st.selectbox(
                label="X-axis",
                options=columns,
                index=x_idx,
                key=f"{self.name}:x-axis:selectbox",
            )
            col1, col2 = st.columns(2)
            x_axis_max = col1.number_input(
                "X-axis Max",
                value=df[x_axis].max(),
                key=f"{self.name}:x-axis max:number_input",
            )
            x_axis_min = col2.number_input(
                "X-axis Min",
                value=df[x_axis].min(),
                key=f"{self.name}:x-axis min:number_input",
            )
            domain_x = [x_axis_min, x_axis_max]

            y_axis = "max objective"

            if self.is_single:
                df = df.sort_values(x_axis)
                df["max objective"] = df["objective"].cummax()
            else:
                df = df.sort_values(["label", x_axis])
                df[y_axis] = df.groupby(["label"])["objective"].transform("cummax")

            col1, col2 = st.columns(2)
            y_axis_max = col1.number_input(
                "Y-axis Max",
                value=df[y_axis].max(),
                key=f"{self.name}:y-axis max:number_input",
            )
            y_axis_min = col2.number_input(
                "Y-axis Min",
                value=df[y_axis].min(),
                key=f"{self.name}:y-axis min:number_input",
            )
            domain_y = [y_axis_min, y_axis_max]

        st.header(self.name)

        encode_kwargs = {}
        if not (self.is_single):
            encode_kwargs["color"] = "label"
        encode_kwargs["tooltip"] = "label"

        c = (
            alt.Chart(df)
            .mark_line(interpolate="basis")
            .encode(
                x=alt.X(
                    x_axis,
                    title=x_axis.replace("_", " ").title(),
                    scale=alt.Scale(domain=domain_x),
                ),
                y=alt.Y(
                    y_axis,
                    title=y_axis.replace("_", " ").title(),
                    scale=alt.Scale(domain=domain_y),
                ),
                **encode_kwargs,
            )
            .interactive()
        )

        st.altair_chart(c, use_container_width=True)


class ProfilePlotView(DataView):
    def __init__(self, data):
        super().__init__("Utilization", data)
        self.profile_type = "start/end"

    def get_profile(self, df, num_workers=0):
        """
        Args:
            df: data frame on which to compute the profile.
            num_workers: number of workers to use for normalization. if 0 then it is ignored.
        """
        # profile_type = "submit/gather"
        # profile_type = "start/end"

        if self.profile_type == "submit/gather":
            column_start = "timestamp_submit"
            column_end = "timestamp_gather"
        else:
            column_start = "timestamp_start"
            column_end = "timestamp_end"

        hist = []
        for _, row in df.iterrows():
            hist.append((row[column_start], 1))
            hist.append((row[column_end], -1))

        n_processes = 0
        profile_dict = dict(t=[0], n_processes=[0])
        for e in sorted(hist):
            t, incr = e
            n_processes += incr
            profile_dict["t"].append(t)
            profile_dict["n_processes"].append(n_processes)
        profile = pd.DataFrame(profile_dict)

        if num_workers > 0:
            profile["n_processes"] = profile["n_processes"] / num_workers

        return profile

    def get_perc_util(self, profile, num_workers):
        csum = 0
        for i in profile.index[:-1]:
            csum += (profile.loc[i + 1, "t"] - profile.loc[i, "t"]) * profile.loc[
                i, "n_processes"
            ]
        perc_util = csum / (profile["t"].iloc[-1] * num_workers)
        return perc_util

    def show(self):

        if self.is_single:
            df = self.data["data"]["search"]["results"]
            metadata = self.data["metadata"]
            columns = list(df.columns)
        else:
            # df = []
            # columns = None
            # for i in range(len(self.data)):
            #     df_i = self.data[i]["data"]["search"]["results"]
            #     label = self.data[i]["metadata"]["label"]
            #     df.append((label, df_i))

            #     if columns is None:
            #         columns = set(df_i.columns)
            #     else:
            #         columns = columns.intersection(set(df_i.columns))
            # columns = list(columns)
            df = []
            metadata = {}
            for i in range(len(self.data)):
                df_i = self.data[i]["data"]["search"]["results"]
                df_i["db:id"] = self.data[i]["id"]
                metadata[self.data[i]["id"]] = self.data[i]["metadata"]
                df.append(df_i)
            df = pd.concat(df, axis=0)
            columns = list(df.columns)

        # Options for the plot
        with st.sidebar.expander(self.name):

            normalize = st.checkbox("Normalize by number of workers")

            x_idx = None
            profile_types = []
            if "timestamp_end" in columns:
                x_idx = columns.index("timestamp_end")
                profile_types.append("start/end")
            if "timestamp_gather" in columns:
                if x_idx is None:
                    x_idx = columns.index("timestamp_gather")
                profile_types.append("submit/gather")
            if len(profile_types) == 0:
                st.warning(
                    "Nothing to display as no profiling information provided (e.g., timestamp_submit/gather, timestamp_start/end!)"
                )
                return

            self.profile_type = st.selectbox(
                label="Type of profile",
                options=profile_types,
                index=0,
                key=f"{self.name}:type of profile:selectbox",
            )

            col1, col2 = st.columns(2)
            x_axis = columns[x_idx]
            x_axis_max = col1.number_input(
                "X-axis Max",
                value=df[x_axis].max(),
                key=f"{self.name}:x-axis max:number_input",
            )
            x_axis_min = col2.number_input(
                "X-axis Min",
                value=df[x_axis].min(),
                key=f"{self.name}:x-axis min:number_input",
            )
            domain_x = [x_axis_min, x_axis_max]

            y_label = {
                "start/end": "# Jobs Running",
                "submit/gather": "# Jobs Pending",
            }[self.profile_type]

            if self.is_single:
                num_workers = metadata["search"]["num_workers"]

                if normalize:
                    profiles = self.get_profile(df, num_workers)
                else:
                    profiles = self.get_profile(df)

                profiles = profiles[
                    (x_axis_min <= profiles["t"]) & (profiles["t"] <= x_axis_max)
                ]

                utilization = self.get_perc_util(
                    profiles, 1 if normalize else num_workers
                )
            else:
                profiles = []
                utilization = []

                for i, (exp_id, df_group) in enumerate(df.groupby("db:id")):
                    num_workers = metadata[exp_id]["search"]["num_workers"]
                    label = metadata[exp_id]["label"]

                    if normalize:
                        profiles_i = self.get_profile(df_group, num_workers)
                    else:
                        profiles_i = self.get_profile(df_group)
                    profiles_i["label"] = metadata[exp_id]["label"]

                    profiles_i = profiles_i[
                        (x_axis_min <= profiles_i["t"])
                        & (profiles_i["t"] <= x_axis_max)
                    ]

                    utilization_i = self.get_perc_util(
                        profiles_i, 1 if normalize else num_workers
                    )

                    profiles.append(profiles_i)
                    utilization.append((label, utilization_i))

                profiles = pd.concat(profiles, axis=0)

        st.header(self.name)

        if self.is_single:
            st.markdown(
                f"{utilization*100:.2f}%"
                + " of utilization between $t_{min}="
                + f"${x_axis_min:.0f}"
                + " and $t_{max}"
                + f"=${x_axis_max:.0f}"
            )
        else:
            text = ""
            for label, util in utilization:
                text += (
                    f"* __{label}__: {util*100:.2f}%"
                    + " of utilization between $t_{min}="
                    + f"${x_axis_min:.0f}"
                    + " and $t_{max}"
                    + f"=${x_axis_max:.0f}\n"
                )
            st.markdown(text)

        encode_kwargs = {}
        if not (self.is_single):
            encode_kwargs["color"] = "label"

        c = (
            alt.Chart(profiles)
            .mark_line(interpolate="step-after")
            .encode(
                x=alt.X(
                    "t",
                    title="Time (sec.)",
                    scale=alt.Scale(domain=domain_x),
                ),
                y=alt.Y(
                    "n_processes",
                    title=y_label,
                    # scale=alt.Scale(domain=domain_y),
                ),
                **encode_kwargs,
            )
            .interactive()
        )

        st.altair_chart(c, use_container_width=True)


def main(database_path):

    st.set_page_config(layout="wide")
    dashboard = Dashboard(database_path)
    dashboard.show()


if __name__ == "__main__":
    database_path = sys.argv[1]
    main(database_path)
