import abc
import copy
import json
import os
import statistics as stat
import tempfile
from functools import partial, reduce
from itertools import compress

import numpy as np
import pandas as pd
import streamlit as st
import tree
from matplotlib import pyplot as plt
from numpy.core.numeric import NaN
from tinydb import Query, TinyDB


class View(abc.ABC):
    @abc.abstractmethod
    def show(self):
        ...


class Dashboard(View):
    def show(self):

        st.title("DeepHyper Dashboard")

        source_selection = SourceSelection()
        with st.sidebar.expander("File Selection"):
            source_selection.show()

        if source_selection.uploaded_file is not None:
            file_extension = source_selection.uploaded_file.name.split(".")[-1]

            views = {"csv": CSVView, "json": JsonView}

            def default(_):
                return st.sidebar.warning(
                    f"File should either be a `csv` or a `json`, not '{file_extension}'."
                )

            views.get(file_extension, default)(source_selection.uploaded_file).show()

            source_selection.uploaded_file.close()


class SourceSelection(View):
    def __init__(self) -> None:
        super().__init__()
        self.upload_type = None
        self.uploaded_file = None

    def show(self):
        st.markdown("The file should be a `csv` or a `json`.")
        self.upload_type = st.radio(
            "Selection Type:", ("Upload a Local File", "Enter a File Path")
        )

        if self.upload_type == "Upload a Local File":
            self.uploaded_file = st.file_uploader("")
        else:
            path = st.text_input("Enter the file path")
            if path is not None and len(path) > 0:
                if not (os.path.exists(path)):
                    st.warning("File not found!")
                else:
                    self.uploaded_file = open(path, "r")
            else:
                self.uploaded_file = None


#! could be improved
class CSVView(View):
    ...


class JsonView(View):
    def __init__(self, file) -> None:
        super().__init__()

        # TODO: improve this code
        temp_file = tempfile.NamedTemporaryFile(mode="w+")
        json.dump(json.load(file), temp_file)
        temp_file.read()
        self.db = TinyDB(temp_file.name)

    def show(self):
        db_selection = DatabaseSelection(self.db)
        with st.sidebar.expander("Select benchmark criterias"):
            db_selection.show()

        views = {
            "Profile": ProfileView,
            "Search": SearchView,
            "PercUtil": PercUtilView,
            "Table": TableView,
            "Compare": ComparatorView,
        }
        view_selection = ViewSelection(db_selection.data, views)
        with st.sidebar.expander("Select views to display"):
            view_selection.show()
        for view in view_selection.selected_views:
            with st.container():
                view.show()


#! doesn't work if what is a leaf in a dict is not in another
def _merge_dict_in(synthesis, not_criterias, to_merge):
    def _add_entry(synthesis, not_criterias, path, val):
        path = list(path)
        if path:
            if path not in not_criterias:
                item = reduce(lambda d, key: d[key], path[:-1], synthesis)
                key = path[-1]
                if isinstance(val, dict):
                    if key not in item.keys():
                        item[key] = {}
                else:
                    if key not in item.keys():
                        item[key] = [val]
                    elif val not in item[key]:
                        item[key].append(val)
            else:
                return 0

    tree.traverse_with_path(partial(_add_entry, synthesis, not_criterias), to_merge)


# more secure version but lack the data organization the one above returns
def _merge_adresses_in(synthesis, not_criterias, to_merge):
    def _add_entry(synthesis, not_criterias, path, val):
        if list(path) not in not_criterias:
            if synthesis and path in tuple(zip(*synthesis))[0]:
                i = tuple(zip(*synthesis))[0].index(path)
                if val not in synthesis[i][1]:
                    synthesis[i][1].append(val)
            else:
                synthesis.append((path, [val]))

    tree.map_structure_with_path(
        partial(_add_entry, synthesis, not_criterias), to_merge
    )


class DatabaseSelection(View):
    def __init__(self, db) -> None:
        super().__init__()
        self.db = db
        self.data = []

    def _select_choices(self, criterias):
        def _add_choice(choices, path, val):
            path = list(path)
            if path:
                key = path[-1]
                if isinstance(val, dict):
                    if len(path) == 1:
                        st.markdown("------")
                    h = len(path) + 1 if len(path) < 6 else 6
                    st.markdown(f"{(h)*'#'} {key} :")
                elif isinstance(val, list):
                    val.sort()
                    if len(val) == 1:
                        default = val
                    else:
                        default = None
                    choice = st.multiselect(label=key, options=val, default=default)
                    choices.append((path, choice))
                else:
                    return 0

        choices = []
        tree.traverse_with_path(partial(_add_choice, choices), criterias)
        return choices

    def _generate_query(self, choices):
        query = Query().noop()
        for path, val in choices:
            item = reduce(lambda q, key: q[key], path, Query())
            if len(val) != 0:
                test = item.one_of(val)
            else:
                test = Query().noop()
            query = (~(item.exists()) | test) & query
        return query

    def show(self):
        # Get the selectable caracteristics from the database
        headers = list(map(lambda x: dict(x), self.db.all()))
        criterias = {}
        not_criterias = [
            ["summary", "description"],
            ["summary", "date"],
            ["parameters", "random_state"],
            ["results"],
        ]
        list(map(partial(_merge_dict_in, criterias, not_criterias), headers))

        # Show the possible criterias
        with st.container():
            choices = self._select_choices(criterias)

        # Select the corresponding runs
        query = self._generate_query(choices)
        self.data = list(map(lambda x: dict(x), self.db.search(query)))
        select_size = len(self.data)
        message = "{nb} benchmark{s} found.".format(
            nb=select_size, s="s" if select_size > 1 else ""
        )
        if select_size != 0:
            st.info(message)
        else:
            st.warning(message)


class ViewSelection(View):
    def __init__(self, data, views_dict) -> None:
        super().__init__()
        self.data = data
        self.views_dict = views_dict
        self.allowed_views = {}
        self.selected_views = []

    def _filter_allowed_views(self):
        for key, view_cls in self.views_dict.items():
            view = view_cls(self.data)
            if view.allowed:
                self.allowed_views[key] = view

    def show(self):
        self._filter_allowed_views()
        selected = st.multiselect("select", self.allowed_views.keys())
        for key in selected:
            self.selected_views.append(self.allowed_views[key])


supported_outputs = [
    "search",
    "profile",
    "init_time",
    "exec_time",
    "perc_util",
    "best_obj",
]


def _filter_results(data, to_filter):
    def _retrieve_results(filtered_results, path, val):
        if path:
            key = path[-1]
            if key in to_filter:
                filtered_results[key] = val
            if key in supported_outputs:
                return 0

    def _filter(run):
        filtered_results = {}
        tree.traverse_with_path(
            partial(_retrieve_results, filtered_results), run["results"]
        )
        run["results"] = filtered_results

    list(map(_filter, data))
    to_del = []
    for idx, run in enumerate(data):
        if not run["results"]:
            to_del.append(idx)
    for idx in reversed(to_del):
        del data[idx]
    return data


class GroupIdenticalResults(View):
    def __init__(self, data, key) -> None:
        self.data = copy.deepcopy(data)
        self.key = key

    def _sort_run(self, headers, results_list, data):
        header = data
        results = header.pop("results")
        header["summary"].pop("date")
        header["summary"].pop("description")
        if "random_state" in header["parameters"]:
            header["parameters"].pop("random_state")
        if header in headers:
            idx = headers.index(header)
            grouped_results = results_list[idx]
            for key in grouped_results.keys():
                grouped_results[key].append(results[key])
        else:
            grouped_results = {}
            for key, val in results.items():
                grouped_results[key] = [val]
            headers.append(header)
            results_list.append(grouped_results)

    def show(self):
        if st.checkbox(
            "Group identical runs together",
            True,
            key=f"Group identical runs together {self.key}",
        ):
            headers = []
            results_list = []
            list(map(partial(self._sort_run, headers, results_list), self.data))
            self.data = list(
                map(lambda h, r: {**h, "results": r}, headers, results_list)
            )
        else:
            for d in self.data:
                for key, val in d["results"].items():
                    d["results"][key] = [val]


class RunHeaderView(View):
    def __init__(self, header) -> None:
        self.header = header

    def _show_header_element(self, path, val):
        path = list(path)
        if path:
            key = path[-1]
            if isinstance(val, dict):
                if len(path) == 1:
                    st.markdown("------")
                h = len(path) + 1 if len(path) < 6 else 6
                st.markdown(f"{(h)*'#'} {key} :")
            else:
                st.markdown(f"**{key}:** {val}")

    def show(self):
        tree.traverse_with_path(partial(self._show_header_element), self.header)


def _get_diff(synthesis):
    diff = []
    tree.traverse_with_path(
        lambda path, val: diff.append(list(path)) and 0
        if isinstance(val, list) and len(val) > 1
        else None,
        synthesis,
    )
    return diff


def _get_names(headers, diff):
    names = []
    unnamed = 0
    for header in headers:
        name = []
        for path in diff:
            try:
                val = reduce(dict.get, path, header)
                if isinstance(val, str):
                    name.append(val)
                else:
                    name.append(f"{path[-1]}: {val}")
            except:
                pass
        if name:
            names.append(" - ".join(name))
        else:
            unnamed += 1
            names.append(f"config {unnamed}")
    return names


class ConfigurationsSelection(View):
    def __init__(self, data, key) -> None:
        self.key = key
        self.data = copy.deepcopy(data)
        self.headers = copy.deepcopy(data)
        list(map(lambda d: d.pop("results"), self.headers))
        synthesis = {}
        list(map(partial(_merge_dict_in, synthesis, []), self.headers))
        diff = _get_diff(synthesis)
        self.config_names = _get_names(self.headers, diff)

    def show(self):
        new_names = []
        to_keep = []
        with st.expander("Configurations list"):
            for name in self.config_names:
                new_name = st.text_input("", name, key=f"{name} {self.key}")
                new_names.append(new_name)
                to_keep.append(
                    st.checkbox(f"Show", True, key=f"Show {name} {self.key}")
                )
        self.data = list(compress(self.data, to_keep))
        self.config_names = list(compress(new_names, to_keep))
        self.headers = copy.deepcopy(self.data)
        list(map(lambda d: d.pop("results"), self.headers))
        for idx, header in enumerate(self.headers):
            header_view = RunHeaderView(header)
            with st.expander(self.config_names[idx]):
                header_view.show()


def _apply_checkup(checkup, v):
    idx, results = v
    for key, val in results.items():
        verif_result = list(map(partial(checkup, key, idx), val))
        results[key] = verif_result
    return results


def _apply_aggregation(preprocess, aggregate, results):
    for key, val in results.items():
        clean_result = list(map(preprocess, val))
        aggr_result = aggregate(clean_result)
        results[key] = aggr_result
    return results


def _regroup_results(res_list, displayable):
    def _add_res(store, v):
        i, results = v
        for key, value in results.items():
            store[key]["values"].append(value)
            store[key]["ids"].append(i)

    regrouped = dict.fromkeys(displayable)
    for key in regrouped.keys():
        regrouped[key] = {"values": [], "ids": []}
    list(map(partial(_add_res, regrouped), enumerate(res_list)))
    return regrouped


def _display_results(display, results, names, colors):
    for key, data in results.items():
        values = data["values"]
        ids = data["ids"]
        if values:
            display(key, values, ids, names, colors)


class AnalysisView(View):
    def __init__(self, data):
        self.data = []
        self.title = ""

    @property
    def allowed(self) -> bool:
        return bool(self.data)

    @abc.abstractmethod
    def _checkup(self, key, idx, val):
        ...

    def _show_menu(self):
        pass

    def _preprocess(self, val):
        return val

    @abc.abstractmethod
    def _aggregate(self, val_list):
        ...

    @abc.abstractmethod
    def _display(self, results, names, colors):
        ...

    def prepare(self, res_list):
        # data checkup
        res_list = list(
            map(partial(_apply_checkup, self._checkup), enumerate(res_list))
        )
        # show dispayers menus
        self._show_menu()
        # apply preprocessings & aggregation
        res_list = list(
            map(
                partial(_apply_aggregation, self._preprocess, self._aggregate), res_list
            )
        )
        # regroup everything together
        results = _regroup_results(res_list, self.supported_outputs)
        return results

    def show(self):
        st_display = st.container()
        st_display.header(self.title)
        st_configs = st.container()
        menu = st.sidebar.expander(self.title)
        group_identical_res = GroupIdenticalResults(self.data, self.title)
        with menu:
            group_identical_res.show()
            show_config_list = st.checkbox(
                "Show configurations list",
                False,
                key=f"Show configurations list {self.title}",
            )
        config_select = ConfigurationsSelection(group_identical_res.data, self.title)
        if show_config_list:
            with st_configs:
                config_select.show()
        res_list = list(
            map(lambda d: d.pop("results"), copy.deepcopy(config_select.data))
        )
        names = config_select.config_names
        colors = plt.get_cmap("gnuplot")(np.linspace(0.1, 0.80, len(names)))
        with menu:
            results = self.prepare(res_list)
        with st_display:
            self._display(results, names, colors)


class SingleGraphView(AnalysisView):
    @abc.abstractmethod
    def _plot(self, key, values, ids, names, colors):
        ...

    def _display(self, results, names, colors):
        _display_results(self._plot, results, names, colors)


class ProfileView(SingleGraphView):
    def __init__(self, data):
        self.title = "Profile"
        self.supported_outputs = ["profile"]
        self.data = _filter_results(copy.deepcopy(data), self.supported_outputs)
        self.warnings = []
        self.normalizable = True
        self.normalize = False
        self._duration = -1

    def _checkup(self, key, idx, val):
        if val.get("data"):
            temp = val["data"]
        else:
            temp = val
            self.normalizable = False
        if "n_jobs_running" not in temp.keys() or "timestamp" not in temp.keys():
            self._warnings.append(
                f"config {idx+1} : a run is missing 'n_jobs_running' or 'timestamp' in profile."
            )
            val = None
        else:
            duration = float(temp["timestamp"][-1] - temp["timestamp"][0])
            self._duration = max(duration, self._duration)
        return val

    def _show_menu(self):
        if self.normalizable:
            self.normalize = st.checkbox("Normalize the profiles", True)
        self._roll_val = st.slider(
            "Window size (in s.)",
            min_value=0,
            max_value=int(self._duration / 2),
            value=1,
        )
        self._t0, self._t_max = st.slider(
            "Time Range",
            min_value=float(0),
            max_value=self._duration,
            value=(float(0), self._duration),
        )

    def _old_preprocess(self, val):
        if val is not None:
            profile = pd.DataFrame(
                {"n_jobs_running": val["n_jobs_running"]}, index=val["timestamp"]
            )
            profile.index -= profile.index[0]
            profile = profile[
                (profile.index >= self._t0) & (profile.index <= self._t_max)
            ]
            new_base = np.arange(0, profile.index[-1], 0.1)
            profile = (
                profile.reindex(profile.index.union(new_base))
                .interpolate("values")
                .loc[new_base]
            )
            profile = profile.rolling(self._roll_val).mean()
        else:
            profile = pd.DataFrame({"n_jobs_running": [0]}, index=[0])
        return profile

    def _preprocess(self, val):
        num_workers = None
        if val is not None:
            if val.get("num_workers"):
                num_workers = val["num_workers"]
                temp = val["data"]
            else:
                temp = val
            profile = pd.DataFrame(
                {
                    "n_jobs_running": temp["n_jobs_running"],
                    "timestamp": temp["timestamp"],
                }
            )
            profile.timestamp -= profile.timestamp[0]
            profile = profile[
                (profile.timestamp >= self._t0) & (profile.timestamp <= self._t_max)
            ]
            if self.normalize:
                profile.n_jobs_running /= num_workers
        else:
            profile = pd.DataFrame({"n_jobs_running": [0]}, index=[0])
        return profile

    def _aggregate(self, df_list):
        times = np.unique(
            np.concatenate([df.timestamp.to_numpy() for df in df_list], axis=0)
        )
        times = np.concatenate([times, [self._t_max]])

        series = []
        for df in df_list:
            df = df.sort_values("timestamp")
            x, y = df.timestamp.to_numpy(), df.n_jobs_running.to_numpy()

            s = pd.Series(data=y, index=x)
            s = s.reindex(times).fillna(method="ffill")  # .fillna(method="bfill")
            s.index = pd.to_datetime(s.index, unit="s")
            if self._roll_val > 0:
                s = s.rolling(f"{self._roll_val}s", min_periods=1).mean()
            series.append(s)

        array = np.array([s.to_numpy() for s in series])
        loc = np.nanmean(array, axis=0)
        loc_max = np.nanmax(array, axis=0)
        loc_min = np.nanmin(array, axis=0)
        loc_std = np.nanstd(array, axis=0)

        return (times, loc, loc_max, loc_min, loc_std)

    def _old_plot(self, key, values, ids, names, colors):
        fig = plt.figure()
        for i, data in zip(ids, values):
            if data is not None:
                times, loc, loc_max, loc_min, loc_std = data
                plt.step(
                    times,
                    loc,
                    where="post",
                    label=names[i],
                    color=colors[i],
                )
                plt.fill_between(
                    times, loc_min, loc_max, step="post", alpha=0.3, color=colors[i]
                )
        plt.grid()
        plt.xlabel("Time (sec.)")
        if self.normalize:
            plt.ylabel("Percentage of Utilization")
        else:
            plt.ylabel("Number of Used Workers")
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05))
        plt.tight_layout()
        st.pyplot(fig)
    
    def _preprocess(self, val):
        if val is not None:
            profile = pd.DataFrame(
                {"n_jobs_running": val["n_jobs_running"], "timestamp": val["timestamp"]}
            )
            profile.timestamp -= profile.timestamp[0]
            profile = profile[
                (profile.timestamp >= self._t0) & (profile.timestamp <= self._t_max)
            ]
        else:
            profile = pd.DataFrame({"n_jobs_running": [0]}, index=[0])
        return profile
    
    def _aggregate(self, df_list):
        times = np.unique(
            np.concatenate([df.timestamp.to_numpy() for df in df_list],
                            axis=0))
        times = np.concatenate([[self._t0], times, [self._t_max]])

        series = []
        for df in df_list:
            df = df.sort_values("timestamp")
            x, y = df.timestamp.to_numpy(
            ), df.n_jobs_running.to_numpy() 

            s = pd.Series(data=y, index=x)
            s = s.reindex(times).fillna(method="ffill").fillna(method="bfill")
            series.append(s)

        array = np.array([s.to_numpy() for s in series])
        loc = np.nanmean(array, axis=0)
        loc_max = np.nanmax(array, axis=0)
        loc_min = np.nanmin(array, axis=0)
        loc_std = np.nanstd(array, axis=0)

        return (times, loc, loc_max, loc_min, loc_std)
    
    def _plot(self, key, values, ids, names, colors):
        fig = plt.figure()
        for i, data in zip(ids, values):
            if data is not None:
                times, loc, loc_max, loc_min, loc_std = data
                plt.step(times, loc,
                    where="post",
                    label=names[i],
                    color=colors[i],
                )
                plt.fill_between(
                    times,
                    loc_min,
                    loc_max,
                    step="post",
                    alpha=0.3,
                    color=colors[i]
                )
        plt.grid()
        plt.xlabel("Time (sec.)")
        plt.ylabel("Number of Used Workers")
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig)


class SearchView(SingleGraphView):
    def __init__(self, data):
        self.title = "Search"
        self.supported_outputs = ["search"]
        self.data = _filter_results(copy.deepcopy(data), self.supported_outputs)
        self.warnings = []
        self._iterations = -1
        self._obj_min, self._obj_max = float("inf"), float("-inf")

    def _checkup(self, key, idx, val):
        if "objective" not in val.keys():
            self.warnings.append(
                f"config {idx+1} : a run is missing 'objective' in search."
            )
            val = None
        else:
            iterations = len(val["objective"])
            obj_min, obj_max = min(val["objective"]), max(val["objective"])
            self._iterations = max(iterations, self._iterations)
            self._obj_min, self._obj_max = min(obj_min, self._obj_min), max(
                obj_max, self._obj_max
            )
        return val

    def _show_menu(self):
        self._obj_min, self._obj_max = st.slider(
            "Objective Range",
            min_value=self._obj_min,
            max_value=self._obj_max,
            value=(self._obj_min, self._obj_max),
        )
        self._it_min, self._it_max = st.slider(
            "Iteration Range",
            min_value=0,
            max_value=self._iterations,
            value=(0, self._iterations),
        )

    def _preprocess(self, val):
        def to_max(l):
            r = [l[0]]
            for e in l[1:]:
                r.append(max(r[-1], e))
            return r

        if val is not None:
            objective = val["objective"]
            search = pd.DataFrame({"objective": to_max(objective)})
            search = search[
                (search.index >= self._it_min) & (search.index <= self._it_max)
            ]
        else:
            search = pd.DataFrame()
        return search

    def _aggregate(self, val_list):
        df_concat = pd.concat(val_list)
        by_row_index = df_concat.groupby(df_concat.index)
        df_mean = by_row_index.mean()
        df_max = by_row_index.max()
        df_min = by_row_index.min()
        df_std = by_row_index.std()
        return (df_mean, df_max, df_min, df_std)

    def _plot(self, key, values, ids, names, colors):
        fig = plt.figure()
        for i, df in zip(ids, values):
            if df is not None:
                df_mean, df_max, df_min, df_std = df
                plt.plot(df_mean, label=names[i], color=colors[i])
                plt.fill_between(
                    df_mean.index,
                    df_min.objective,
                    df_max.objective,
                    alpha=0.2,
                    color=colors[i],
                )
        plt.xlabel("Iteration")
        plt.ylabel("Objective")
        plt.grid()
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05))
        plt.tight_layout()
        st.pyplot(fig)


class PercUtilView(SingleGraphView):
    def __init__(self, data):
        self.title = "PercUtil"
        self.supported_outputs = ["perc_util"]
        self.data = _filter_results(copy.deepcopy(data), self.supported_outputs)
        self.warnings = []

    def _checkup(self, key, idx, val):
        self._called = True
        if type(val) not in [int, float]:
            self.warnings.append(f"config {idx+1}: 'perc_util' is not numerical.")
            val = 0
        return val

    def _aggregate(self, val_list):
        avrg = stat.mean(val_list)
        try:
            std = stat.stdev(val_list)
        except:
            std = 0
        return (avrg, std)

    def _plot(self, key, values, ids, names, colors):
        fig = plt.figure()
        for i, val in zip(ids, values):
            avrg, std = val
            avrg *= 100
            std *= 100
            err_color = colors[i].copy()
            color = colors[i]
            color[-1] = 0.7
            text_color = "white" if color[:-1].sum() < 1.5 else "0.2"
            plt.barh(names[i], avrg, xerr=std, color=color, ecolor=err_color)
            plt.barh(names[i], 100 - avrg, left=avrg, color="lightgrey")
            plt.text(
                avrg / 2,
                i,
                f"{round(avrg, 2)}%\nUsed",
                ha="center",
                va="center",
                color=text_color,
            )
            plt.text(
                avrg / 2 + 50,
                i,
                f"{round(100-avrg, 2)}%\nUnused",
                ha="center",
                va="center",
                color="0.2",
            )
        plt.xlabel("Percentage")
        plt.tight_layout()
        st.pyplot(fig)


class TableView(AnalysisView):
    def __init__(self, data):
        self.title = "Table"
        self.supported_outputs = ["best_obj", "init_time", "exec_time", "perc_util"]
        self.data = _filter_results(copy.deepcopy(data), self.supported_outputs)
        self.warnings = []

    def _checkup(self, key, idx, val):
        self._called = True
        if type(val) not in [int, float]:
            self.warnings.append(f"config {idx+1}: '{key}' is not numerical.")
            val = 0
        return val

    def _show_menu(self):
        aggregators = {
            "mean": stat.mean,
            "median": stat.median,
            "std": stat.stdev,
            "max": max,
            "min": min,
        }
        aggregator_choice = st.radio(
            "What value should be computed over the same config ?",
            aggregators.keys(),
            key=f"What value should be computed over the same config ? Table",
        )
        self.aggregator = aggregators.get(aggregator_choice, stat.mean)

    def _aggregate(self, val_list):
        if self.aggregator == stat.stdev:
            try:
                avrg = self.aggregator(val_list)
            except:
                avrg = 0
        else:
            avrg = self.aggregator(val_list)
        return round(avrg, 2)

    def _display(self, results, names, colors):
        df = pd.DataFrame(results, index=names)
        for key, data in results.items():
            values = data["values"]
            ids = data["ids"]
            if values:
                df[key] = NaN
                for i, val in zip(ids, values):
                    df[key][i] = val
        st.dataframe(df)


def _keys_in_nested_dict(keys, nested_dict):
    def _nested_dict_of_key(d, key):
        if key in d.keys():
            return d[key]

    all_correct = True
    for key in keys:
        all_correct = all_correct and (
            reduce(_nested_dict_of_key, key, nested_dict) is not None
        )
    return all_correct


class ComparatorView(AnalysisView):
    def __init__(self, data):
        self.title = "Compare"
        self.supported_outputs = ["best_obj", "init_time", "exec_time", "perc_util"]
        self.data = _filter_results(copy.deepcopy(data), self.supported_outputs)
        self.warnings = []

    @property
    def allowed(self) -> bool:
        synthesis = {}
        headers = copy.deepcopy(self.data)
        list(map(lambda d: d.pop("results"), headers))
        list(map(partial(_merge_dict_in, synthesis, []), headers))
        diff = _get_diff(synthesis)
        return self.data and diff

    def _choose_param(self, diff):
        choices = list(map(lambda d: "/".join(d), diff))
        choice = st.selectbox("Choose the parameter on which compare the runs", choices)
        idx = choices.index(choice)
        return diff[idx]

    def _sort_run(self, param_path, headers, results_list, data):
        header = data
        results = header.pop("results")
        item = reduce(dict.get, param_path[:-1], header)
        param_val = item.pop(param_path[-1])
        if header in headers:
            idx = headers.index(header)
            grouped_results = results_list[idx]
            for key in grouped_results.keys():
                grouped_results[key]["values"].append(results[key])
                grouped_results[key]["param_ids"].append(param_val)
        else:
            grouped_results = {}
            for key, val in results.items():
                grouped_results[key] = {"values": [val], "param_ids": [param_val]}
            headers.append(header)
            results_list.append(grouped_results)

    def _regroup_for_comparison(self, res_list, displayable):
        def _add_res(store, v):
            i, results = v
            for key, data in results.items():
                store[key]["values"].append(data["values"])
                store[key]["param_ids"].append(data["param_ids"])
                store[key]["ids"].append(i)

        regrouped = dict.fromkeys(displayable)
        for key in regrouped.keys():
            regrouped[key] = {"values": [], "param_ids": [], "ids": []}
        list(map(partial(_add_res, regrouped), enumerate(res_list)))
        return regrouped

    def _checkup(self, key, idx, val):
        return val

    def _show_menu(self):
        pass

    def _aggregate(self, val_list):
        return stat.mean(val_list)

    def _display_results(self, results, names, colors):
        for key, data in results.items():
            values = data["values"]
            param_values = data["param_ids"]
            ids = data["ids"]
            if values:
                self._display(key, values, param_values, ids, names, colors)

    def _display(self, key, values, param_values, ids, names, colors):
        fig = plt.figure()
        for idx in ids:
            x = param_values[idx]
            y = values[idx]
            x, y = zip(*sorted(zip(x, y)))
            plt.plot(
                x,
                y,
                label=names[idx],
                color=colors[idx],
                marker="o",
            )
        plt.xlabel(self.param_name)
        plt.ylabel(key)
        plt.grid()
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig)

    def show(self):
        st_display = st.container()
        st_display.header(self.title)
        st_configs = st.container()
        menu = st.sidebar.expander(self.title)
        # group identicals
        group_identical_res = GroupIdenticalResults(self.data, self.title)
        with menu:
            group_identical_res.show()
        # get the differences on which obtain the parameter
        synthesis = {}
        headers = group_identical_res.data
        res_list = list(map(lambda d: d.pop("results"), headers))
        list(map(partial(_merge_dict_in, synthesis, []), headers))
        diff = _get_diff(synthesis)
        # choose a param on which do the comparison
        with menu:
            comp_param = self._choose_param(diff)
            self.param_name = "/".join(comp_param)
        # filter those who have it
        to_keep = list(
            map(partial(_keys_in_nested_dict, [comp_param]), group_identical_res.data)
        )
        headers = list(compress(headers, to_keep))
        res_list = list(compress(res_list, to_keep))
        # preprocess everything
        res_list = list(
            map(partial(_apply_checkup, self._checkup), enumerate(res_list))
        )
        # show dispayers menus
        self._show_menu()
        # apply preprocessings & aggregation
        res_list = list(
            map(
                partial(_apply_aggregation, self._preprocess, self._aggregate), res_list
            )
        )
        # fuse according to parameter
        self.data = list(map(lambda h, r: {**h, "results": r}, headers, res_list))
        headers = []
        res_list = []
        list(map(partial(self._sort_run, comp_param, headers, res_list), self.data))
        self.data = list(map(lambda h, r: {**h, "results": r}, headers, res_list))
        # select_config_view
        with menu:
            show_config_list = st.checkbox(
                "Show configurations list",
                False,
                key=f"Show configurations list {self.title}",
            )
        config_select = ConfigurationsSelection(self.data, self.title)
        if show_config_list:
            with st_configs:
                config_select.show()
        res_list = list(
            map(lambda d: d.pop("results"), copy.deepcopy(config_select.data))
        )
        names = config_select.config_names
        colors = plt.get_cmap("gnuplot")(np.linspace(0.1, 0.80, len(names)))
        # fuse the remaining
        results = self._regroup_for_comparison(res_list, self.supported_outputs)
        # display
        with st_display:
            self._display_results(results, names, colors)


def main():

    dashboard = Dashboard()
    dashboard.show()


if __name__ == "__main__":
    main()
