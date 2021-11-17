import statistics as stat

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
from numpy.core.numeric import NaN


class ProfileProcessor():
    def __init__(self, menu: st.expander):
        self.warnings = []
        menu.markdown("**Evolution of Worker Use:**")
        self.roll_val = int(menu.slider(
            'Roll value (in s.)', .1, 25., 12.5)*10)
        self.menu = menu

    def verify(self, idx, key, val):
        if "n_jobs_running" not in val.keys() or "timestamp" not in val.keys():
            self.warnings.append(
                f"config {idx+1} : a run is missing 'n_jobs_running' or 'timestamp' in {key}.")
            val = None
        return val

    def preprocess(self, val):
        if val is not None:
            profile = pd.DataFrame(
                {"n_jobs_running": val["n_jobs_running"]}, index=val["timestamp"])
            profile.index -= profile.index[0]
            t0, t_max = float(profile.index[0]), float(profile.index[-1])
            t0, t_max = self.menu.slider(
                "Time Range", min_value=t0, max_value=t_max, value=(t0, t_max)
            )
            profile = profile[(profile.index >= t0) & (profile.index <= t_max)]
            new_base = np.arange(0, t_max, 0.1)
            profile = profile.reindex(profile.index.union(
                new_base)).interpolate('values').loc[new_base]
            profile = profile.rolling(self.roll_val).mean()
        else:
            profile = pd.DataFrame({"n_jobs_running": 0}, index=[0])
        return profile

    def fuse(self, df_list):
        df_concat = pd.concat(df_list)
        by_row_index = df_concat.groupby(df_concat.index)
        df_mean = by_row_index.mean()
        df_max = by_row_index.max()
        df_min = by_row_index.min()
        df_std = by_row_index.std()
        return (df_mean, df_max, df_min, df_std)

    def display(self, st, key, df_list, names, colors):
        st.subheader("Evolution of Worker Use:")
        for warning in self.warnings:
            st.warning(warning)
        fig = plt.figure()
        for i in range(len(df_list)):
            if df_list[i] is not None:
                df_mean, df_max, df_min, df_std = df_list[i]
                plt.plot(df_mean, label=names[i], color=colors[i])
                plt.fill_between(df_mean.index, df_min.n_jobs_running,
                                 df_max.n_jobs_running, alpha=.2, color=colors[i])
        plt.xlabel("Time (sec.)")
        plt.ylabel("Number of Used Workers")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        st.pyplot(fig)
        return True
    
    def compare(self, st, key, val_list, comp_param, param_values, names):
        pass

class SearchProcessor():
    def __init__(self):
        self.warnings = []

    def verify(self, idx, key, val):
        if "objective" not in val.keys():
            self.warnings.append(
                f"config {idx+1} : a run is missing 'objective' in {key}.")
            val = None
        return val

    def preprocess(self, val):
        def to_max(l):
            r = [l[0]]
            for e in l[1:]:
                r.append(max(r[-1], e))
            return r
        if val is not None:
            objective = val["objective"]
            start = 1 if objective[0] == -1 else 0
            search = pd.DataFrame({"objective": to_max(objective[start:])})
        else:
            search = pd.DataFrame()
        return search

    def fuse(self, df_list):
        df_concat = pd.concat(df_list)
        by_row_index = df_concat.groupby(df_concat.index)
        df_mean = by_row_index.mean()
        df_max = by_row_index.max()
        df_min = by_row_index.min()
        df_std = by_row_index.std()
        return (df_mean, df_max, df_min, df_std)

    def display(self, st, key, df_list, names, colors):
        st.subheader("Objective Evolution:")
        for warning in self.warnings:
            st.warning(warning)
        fig = plt.figure()
        for i in range(len(df_list)):
            if df_list[i] is not None:
                df_mean, df_max, df_min, df_std = df_list[i]
                plt.plot(df_mean, label=names[i], color=colors[i])
                plt.fill_between(df_mean.index, df_min.objective,
                                 df_max.objective, alpha=.2, color=colors[i])
        plt.xlabel("Iteration")
        plt.ylabel("Objective")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        st.pyplot(fig)
        return True

    def compare(self, st, key, val_list, comp_param, param_values, names):
        pass

class PercUtilProcessor():
    def __init__(self):
        self.warnings = []

    def verify(self, idx, key, val):
        if type(val) not in [int, float]:
            self.warnings.append(f"config {idx+1}: '{key}' is not numerical.")
            val = 0
        return val

    def preprocess(self, val):
        return val

    def fuse(self, val_list):
        avrg = stat.mean(val_list)
        try:
            std = stat.stdev(val_list)
        except:
            std = 0
        return (avrg, std)

    def display(self, st, key, val_list, names, colors):
        st.subheader("Worker Use:")
        for warning in self.warnings:
            st.warning(warning)
        fig = plt.figure()
        p = -1
        for i in reversed(range(len(val_list))):
            if val_list[i] is not None:
                p += 1
                avrg, std = val_list[i]
                avrg *= 100
                std *= 100
                err_color = colors[i].copy()
                color = colors[i]
                color[-1] = 0.7
                text_color = 'white' if color[:-1].sum() < 1.5 else '0.2'
                plt.barh(names[i], avrg, xerr=std,
                         color=color, ecolor=err_color)
                plt.barh(names[i], 100-avrg,
                         left=avrg, color='lightgrey')
                plt.text(avrg/2, p, f"{round(avrg, 2)}%Used",
                         ha='center', va='center', color=text_color)
                plt.text(avrg/2+50, p, f"{round(100-avrg, 2)}%Unused",
                         ha='center', va='center', color='0.2')

        plt.xlabel("Percentage")
        plt.tight_layout()
        st.pyplot(fig)
        return False

    def compare(self, st, key, val_list, comp_param, param_values, names):
        pass

class DefaultProcessor():

    def __init__(self, menu: st.expander):
        self.warnings = []
        menu.markdown("**Other Results:**")
        fuser_choice = menu.radio(
            "What value should be computed over the same config ?",
            ('mean', 'median', 'max', 'min'))
        fusers = {
            'mean': stat.mean,
            'median': stat.median,
            'max': max,
            'min': min
        }
        self.fuser = fusers.get(fuser_choice, stat.mean)

    def verify(self, idx, key, val):
        if type(val) not in [int, float]:
            self.warnings.append(f"config {idx+1}: '{key}' is not numerical.")
            val = NaN
        return val

    def preprocess(self, val):
        return val

    def fuse(self, val_list):
        avrg = self.fuser(val_list)
        return round(avrg, 4)

    def display(self, st, key, val_list, names, colors):
        return False

    def display_raw(self, st, raw_data, names):
        st.subheader("Other Results:")
        for warning in self.warnings:
            st.warning(warning)
        df = pd.DataFrame(raw_data, index=[
                          names[i] for i in range(len(list(raw_data.values())[0]))])
        st.dataframe(df)

    def compare(self, st, key, val_list, comp_param, param_values, names):
        st.subheader(f"{key}:")
        fig = plt.figure()
        for i in range(len(val_list)):
            if val_list[i] is not None:
                val = val_list[i]
                plt.plot(param_values, val, label=names[i], marker='o')
        plt.xlabel(comp_param)
        plt.legend()
        plt.grid()
        plt.tight_layout()
        st.pyplot(fig)
        return True