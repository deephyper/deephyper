from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


class ProfileProcessor():
    def preprocess(val, roll_val):
        profile = pd.DataFrame({"n_jobs_running" : val["n_jobs_running"]}, index=val["timestamp"])
        profile.index -= profile.index[0]
        new_base = np.arange(0, val["timestamp"][-1], 0.1)
        profile = profile.reindex(profile.index.union(new_base)).interpolate('values').loc[new_base]
        profile = profile.rolling(roll_val).mean()
        return profile.to_dict()
    
    def fuse(val_list):
        df_list = list(map(lambda x: pd.DataFrame(x), val_list))
        df_concat = pd.concat(df_list)
        by_row_index = df_concat.groupby(df_concat.index)
        df = by_row_index.mean()
        return df.to_dict()
    
    def display(st, val_list):
        fig = plt.figure()
        for i in range(len(val_list)):
            if val_list[i] is not None:
                df = pd.DataFrame.from_dict(val_list[i])
            plt.plot(df)
        st.pyplot(fig)

class SearchProcessor():
    def preprocess(val):
        def to_max(l):
            r = [l[0]]
            for e in l[1:]:
                r.append(max(r[-1], e))
            return r

        objective = val["objective"]
        search = pd.DataFrame({"objective" : to_max(objective)})
        return search.to_dict()
    
    def fuse(val_list):
        df_list = list(map(lambda x: pd.DataFrame(x), val_list))
        df_concat = pd.concat(df_list)
        by_row_index = df_concat.groupby(df_concat.index)
        df = by_row_index.mean()
        return df.to_dict()
    
    def display(st, val_list):
        fig = plt.figure()
        for i in range(len(val_list)):
            if val_list[i] is not None:
                df = pd.DataFrame.from_dict(val_list[i])
            plt.plot(df)
        st.pyplot(fig)

class DefaultProcessor():
    def preprocess(val):
        return val

    def fuse(val_list):
        if type(val_list[0]) in [int, float]:
            avrg = sum(val_list)/len(val_list)
            return avrg
        else:
            return val_list[0]
    
    def display(st, val_list):
        for i in range(len(val_list)):
            if val_list[i] is not None:
                st.markdown(f"**Config {i+1} :** {val_list[i]}")