import pandas as pd
import matplotlib.pyplot as plt

def quick_plot(fname):
    df = pd.read_csv(fname)
    plt.plot(df.objective)
    plt.show()