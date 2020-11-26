from pprint import pprint

import pandas as pd
from deephyper.benchmark.datasets import airlines as dataset
from deephyper.problem import filter_parameters
from dhproj.advanced_hpo.mapping import CLASSIFIERS
from sklearn.utils import check_random_state

df = pd.read_csv("results.csv")
config = df.iloc[df.objective.argmax()][:-2].to_dict()
print("Best config is:")
pprint(config)

config["random_state"] = check_random_state(42)

rs_data = check_random_state(42)

ratio_test = 0.33
ratio_valid = (1 - ratio_test) * 0.33

train, valid, test = dataset.load_data(
    random_state=rs_data, test_size=ratio_test, valid_size=ratio_valid,
)

clf_class = CLASSIFIERS[config["classifier"]]
config["n_jobs"] = 4
clf_params = filter_parameters(clf_class, config)

clf = clf_class(**clf_params)

clf.fit(*train)

acc_train = clf.score(*train)
acc_valid = clf.score(*valid)
acc_test = clf.score(*test)

print(f"Accuracy on Training: {acc_train:.3f}")
print(f"Accuracy on Validation: {acc_valid:.3f}")
print(f"Accuracy on Testing: {acc_test:.3f}")
