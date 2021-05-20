from dhproj.advanced_hpo.mapping import CLASSIFIERS
from deephyper.benchmark.datasets import airlines as dataset
from sklearn.utils import check_random_state

rs_clf = check_random_state(42)

rs_data = check_random_state(42)

ratio_test = 0.33
ratio_valid = (1 - ratio_test) * 0.33

train, valid, test = dataset.load_data(
    random_state=rs_data,
    test_size=ratio_test,
    valid_size=ratio_valid,
    categoricals_to_integers=True,
)

for clf_name, clf_class in CLASSIFIERS.items():
    print(clf_name)

    clf = clf_class(random_state=rs_clf)

    clf.fit(*train)

    acc_train = clf.score(*train)
    acc_valid = clf.score(*valid)
    acc_test = clf.score(*test)

    print(f"Accuracy on Training: {acc_train:.3f}")
    print(f"Accuracy on Validation: {acc_valid:.3f}")
    print(f"Accuracy on Testing: {acc_test:.3f}\n")
