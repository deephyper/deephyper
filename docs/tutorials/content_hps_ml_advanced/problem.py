import ConfigSpace as cs
from deephyper.problem import HpProblem


Problem = HpProblem(seed=45)

#! Default value are very important when adding conditional and forbidden clauses
#! Otherwise the creation of the problem can fail if the default configuration is not
#! Acceptable
classifier = Problem.add_hyperparameter(
    name="classifier",
    value=["RandomForest", "GradientBoosting"],
    default_value="RandomForest",
)

# For both
Problem.add_hyperparameter(name="n_estimators", value=(1, 1000, "log-uniform"))
Problem.add_hyperparameter(name="max_depth", value=(1, 50))
Problem.add_hyperparameter(
    name="min_samples_split", value=(2, 10),
)
Problem.add_hyperparameter(name="min_samples_leaf", value=(1, 10))
criterion = Problem.add_hyperparameter(
    name="criterion",
    value=["friedman_mse", "mse", "mae", "gini", "entropy"],
    default_value="gini",
)

# GradientBoosting
loss = Problem.add_hyperparameter(name="loss", value=["deviance", "exponential"])
learning_rate = Problem.add_hyperparameter(name="learning_rate", value=(0.01, 1.0))
subsample = Problem.add_hyperparameter(name="subsample", value=(0.01, 1.0))

gradient_boosting_hp = [loss, learning_rate, subsample]
for hp_i in gradient_boosting_hp:
    Problem.add_condition(cs.EqualsCondition(hp_i, classifier, "GradientBoosting"))

forbidden_criterion_rf = cs.ForbiddenAndConjunction(
    cs.ForbiddenEqualsClause(classifier, "RandomForest"),
    cs.ForbiddenInClause(criterion, ["friedman_mse", "mse", "mae"]),
)
Problem.add_forbidden_clause(forbidden_criterion_rf)

forbidden_criterion_gb = cs.ForbiddenAndConjunction(
    cs.ForbiddenEqualsClause(classifier, "GradientBoosting"),
    cs.ForbiddenInClause(criterion, ["gini", "entropy"]),
)
Problem.add_forbidden_clause(forbidden_criterion_gb)

if __name__ == "__main__":
    print(Problem)
