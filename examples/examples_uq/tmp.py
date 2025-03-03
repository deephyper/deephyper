exit()

# %%
# Analysis of the results
# -----------------------

results

# %%
# Evolution of the objective
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
from deephyper.analysis.hpo import plot_search_trajectory_single_objective_hpo


_ = plt.figure(figsize=(WIDTH_PLOTS, HEIGHT_PLOTS))
_ = plot_search_trajectory_single_objective_hpo(results)

# %%
# Worker utilization
# ~~~~~~~~~~~~~~~~~~
from deephyper.analysis.hpo import plot_worker_utilization


_ = plt.figure(figsize=(WIDTH_PLOTS, HEIGHT_PLOTS))
_ = plot_worker_utilization(results)

# %% 
# The best tecision tree
# ~~~~~~~~~~~~~~~~~~~~~~

from deephyper.analysis.hpo import parameters_from_row

topk_rows = results.nlargest(5, "objective").reset_index(drop=True)

for i, row in topk_rows.iterrows():
    parameters = parameters_from_row(row)
    obj = row["objective"]
    print(f"Top-{i+1} -> {obj=:.3f}: {parameters}")
    print()

best_job = topk_rows.iloc[0]

hpo_dir = "hpo_sklearn_classification"
model_checkpoint_dir = os.path.join(hpo_dir, "models")

with open(os.path.join(model_checkpoint_dir, f"model_0.{best_job.job_id}.pkl"), "rb") as f:
    best_model = pickle.load(f)

# %%
_ = plt.figure(figsize=(WIDTH_PLOTS, HEIGHT_PLOTS))
plot_decision_boundary_decision_tree(
    tx, ty, best_model, steps=1000, color_map="viridis"
)

# %%
_ = plt.figure(figsize=(WIDTH_PLOTS, HEIGHT_PLOTS))
disp = CalibrationDisplay.from_predictions(ty, best_model.predict_proba(tx)[:, 1])

# %% 
# Ensemble of decision trees
# --------------------------

from deephyper.ensemble import EnsemblePredictor
from deephyper.ensemble.aggregator import MixedCategoricalAggregator
from deephyper.ensemble.loss import CategoricalCrossEntropy 
from deephyper.ensemble.selector import GreedySelector, TopKSelector
from deephyper.predictor.sklearn import SklearnPredictorFileLoader

# %%
def plot_decision_boundary_and_uncertainty(
    dataset, labels, model, steps=1000, color_map="viridis", s=5
):

    fig, axs = plt.subplots(
        3, sharex="all", sharey="all", figsize=(WIDTH_PLOTS, HEIGHT_PLOTS * 2)
    )

    # Define region of interest by data limits
    xmin, xmax = dataset[:, 0].min() - 1, dataset[:, 0].max() + 1
    ymin, ymax = dataset[:, 1].min() - 1, dataset[:, 1].max() + 1
    x_span = np.linspace(xmin, xmax, steps)
    y_span = np.linspace(ymin, ymax, steps)
    xx, yy = np.meshgrid(x_span, y_span)

    # Make predictions across region of interest
    y_pred = model.predict(np.c_[xx.ravel(), yy.ravel()].astype(np.float32))
    y_pred_proba = y_pred["loc"]
    y_pred_aleatoric = y_pred["uncertainty_aleatoric"]
    y_pred_epistemic = y_pred["uncertainty_epistemic"]

    # Plot decision boundary in region of interest

    # 1. MODE
    color_map = plt.get_cmap("viridis")
    z = y_pred_proba[:, 1].reshape(xx.shape)

    cont = axs[0].contourf(xx, yy, z, cmap=color_map, vmin=0, vmax=1, alpha=0.5)

    # Get predicted labels on training data and plot
    axs[0].scatter(
        dataset[:, 0],
        dataset[:, 1],
        c=labels,
        cmap=color_map,
        s=s,
        lw=0,
    )
    plt.colorbar(cont, ax=axs[0], label="Probability of class 1")

    # 2. ALEATORIC
    color_map = plt.get_cmap("plasma")
    z = y_pred_aleatoric.reshape(xx.shape)

    cont = axs[1].contourf(xx, yy, z, cmap=color_map, vmin=0, vmax=0.69, alpha=0.5)

    # Get predicted labels on training data and plot
    axs[1].scatter(
        dataset[:, 0],
        dataset[:, 1],
        c=labels,
        cmap=color_map,
        s=s,
        lw=0,
    )
    plt.colorbar(cont, ax=axs[1], label="Aleatoric uncertainty")

    # 3. EPISTEMIC
    z = y_pred_epistemic.reshape(xx.shape)

    cont = axs[2].contourf(xx, yy, z, cmap=color_map, vmin=0, vmax=0.69, alpha=0.5)

    # Get predicted labels on training data and plot
    axs[2].scatter(
        dataset[:, 0],
        dataset[:, 1],
        c=labels,
        cmap=color_map,
        s=s,
        lw=0,
    )
    plt.colorbar(cont, ax=axs[2], label="Epistemic uncertainty")

    plt.show()

# %%
def create_ensemble_from_checkpoints(ensemble_selector: str = "topk"):

    # 0. Load data
    _, (vx, vy) = load_data_train_valid(verbose=0)

    # !1.3 SKLEARN EXAMPLE
    predictor_files = SklearnPredictorFileLoader.find_predictor_files(
        model_checkpoint_dir
    )
    predictor_loaders = [SklearnPredictorFileLoader(f) for f in predictor_files]
    predictors = [p.load() for p in predictor_loaders]

    # 2. Build an ensemble
    ensemble = EnsemblePredictor(
        predictors=predictors,
        aggregator=MixedCategoricalAggregator(
            uncertainty_method="entropy", decomposed_uncertainty=True
        ),
        # ! You can specify parallel backends for the evaluation of the ensemble
        evaluator={
            "method": "process",
            "method_kwargs": {"num_workers": 8},
        },
    )
    y_predictors = ensemble.predictions_from_predictors(
        vx, predictors=ensemble.predictors
    )

    # Use TopK or Greedy/Caruana
    if ensemble_selector == "topk":
        selector = TopKSelector(
            loss_func=CategoricalCrossEntropy(),
            k=20,
        )
    elif ensemble_selector == "greedy":
        selector = GreedySelector(
            loss_func=CategoricalCrossEntropy(),
            aggregator=MixedCategoricalAggregator(),
            k=20,
            k_init=5,
            eps_tol=1e-5,
        )
    else:
        raise ValueError(f"Unknown ensemble_selector: {ensemble_selector}")

    selected_predictors_indexes, selected_predictors_weights = selector.select(
        vy, y_predictors
    )
    print(f"{selected_predictors_indexes=}")
    print(f"{selected_predictors_weights=}")

    ensemble.predictors = [ensemble.predictors[i] for i in selected_predictors_indexes]
    ensemble.weights = selected_predictors_weights

    return ensemble

# %%
ensemble = create_ensemble_from_checkpoints("topk")

# %%
plot_decision_boundary_and_uncertainty(tx, ty, ensemble, steps=1000, color_map="viridis")

ty_pred = ensemble.predict(tx)["loc"]

cce = log_loss(ty, ty_pred)
acc = accuracy_score(ty, np.argmax(ty_pred, axis=1))

print(f"{cce=:.3f}, {acc=:.3f}")

# %%
plt.figure(figsize=(WIDTH_PLOTS, HEIGHT_PLOTS))
disp = CalibrationDisplay.from_predictions(ty, ty_pred[:, 1])
plt.show()

# %%
ensemble = create_ensemble_from_checkpoints("greedy")

# %%
plot_decision_boundary_and_uncertainty(tx, ty, ensemble, steps=1000, color_map="viridis")

ty_pred = ensemble.predict(tx)["loc"]

cce = log_loss(ty, ty_pred)
acc = accuracy_score(ty, np.argmax(ty_pred, axis=1))

print(f"{cce=:.3f}, {acc=:.3f}")

# %% [markdown]
# The improvement over the default hyperparameters is significant.
# 
# For CCE, we moved from about 6 to 0.4.
# 
# For Accuracy, we moved from 0.82 to 0.87.

# %%
plt.figure(figsize=(WIDTH_PLOTS, HEIGHT_PLOTS))
disp = CalibrationDisplay.from_predictions(ty, ty_pred[:, 1])
plt.show()