import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from deephyper.ensemble import UQBaggingEnsembleRegressor
from deephyper.nas.preprocessing import stdscaler
from dh_project.cubic.load_data import load_data_train_valid, load_data_train_test

(X, y), (vX, vy) = load_data_train_valid(random_state=42)
_, (tX, ty) = load_data_train_test(random_state=42)

scaler_X = stdscaler()
X = scaler_X.fit_transform(X)
vX = scaler_X.transform(vX)
tX = scaler_X.transform(tX)

scaler_y = stdscaler()
y = scaler_y.fit_transform(y)
vy = scaler_y.transform(vy)
ty = scaler_y.transform(ty)


def nll(y, rv_y):
    """Negative log likelihood for Tensorflow probability."""
    return -rv_y.log_prob(y)


ensemble = UQBaggingEnsembleRegressor(
    model_dir="save/model",
    loss=nll,  # default is nll
    size=20,
    verbose=True,
    ray_address="",
    num_cpus=1,
    num_gpus=None,
    selection="topk",
)

ensemble.fit(vX, vy)

vy_pred_dist = ensemble.predict(vX)
ty_pred_dist = ensemble.predict(tX)

scores_valid = ensemble.evaluate(vX, vy)
scores_test = ensemble.evaluate(tX, ty)
print("scaled valid loss: ", scores_valid)
print("scaled test loss: ", scores_test)

print(ensemble.members_files)

plt.figure()

plt.plot(
    scaler_X.inverse_transform(vX).flatten(),
    scaler_y.inverse_transform(vy).flatten(),
    "b*",
    label="validation data",
)

x = scaler_X.inverse_transform(tX).flatten()

plt.plot(x, scaler_y.inverse_transform(ty).flatten(), "k--", label="Truth")

samples = ty_pred_dist.sample(1000)
samples = scaler_y.inverse_transform(samples)
mean = np.mean(samples, axis=0).flatten()
std = np.std(samples, axis=0).flatten()

plt.plot(x, mean, "r-", label="mean", alpha=0.8)

plt.plot(x, mean - 2 * std, "r-")
plt.fill_between(x, y1=mean, y2=mean - 2 * std, color="red", alpha=0.1)

plt.plot(x, mean + 2 * std, "r-")
plt.fill_between(x, y1=mean, y2=mean + 2 * std, color="red", alpha=0.1)

plt.legend()
plt.show()
