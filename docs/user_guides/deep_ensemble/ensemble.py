import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from deephyper.ensemble import BaggingEnsembleRegressor
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

def mse(y_true, y_pred):
    return tf.square(y_true - y_pred)

ensemble = BaggingEnsembleRegressor(
    model_dir="save/model",
    loss=mse,  # default is mse
    size=5,
    verbose=True,
    ray_address="",
    num_cpus=1,
    num_gpus=None,
    selection="topk",
)

ensemble.fit(vX, vy)

vy_pred = ensemble.predict(vX)
ty_pred = ensemble.predict(tX)

scores_valid = ensemble.evaluate(vX, vy)
scores_test = ensemble.evaluate(tX, ty)
print("scaled valid loss: ", scores_valid)
print("scaled test loss: ", scores_test)

err_valid = tf.reduce_mean(
    mse(scaler_y.inverse_transform(vy), scaler_y.inverse_transform(vy_pred))
).numpy()
err_test = tf.reduce_mean(
    mse(scaler_y.inverse_transform(ty), scaler_y.inverse_transform(ty_pred))
).numpy()
print("reversed scaling valid loss: ", err_valid)
print("reversed scaling test loss: ", err_test)

print(ensemble.members_files)

plt.figure()
plt.plot(tX.flatten(), ty.flatten(), "-")
plt.plot(tX.flatten(), ty_pred.flatten(), "-")
plt.show()
