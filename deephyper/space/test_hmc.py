from jax import random
import jax.numpy as jnp

# from jax.nn import softplus, one_hot, relu, leaky_relu
import numpyro
import numpyro.distributions as dist
from numpyro.infer import HMC, MCMC

# from numpyro.infer import MixedHMC

import matplotlib.pyplot as plt


# Example: Constraint on DiscreteUniform variables
# y <= x
# def model():
#     x = numpyro.sample("x", dist.DiscreteUniform(low=1, high=10))
#     y = numpyro.sample("y", dist.DiscreteUniform(low=1, high=10))
#     # Continuous soft constraint
#     # numpyro.factor("c1", -jnp.exp(5 * softplus(y - x)))
#     # Discrete hard constraint
#     numpyro.factor("c1", -5 * jnp.where(y > x, 1, 0))


# Example: Constraint on Uniform variables
# x + y <= 1
def model():
    x = numpyro.sample("x", dist.Uniform(low=0, high=1))
    y = numpyro.sample("y", dist.Uniform(low=0, high=1))
    # Continuous soft constraint
    # numpyro.factor("c1", -jnp.exp(5 * softplus(y + x - 1)))
    numpyro.factor("c1", -jnp.exp(10 * (y + x - 1)) - 1)
    # numpyro.factor("c1", -(10 * (y + x - 1)))
    # Discrete hard constraint # INFINITE LOOP
    # numpyro.factor("c1", -5 * jnp.where(y + x < 1, 1, 0))


# Example: Constraint on Categorical variables
# not (x == 0 and y == 1)
# def model():
#     num_cat = 10
#     dist_x = dist.Categorical(probs=jnp.ones(num_cat) / num_cat)
#     dist_y = dist.Categorical(probs=jnp.ones(num_cat) / num_cat)
#     x = numpyro.sample("x", dist_x)
#     y = numpyro.sample("y", dist_y)
#     numpyro.factor("c1", -5 * jnp.where(jnp.logical_and(x == 0, y == 1), 1, 0))


# Example: Constraint on Categorical variables
# not (x == y)
# def model():
#     num_cat_x = 100
#     num_cat_y = 200
#     dist_x = dist.Categorical(probs=jnp.ones(num_cat_x) / num_cat_x)
#     dist_y = dist.Categorical(probs=jnp.ones(num_cat_y) / num_cat_y)
#     x = numpyro.sample("x", dist_x)
#     y = numpyro.sample("y", dist_y)
#     # Continuous version of the constraint
#     num_cat = max(num_cat_x, num_cat_y)  # jnp.maximum(num_cat_x, num_cat_y)
#     x_onehot = one_hot(x, num_cat)
#     y_onehot = one_hot(y, num_cat)
#     numpyro.factor("c1", -jnp.exp(5 * softplus(jnp.dot(x_onehot, y_onehot) - 1)))
#     # Discrete version of the constraint
#     # numpyro.factor("c1", -5 * jnp.where(x == y, 1, 0))


# Sampler
# kernel = MixedHMC(HMC(model, trajectory_length=1.2), num_discrete_updates=20)
kernel = HMC(model, trajectory_length=1.2)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=10_000, progress_bar=True)
mcmc.run(random.PRNGKey(0))
# mcmc.print_summary()
samples = mcmc.get_samples()

# print(samples[:10])
print(samples)

# print(samples["x_int"] + samples[ "x_real"])
# assert "x" in samples and "c" in samples
# assert abs(jnp.mean(samples["x"]) - 1.3) < 0.1
# assert abs(jnp.var(samples["x"]) - 4.36) < 0.5


plt.figure()
plt.scatter(samples["x"], samples["y"], alpha=0.1)
# plt.hist2d(samples["x"], samples["y"], bins=(100, 200))
plt.show()
