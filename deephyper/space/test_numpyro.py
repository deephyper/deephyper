import jax

jax.disable_jit()

import jax.numpy as jnp
import jax.random
import numpyro

import numpyro.distributions as dist

from numpyro.infer import HMC, MCMC, MixedHMC, NUTS

# from numpyro.infer import NUTS

import numpy as np

from _distribution import DiscreteLogUniform


def model_1():
    numpyro.sample("x", dist.DiscreteUniform(10, 12))
    numpyro.sample("y", dist.Categorical(np.asarray([0.25, 0.25, 0.25, 0.25])))


def model_2():
    x1 = numpyro.sample("x1", dist.Uniform(low=1, high=1001))
    x2 = numpyro.sample("x2", dist.LogUniform(low=1, high=1001))


num_samples = 1000
kernel = HMC(model_2, trajectory_length=1.2)
# kernel = MixedHMC(kernel, num_discrete_updates=20)
# kernel = NUTS(model_2)
mcmc = MCMC(kernel, num_warmup=100, num_samples=num_samples, progress_bar=False)

key = jax.random.PRNGKey(0)
with jax.disable_jit(False):
    mcmc.run(key)
    samples = mcmc.get_samples()

samples["x1"] = samples["x1"].astype(jnp.int32)
samples["x2"] = samples["x2"].astype(jnp.int32)
print(samples)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, sharey=True)
axes[0].hist(samples["x1"], bins=20)

axes[1].hist(samples["x2"], bins=20)
plt.show()
