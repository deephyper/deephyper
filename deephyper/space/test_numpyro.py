import jax.random
import numpyro

import numpyro.distributions as dist

from numpyro.infer import HMC, MCMC, MixedHMC

# from numpyro.infer import NUTS

import numpy as np


def model():
    numpyro.sample("x", dist.DiscreteUniform(10, 12))
    numpyro.sample("y", dist.Categorical(np.asarray([0.25, 0.25, 0.25, 0.25])))


num_samples = 10
kernel = HMC(model, trajectory_length=1.2)
kernel = MixedHMC(kernel, num_discrete_updates=20)
# kernel = NUTS(model)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=num_samples, progress_bar=False)
key = jax.random.PRNGKey(0)
mcmc.run(key)
samples = mcmc.get_samples()

print(samples)
