import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro.distributions as dist
from jax import lax
from numpyro.distributions import constraints
from numpyro.distributions.distribution import TransformedDistribution
from numpyro.distributions.transforms import ExpTransform
from numpyro.distributions.util import promote_shapes, validate_sample
from numpyro.util import is_prng_key, not_jax_tracer


class LogUniform(TransformedDistribution):
    arg_constraints = {"low": constraints.positive, "high": constraints.positive}
    reparametrized_params = ["low", "high"]
    pytree_data_fields = ("low", "high", "_support")

    def __init__(self, low, high, *, validate_args=None):
        base_dist = dist.Uniform(jnp.log(low), jnp.log(high))
        self.low, self.high = promote_shapes(low, high)
        self._support = constraints.interval(self.low, self.high)
        super(LogUniform, self).__init__(
            base_dist, ExpTransform(), validate_args=validate_args
        )

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return self._support

    @property
    def mean(self):
        return (self.high - self.low) / jnp.log(self.high / self.low)

    @property
    def variance(self):
        return (
            0.5 * (self.high**2 - self.low**2) / jnp.log(self.high / self.low)
            - self.mean**2
        )

    def cdf(self, x):
        return self.base_dist.cdf(jnp.log(x))

    def entropy(self):
        log_low = jnp.log(self.low)
        log_high = jnp.log(self.high)
        return (log_low + log_high) / 2 + jnp.log(log_high - log_low)


class DiscreteLogUniform(dist.Distribution):
    arg_constraints = {"low": constraints.dependent, "high": constraints.dependent}
    has_enumerate_support = True
    pytree_data_fields = ("low", "high", "_support")

    def __init__(self, low=0, high=1, *, validate_args=None):
        self.low, self.high = promote_shapes(low, high)
        batch_shape = lax.broadcast_shapes(jnp.shape(low), jnp.shape(high))
        self._support = constraints.integer_interval(low, high)
        super().__init__(batch_shape, validate_args=validate_args)

    @constraints.dependent_property(is_discrete=True, event_dim=0)
    def support(self):
        return self._support

    def sample(self, key, sample_shape=()):
        shape = sample_shape + self.batch_shape
        log_base = 10
        return jnp.floor(
            jnp.power(
                log_base,
                random.uniform(
                    key,
                    shape=shape,
                    minval=jnp.log(self.low) / jnp.log(log_base),
                    maxval=jnp.log(self.high + 1) / jnp.log(log_base),
                ),
            )
        ).astype(jnp.int32)

    @validate_sample
    def log_prob(self, value):
        low = jnp.log10(self.low)
        high = jnp.log10(self.high + 1)
        shape = lax.broadcast_shapes(jnp.shape(value), self.batch_shape)
        return -jnp.broadcast_to(jnp.log(high - low), shape)

    def cdf(self, value):
        low = jnp.log10(self.low)
        high = jnp.log10(self.high + 1)
        cdf = (jnp.floor(value) + 1 - low) / (high - low + 1)
        return jnp.clip(cdf, 0.0, 1.0)

    def icdf(self, value):
        low = jnp.log10(self.low)
        high = jnp.log10(self.high + 1)
        return low + value * (high - low + 1) - 1

    @property
    def mean(self):
        return (self.high - self.low) / jnp.log(self.high / self.low)

    @property
    def variance(self):
        return (
            0.5 * (self.high**2 - self.low**2) / jnp.log(self.high / self.low)
            - self.mean**2
        )

    def enumerate_support(self, expand=True):
        if not not_jax_tracer(self.high) or not not_jax_tracer(self.low):
            raise NotImplementedError("Both `low` and `high` must not be a JAX Tracer.")
        if np.any(np.amax(self.low) != self.low):
            # NB: the error can't be raised if inhomogeneous issue happens when tracing
            raise NotImplementedError(
                "Inhomogeneous `low` not supported by `enumerate_support`."
            )
        if np.any(np.amax(self.high) != self.high):
            # NB: the error can't be raised if inhomogeneous issue happens when tracing
            raise NotImplementedError(
                "Inhomogeneous `high` not supported by `enumerate_support`."
            )
        values = (self.low + jnp.arange(np.amax(self.high - self.low) + 1)).reshape(
            (-1,) + (1,) * len(self.batch_shape)
        )
        if expand:
            values = jnp.broadcast_to(values, values.shape[:1] + self.batch_shape)
        return values

    def entropy(self):
        log_low = jnp.log(self.low)
        log_high = jnp.log(self.high)
        return (log_low + log_high) / 2 + jnp.log(log_high - log_low)
