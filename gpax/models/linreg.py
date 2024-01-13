import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDiagonalNormal
from numpyro.optim import Adam
import jax

class LinReg:
    """Simple linear regression model"""
    def __init__(self):
        self.params = None

    @staticmethod
    def model(x, y=None):
        beta = numpyro.sample('beta', dist.Normal(jnp.zeros(x.shape[1]), 10*jnp.ones(x.shape[1])))
        alpha = numpyro.sample('alpha', dist.Normal(0, 10))
        sigma = numpyro.sample('sigma', dist.HalfCauchy(1))

        mu = alpha + jnp.dot(x, beta)
        with numpyro.plate('data', x.shape[0]):
            numpyro.sample('obs', dist.Normal(mu, sigma), obs=y)

    def train(self, x, y, learning_rate=0.01, num_iterations=5000):
        guide = AutoDiagonalNormal(self.model)
        optimizer = Adam(step_size=learning_rate)
        svi = SVI(self.model, guide, optimizer, loss=Trace_ELBO(), x=x, y=y)

        params, _, _ = svi.run(jax.random.PRNGKey(0), num_iterations, progress_bar=False)
        self.params = svi.guide.median(params)

    def predict(self, x_new):
        alpha = self.params['alpha']
        beta = self.params['beta']
        sigma = self.params['sigma']
        return alpha + jnp.dot(x_new, beta)

    def get_params(self):
        return self.params
