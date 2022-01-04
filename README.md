# Gpax
GPax is a small Python package for phyiscs-based Gaussian processes (GPs) built on top of NumPyro and JAX.

## How to use
### Simple GP
The code snippet below shows how to use vanilla GP in a fully Bayesian mode. We assume that X and y are 1-dimensional arrays.
```python3
# Get random number generator keys (see JAX documentation for why it is neccessary)
rng_key, rng_key_predict = gpax.utils.get_keys()
# Initialize model
gp_model = gpax.ExactGP(1, kernel='Matern')
# Run MCMC to obtain posterior samples
gp_model.fit(rng_key, X, y, num_chains=1)
```

The prediction with a trained GP model returns the center of the mass of the sampled means (```y_pred```) and samples from multivariate normal posteriors (```y_sampled```). Note that in a [fully Bayesian mode](https://docs.gpytorch.ai/en/v1.5.1/examples/01_Exact_GPs/GP_Regression_Fully_Bayesian.html), we get a multivariate normal posterior for each MCMC sample with kernel hyperparameters.
```python3
y_pred, y_sampled = gp_model.predict(rng_key_predict, X_test)
```
We can plot the GP prediction using the stadard approach where the uncertainty in predictions - represented by a standard deviation in ```y_sampled``` - is depicted as a shaded area around the mean value. See the full example [here]().
```python3
plt.scatter(X, y, marker='x', c='k', zorder=2, label="Noisy observations", alpha=0.7)
plt.plot(X_test, y_pred, lw=1.5, zorder=2, c='b', label='Sampled means (CoM)')
plt.fill_between(X_test, y_pred - y_sampled.std(0), y_pred + y_sampled.std(0),
                color='r', alpha=0.3, label="Model uncertainty")
plt.legend()
```
### Structured GP
The limitation of the standrd GP is that it does not usually allow for the incorporation of prior domain knowledge and can be biased toward a trivial interpolative solution. Recently, we [introduced](https://arxiv.org/abs/2108.10280) a structured Gaussian Process (sGP), where a classical GP is augmented by a structured probabilistic model of the expected system’s behavior. This approach allows us to [balance](https://towardsdatascience.com/unknown-knowns-bayesian-inference-and-structured-gaussian-processes-why-domain-scientists-know-4659b7e924a4) the flexibility of the non-parametric GP approach with a rigid structure of prior (physical) knowledge encoded into the parametric model.
Implementation-wise, we substitute a constant/zero mean function in GP with a a probabilistic model of expected system's behavior. For example, if we have if we have a prior knowledge that our objective function has a discontinuity region, and a power law-like behavior before and afer that region, we may incroporate it using a function
```python3
def piecewise(x: jnp.ndarray, params: Dict[str, float]) -> jnp.ndarray:
    """Power-law behavior before and after the transition"""
    return jnp.piecewise(
        x, [x < params["t"], x >= params["t"]],
        [lambda x: x**params["beta1"], lambda x: x**params["beta2"]])
```
where ```jnp``` corresponds to jax.numpy module. The function above is deterministic. To make it probabilistic, we add priors on its parameters with the help of [NumPyro](https://github.com/pyro-ppl/numpyro)
```python3
def piecewise_priors():
    # Sample model parameters
    t = numpyro.sample("t", numpyro.distributions.Uniform(0.5, 2.5))
    beta1 = numpyro.sample("beta1", numpyro.distributions.LogNormal(0, 1))
    beta2 = numpyro.sample("beta2", numpyro.distributions.LogNormal(0, 1))
    # Return sampled parameters as a dictionary
    return {"t": t, "beta1": beta1, "beta2": beta2}
```
Finally, we train the sGP model and make prediction on new data in almost exactly the same way we did for vanilla GP. The only difference is that we pass our function and its prior as two new arguments when initializing GP
```python3
# Get random number generator keys
rng_key, rng_key_predict = gpax.utils.get_keys()
# initialize structured GP model
sgp_model = gpax.ExactGP(1, kernel='Matern', mean_fn=piecewise, mean_fn_prior=piecewise_priors)
# Run MCMC to obtain posterior samples
sgp_model.fit(rng_key, X, y)
# Get GP prediction on new/test data
y_pred, y_sampled = sgp_model.predict(rng_key_predict, X_test)
```

