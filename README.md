# Gpax
GPax is a small Python package for phyiscs-based Gaussian processes (GPs) built on top of NumPyro and JAX.

## How to use
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
We can plot the GP prediction using the stadard approach where the uncertainty in predictions - represented by a standard deviation in ```y_sampled``` - is depicted as a shaded area around the mean value.
```python3
plt.scatter(X, y, marker='x', c='k', zorder=2, label="Noisy observations", alpha=0.7)
plt.plot(X_test, y_pred, lw=1.5, zorder=2, c='b', label='Sampled means (CoM)')
plt.fill_between(X_test, y_pred - y_sampled.std(0), y_pred + y_sampled.std(0),
                color='r', alpha=0.3, label="Model uncertainty")
plt.legend()
```
