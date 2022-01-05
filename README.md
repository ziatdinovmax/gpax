# GPax
GPax is a small Python package for physics-based Gaussian processes (GPs) built on top of NumPyro and JAX. Its purpose is to take advantage of prior physical knowledge and different data modalities when using GPs for data reconstruction and active learning. It is a work in progress, and more models will be added in the near future.

## How to use
### Simple GP
The code snippet below shows how to use vanilla GP in a fully Bayesian mode
```python3
# Get random number generator keys (see JAX documentation for why it is neccessary)
rng_key, rng_key_predict = gpax.utils.get_keys()
# Initialize model
gp_model = gpax.ExactGP(1, kernel='Matern')
# Run MCMC to obtain posterior samples
gp_model.fit(rng_key, X, y, num_chains=1)  # X and y dimensions are (n, d) and (n,)
```
The prediction with a trained GP model returns the center of the mass of the sampled means (```y_pred```) and samples from multivariate normal posteriors (```y_sampled```). Note that in a [fully Bayesian mode](https://docs.gpytorch.ai/en/v1.5.1/examples/01_Exact_GPs/GP_Regression_Fully_Bayesian.html), we get a multivariate normal posterior for each MCMC sample with kernel hyperparameters.
```python3
y_pred, y_sampled = gp_model.predict(rng_key_predict, X_test)
```
We can plot the GP prediction using the standard approach where the uncertainty in predictions - represented by a standard deviation in ```y_sampled``` - is depicted as a shaded area around the mean value (if the data is 1-dimensional). See the full example [here](https://colab.research.google.com/github/ziatdinovmax/gpax/blob/main/examples/simpleGP.ipynb).
```python3
plt.scatter(X, y, marker='x', c='k', zorder=2, label="Noisy observations", alpha=0.7)
plt.plot(X_test, y_pred, lw=1.5, zorder=2, c='b', label='Sampled means (CoM)')
plt.fill_between(X_test, y_pred - y_sampled.std(0), y_pred + y_sampled.std(0),
                color='r', alpha=0.3, label="Model uncertainty")
plt.legend()
```
### Structured GP
The limitation of the standard GP is that it does not usually allow for the incorporation of prior domain knowledge and can be biased toward a trivial interpolative solution. Recently, we [introduced](https://arxiv.org/abs/2108.10280) a structured Gaussian Process (sGP), where a classical GP is augmented by a structured probabilistic model of the expected system’s behavior. This approach allows us to [balance](https://towardsdatascience.com/unknown-knowns-bayesian-inference-and-structured-gaussian-processes-why-domain-scientists-know-4659b7e924a4) the flexibility of the non-parametric GP approach with a rigid structure of prior (physical) knowledge encoded into the parametric model.
Implementation-wise, we substitute a constant/zero prior mean function in GP with a probabilistic model of the expected system's behavior.

For example, if we have prior knowledge that our objective function has a discontinuous 'phase transition', and a power law-like behavior before and after this transition, we may express it using function
```python3
def piecewise(x: jnp.ndarray, params: Dict[str, float]) -> jnp.ndarray:
    """Power-law behavior before and after the transition"""
    return jnp.piecewise(
        x, [x < params["t"], x >= params["t"]],
        [lambda x: x**params["beta1"], lambda x: x**params["beta2"]])
```
where ```jnp``` corresponds to jax.numpy module. This function is deterministic. To make it probabilistic, we put priors over its parameters with the help of [NumPyro](https://github.com/pyro-ppl/numpyro)
```python3
def piecewise_priors():
    # Sample model parameters
    t = numpyro.sample("t", numpyro.distributions.Uniform(0.5, 2.5))
    beta1 = numpyro.sample("beta1", numpyro.distributions.Normal(3, 1))
    beta2 = numpyro.sample("beta2", numpyro.distributions.Normal(3, 1))
    # Return sampled parameters as a dictionary
    return {"t": t, "beta1": beta1, "beta2": beta2}
```
Finally, we train the sGP model and make predictions on new data in the almost exact same way we did for vanilla GP. The only difference is that we pass our structured probabilistic model as two new arguments (the piecewise function and the corresponding priors over its parameters) when initializing GP.
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
The probabilistic model reflects our prior knowledge about the system, but it does not have to be precise, that is, the model can have a different functional form, as long as it captures general or partial trends in the data. The full example is available [here](https://colab.research.google.com/github/ziatdinovmax/gpax/blob/main/examples/GP_sGP.ipynb).
### Deep kernel learning
Deep kernel learning (DKL), initially [introduced](https://arxiv.org/abs/1511.02222) by Andrew Gordon Wilson, can be understood as a hybrid of deep neural network (DNN) and GP. The DNN serves as a feature extractor that allows reducing the complex high-dimensional features to low-dimensional descriptors on which a standard GP kernel operates. The parameters of DNN and of GP kernel are inferred jointly in an end-to-end fashion. Practically, the DKL training inputs are usually patches from an (easy-to-acquire) structural image, and training targets represent a physical property of interest derived from the (hard-to-acquire) spectra measured in those patches. The DKL output on the new inputs (image patches for which there are no measured spectra) is the expected property value and associated uncertainty, which can be used to derive the next measurement point in the automated experiment. 

GPax package has the fully Bayesian DKL (weights of neural network and GP hyperparameters are inferred using MCMC) and the Variational Inference approximation of DKL, viDKL. The fully Bayesian DKL can provide an asymptotically exact solution but is too slow for most automated experiments. Hence, for the latter, one may use the viDKL
```python3
# Obtain/update DKL posterior
dkl = gpax.viDKL(X.shape[-1], 2)  # input data dimensions are (n, h*w*c)
dkl.fit(rng_key, X_train, y_train, num_steps=100, step_size=0.05)
# Compute UCB acquisition function
obj = gpax.acquisition.UCB(rng_key_predict, dkl, X_unmeasured, maximize=True)
# Select next point to measure (assuming grid data)
next_point_idx = obj.argmax()
# Perform measurement, update trainning data, etc.
```
The full example is available [here](https://colab.research.google.com/github/ziatdinovmax/gpax/blob/main/examples/gpax_viDKL_plasmons.ipynb). Note that in viDKL, we use a simple MLP as a default feature extractor. However, you can easily write a custom DNN using [haiku](https://github.com/deepmind/dm-haiku) and pass it to the viDKL initializer
```python3
class ConvNet(hk.Module):
    def __init__(self, embedim=2):
        super().__init__()
        self._embedim = embedim   

    def __call__(self, x):
        x = hk.Conv2D(32, 3)(x)
        x = jax.nn.relu(x)
        x = hk.MaxPool(2, 2, 'SAME')(x)
        x = hk.Conv2D(64, 3)(x)
        x = jax.nn.relu(x)
        x = hk.Flatten()(x)
        x = hk.Linear(self._embedim)(x)
        return x

dkl = gpax.viDKL(X.shape[1:], 2, nn=ConvNet)  # input data dimensions are (n,h,w,c)
dkl.fit(rng_key, X_train, y_train, num_steps=100, step_size=0.05)
obj = gpax.acquisition.UCB(rng_key_predict, dkl, X_unmeasured, maximize=True)
next_point_idx = obj.argmax()
```
## Installation
If you would like to utilize a GPU acceleration, follow these [instructions](https://github.com/google/jax#installation) to install JAX with a GPU support.

Then, install GPax using pip:

```$ pip install git+https://github.com/ziatdinovmax/gpax```

If you are a Windows user, we recommend to use the Windows Subsystem for Linux (WSL2), which comes free Windows 10 and 11.

## Cite us

If you use GPax in your work, please consider citing our papers:
```
@article{ziatdinov2021physics,
  title={Physics makes the difference: Bayesian optimization and active learning via augmented Gaussian process},
  author={Ziatdinov, Maxim and Ghosh, Ayana and Kalinin, Sergei V},
  journal={arXiv preprint arXiv:2108.10280},
  year={2021}
}

@article{ziatdinov2021hypothesis,
  title={Hypothesis learning in an automated experiment: application to combinatorial materials libraries},
  author={Ziatdinov, Maxim and Liu, Yongtao and Morozovska, Anna N and Eliseev, Eugene A and Zhang, Xiaohang and Takeuchi, Ichiro and Kalinin, Sergei V},
  journal={arXiv preprint arXiv:2112.06649},
  year={2021}
}
```
