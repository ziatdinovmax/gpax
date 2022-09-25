How To Use
==========

Simple GP
---------

The code snippet below shows how to use vanilla GP in a fully Bayesian mode. First, we infer GP model parameters from the available training data

.. code:: python

    import gpax
    # Get random number generator keys for training and prediction
    rng_key_train, rng_key_predict = gpax.utils.get_keys()
    # Initialize model
    gp_model = gpax.ExactGP(1, kernel='RBF')
    # Run Hamiltonian Monte Carlo to obtain posterior samples for the GP model parameters
    gp_model.fit(rng_key_train, X, y)  # X and y are numpy arrays with dimensions (n, d) and (n,)

In the fully Bayesian mode, we get a pair of predictive mean and covariance for each Hamiltonian Monte Carlo sample containing the GP parameters (in this case, the RBF kernel hyperparameters and model noise). Hence, a prediction on new inputs with a trained GP model returns the center of the mass of all the predictive means (``y_pred``) and samples from multivariate normal distributions for all the pairs of predictive means and covariances (``y_sampled``).

.. code:: python

    y_pred, y_sampled = gp_model.predict(rng_key_predict, X_test)

.. image:: imgs/GPax_Fig1.jpg
  :alt: GPax_GP1

For 1-dimensional data, we can plot the GP prediction using the standard approach where the uncertainty in predictions - represented by a standard deviation in ``y_sampled`` - is depicted as a shaded area around the mean value.

.. image:: imgs/GPax_Fig2.jpg
  :alt: GPax_GP2

Structured GP
-------------

The limitation of the standard GP is that it does not usually allow for the incorporation of prior domain knowledge and can be biased toward a trivial interpolative solution. Recently, we `introduced <https://arxiv.org/abs/2108.10280>`_ a structured Gaussian Process (sGP), where a classical GP is augmented by a structured probabilistic model of the expected system’s behavior. This approach allows us to `balance <https://towardsdatascience.com/unknown-knowns-bayesian-inference-and-structured-gaussian-processes-why-domain-scientists-know-4659b7e924a4>`_ the flexibility of the non-parametric GP approach with a rigid structure of prior (physical) knowledge encoded into the parametric model.
Implementation-wise, we substitute a constant/zero prior mean function in GP with a probabilistic model of the expected system's behavior.

For example, if we have prior knowledge that our objective function has a discontinuous 'phase transition', and a power law-like behavior before and after this transition, we may express it using a simple piecewise function

.. code:: python

    import jax.numpy as jnp

    def piecewise(x: jnp.ndarray, params: Dict[str, float]) -> jnp.ndarray:
        """Power-law behavior before and after the transition"""
        return jnp.piecewise(
            x, [x < params["t"], x >= params["t"]],
            [lambda x: x**params["beta1"], lambda x: x**params["beta2"]])

This function is deterministic. To make it probabilistic, we put priors over its parameters with the help of NumPyro

.. code:: python

    import numpyro
    from numpyro import distributions

    def piecewise_priors():
        # Sample model parameters
        t = numpyro.sample("t", distributions.Uniform(0.5, 2.5))
        beta1 = numpyro.sample("beta1", distributions.Normal(3, 1))
        beta2 = numpyro.sample("beta2", distributions.Normal(3, 1))
        # Return sampled parameters as a dictionary
        return {"t": t, "beta1": beta1, "beta2": beta2}

Finally, we train the sGP model and make predictions on new data in the almost exact same way we did for vanilla GP. The only difference is that we pass our structured probabilistic model as two new arguments (the piecewise function and the corresponding priors over its parameters) when initializing GP.

.. code:: python

    # Get random number generator keys
    rng_key_train, rng_key_predict = gpax.utils.get_keys()
    # Initialize structured GP model
    sgp_model = gpax.ExactGP(1, kernel='Matern', mean_fn=piecewise, mean_fn_prior=piecewise_priors)
    # Run MCMC to obtain posterior samples
    sgp_model.fit(rng_key_train, X, y)
    # Get GP prediction on new/test data
    y_pred, y_sampled = sgp_model.predict(rng_key_predict, X_test)

.. image:: imgs/GPax_FIg3.jpg
  :alt: GPax_sGP1

The probabilistic model reflects our prior knowledge about the system, but it does not have to be precise, that is, the model can have a different functional form, as long as it captures general or partial trends in the data. 

.. image:: imgs/GPax_FIg4.jpg
  :scale: 13
  :alt: GPax_sGP2

Active learning & Bayesian optimization
---------------------------------------

Both GP and sGP can be used for active learning to reconstruct the entire data distribution from sparse observations or to localize regions of the parameter space where a particular physical behavior is maximized or minimized with as few measurements as possible (the latter is usually referred to as Bayesian optimization)

.. code:: python

  # Train a GP model (it can be sGP or vanilla GP)
  gp_model.fit(rng_key, X_measured, y_measured)  # A

  # Compute the upper confidence bound (UCB) acquisition function to derive the next measurement point
  acq = gpax.acquisition.UCB(rng_key_predict, gp_model, X_unmeasured, beta=4, maximize=False, noiseless=True)  # B
  next_point_idx = acq.argmin()  # C
  next_point = X_unmeasured[next_point_idx]  # D

  # Perform measurement in next_point, update measured & unmeasured data arrays, and re-run steps A-D.

In the figure below we illustrate the connection between the (s)GP posterior predictive distribution and the acquisiton function used to derive the next measurement points. Here, the posterior mean values indicate that the minimum of a "black box" function describing a behaviour of interest is around x=0.7. At the same time, there is a large dispersion in the samples from the posterior predictive distribution between x=-0.5 and x=0.5, resulting in a high uncertainty in that region. The acquisition function is computed as a function of both predictive mean and uncertainty and its minimum corresponds to the next measurement point in the active learning / Bayesian optimization setup. Here, after taking into account the uncertainty in the prediction, the UCB acquisition function suggests exploring a point at x≈0 where potentially a true minimum is located.

.. image:: imgs/BO.png
  :alt: GPax_BO

Hypothesis learning
-------------------

The structured GP can be also used for hypothesis learning in automated experiments. The `hypothesis learning <https://arxiv.org/abs/2112.06649>`_ is based on the idea that in active learning, the correct model of the system’s behavior leads to a faster decrease in the overall Bayesian uncertainty about the system under study. In the hypothesis learning setup, probabilistic models of the possible system’s behaviors (hypotheses) are wrapped into structured GPs, and a basic reinforcement learning policy is used to select a correct model from several competing hypotheses. A full example is available `here <https://colab.research.google.com/github/ziatdinovmax/gpax/blob/main/examples/hypoAL.ipynb>`_.

.. image:: imgs/HypoAL.gif
  :alt: GPax_HypoAL

Deep Kernel Learning
--------------------

`Deep Kernel Learning <https://arxiv.org/abs/1511.02222>`_ can be understood as a hybrid of deep neural network (DNN) and GP. The DNN serves as a feature extractor that allows reducing the complex high-dimensional features to low-dimensional descriptors on which a standard GP kernel operates. The parameters of DNN and of GP kernel are inferred jointly in an end-to-end fashion. Practically, the DKL training inputs are usually patches from an (easy-to-acquire) structural image, and training targets represent a physical property of interest derived from the (hard-to-acquire) spectra measured in those patches. The DKL output on the new inputs (image patches for which there are no measured spectra) is the expected property value and associated uncertainty, which can be used to derive the next measurement point in the automated experiment.

.. code:: python
  
  import gpax

  # Get random number generator keys for training and prediction
  rng_key, rng_key_predict = gpax.utils.get_keys()

  # Obtain/update DKL posterior; input data dimensions are (n, h*w*c)
  dkl = gpax.viDKL(input_dim=X.shape[-1], z_dim=2, kernel='RBF')  # A
  dkl.fit(rng_key, X_train, y_train, num_steps=100, step_size=0.05)  # B

  # Compute UCB acquisition function
  obj = gpax.acquisition.UCB(rng_key_predict, dkl, X_unmeasured, maximize=True)  # C
  # Select next point to measure (assuming grid data)
  next_point_idx = obj.argmax()  # D

  # Perform measurement in next_point_idx, update measured & unmeasured data arrays, and re-run steps A-D.

Below we show a result of a simple DKL-based search for regions of the nano-plasmonic array that host a specific plasmon mode

.. image:: imgs/DKL_STEM.png
  :alt: GPax_DKL

Note that in viDKL, we use a simple MLP as a default feature extractor. However, you can easily write a custom DNN using `haiku <https://github.com/deepmind/dm-haiku>`_ and pass it to the viDKL initializer

.. code:: python

  import haiku as hk

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

  dkl = gpax.viDKL(X.shape[1:], 2, kernel='RBF', nn=ConvNet)  # input data dimensions are (n,h,w,c)
  dkl.fit(rng_key, X_train, y_train, num_steps=100, step_size=0.05)
  obj = gpax.acquisition.UCB(rng_key_predict, dkl, X_unmeasured, maximize=True)
  next_point_idx = obj.argmax()