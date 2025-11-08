Entropy calculations
====================

This recipe explains how to compute Shannon entropy for different types of
datasets using the functions in the ``dynsight.analysis`` module.

First of all, we import all the packages and objects we'll need:

.. testcode:: recipe4-test

    import numpy as np
    import dynsight

    rng = np.random.default_rng(42)  # set the random seed


Entropy of a discrete variable
------------------------------

Let's compute the Shanon entropy of rolling a dice ``n_sample`` times, which
should be equal to log2(6) bits.

.. testcode:: recipe4-test

    n_sample = 10000
    rolls = rng.integers(1, 7, size=n_sample)

    dice_entropy = dynsight.analysis.shannon(
        data=rolls,
        method="histo",
        base=2,
    )
    # dice_entropy = 2.584832195231254 ~ log2(6)


Entropy of a discrete multivariate variable
-------------------------------------------

Let's compute the Shanon entropy of rolling two dices ``n_sample`` times,
which should be equal to log2(36) bits.

.. testcode:: recipe4-test

    n_sample = 10000
    rolls = rng.integers(1, 7, size=(n_sample, 2))

    dices_entropy = dynsight.analysis.shannon(
        data=rolls,
        method="histo",
        base=2,
    )
    # dices_entropy = 5.168428344754391 ~ log2(36)


Entropy of a continuous variable
---------------------------------

Shannon entropy is not univocally defined for continuous variables, but the
difference between the entropy of different distribution is.
For continuous variables, we need to use the Kozachenko-Leonenko (KL)
estimator, passing the argument ``method="kl"``.
Let's compute the difference between the Shannon entropy of two Gaussian distributions, with standard deviations respectively equal to 1 and 2, which
should be 1 bit.

.. testcode:: recipe4-test

    n_sample = 100000
    data_1 = rng.normal(loc=0.0, scale=1.0, size=n_sample)
    data_2 = rng.normal(loc=0.0, scale=2.0, size=n_sample)

    gauss_entropy_1 = dynsight.analysis.shannon(
        data=data_1,
        method="kl",
        base=2,
    )
    gauss_entropy_2 = dynsight.analysis.shannon(
        data=data_2,
        method="kl",
        base=2,
    )
    diff = gauss_entropy_2 - gauss_entropy_1
    # diff = 0.9994806386420283


Entropy of a continuous multivariate variable
---------------------------------------------

And the same is true for multivariate distributions. Let's compute the
difference between the Shannon entropy of two bivariate Gaussian
distributions, with standard deviations respectively equal to 1 and 2,
which should be 2 bits.

.. testcode:: recipe4-test

    n_sample = 100000
    mean = [1, 1]
    cov = np.array([[1, 0], [0, 1]])
    data_1 = rng.multivariate_normal(
        mean=mean,
        cov=cov,
        size=n_sample,
    )
    data_2 = rng.multivariate_normal(
        mean=mean,
        cov=cov * 4.0,
        size=n_sample,
    )

    gauss_entropy_1 = dynsight.analysis.shannon(
        data=data_1,
        method="kl",
        base=2,
    )
    gauss_entropy_2 = dynsight.analysis.shannon(
        data=data_2,
        method="kl",
        base=2,
    )
    diff_2d = gauss_entropy_2 - gauss_entropy_1
    # diff_2d = 2.0101274002195764


.. raw:: html

    <a class="btn-download" href="_static/recipes/entropy.py" download>⬇️ Download Python Script</a>

.. testcode:: recipe4-test
    :hide:

    assert np.isclose(dice_entropy, np.log2(6), rtol=1e-3)
    assert np.isclose(dices_entropy, np.log2(36), rtol=1e-3)
    assert np.isclose(diff, 1, rtol=1e-3, atol=1e-4)
    assert np.isclose(diff_2d, 2, rtol=1e-2, atol=1e-2)
