Entropy calculations
====================

This recipe explains how to compute Shannon entropy for different types of
datasets using the functions in the `dynsight.analysis` module.

First of all, we import all the packages and objects we'll need:

.. testcode:: recipe4-test

    import numpy as np
    import dynsight
    import matplotlib.pyplot as plt

    np.random.seed(42)  # set the random seed


Entropy of a discrete variable
------------------------------

Let's compute the Shanon entropy of rolling a dice ``n_sample`` times, which
should be equal to log2(6) bits.

.. testcode:: recipe4-test

    n_sample = 10000
    rolls = np.random.randint(1, 7, size=n_sample)

    dice_entropy = dynsight.analysis.compute_shannon(
        data=rolls,
        data_range=(1,6),
        n_bins=6,
        units="bit",
    )
    # dice_entropy = 2.584832195231254 ~ log2(6)


Entropy of a discrete multivariate variable
-------------------------------------------

Let's compute the Shanon entropy of rolling `two` dices ``n_sample`` times,
which should be equal to log2(36) bits.

.. testcode:: recipe4-test

    n_sample = 10000
    rolls = np.random.randint(1, 7, size=(n_sample, 2))

    dices_entropy = dynsight.analysis.compute_shannon_multi(
        data=rolls,
        data_ranges=[(1,6), (1,6)],
        n_bins=[6, 6],
        units="bit",
    )
    # dices_entropy = 5.168428344754391 ~ log2(36)


Entropy of a continuous variable
---------------------------------

Shannon entropy is not univocally defined for continuous variables, but the
difference between the entropy of different distribution is. Let's compute the
difference between the Shannon entropy of two Gaussian distributions, with
standard deviations respectively equal to 1 and 2, which should be 1 bit.

.. testcode:: recipe4-test

    n_sample = 10000000
    data_1 = np.random.normal(loc=0.0, scale=1.0, size=n_sample)
    data_2 = np.random.normal(loc=0.0, scale=2.0, size=n_sample)

    gauss_entropy_1 = dynsight.analysis.compute_kl_entropy(
        data=data_1,
        units="bit",
    )
    gauss_entropy_2 = dynsight.analysis.compute_kl_entropy(
        data=data_2,
        units="bit",
    )
    diff = gauss_entropy_2 - gauss_entropy_1
    # diff = 1.0010395631476854


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
    data_1 = np.random.multivariate_normal(
        mean=mean,
        cov=cov,
        size=n_sample,
    )
    data_2 = np.random.multivariate_normal(
        mean=mean,
        cov=cov * 4.0,
        size=n_sample,
    )

    gauss_entropy_1 = dynsight.analysis.compute_kl_entropy_multi(
        data=data_1,
        units="bit",
    )
    gauss_entropy_2 = dynsight.analysis.compute_kl_entropy_multi(
        data=data_2,
        units="bit",
    )
    diff_2d = gauss_entropy_2 - gauss_entropy_1
    # diff_2d = 1.9983384346024948


%.. raw:: html
%
%    <a class="btn-download" href="../_static/recipes/entropy.py" download>⬇️ Download Python Script</a>

.. testcode:: recipe4-test
    :hide:

    assert np.isclose(dice_entropy, np.log2(6), rtol=1e-3)
    assert np.isclose(dices_entropy, np.log2(36), rtol=1e-3)
    assert np.isclose(diff, 1, rtol=1e-3, atol=1e-4)
    assert np.isclose(diff_2d, 2, rtol=1e-3, atol=1e-4)
