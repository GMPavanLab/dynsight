"""Copiable code from Recipe #4."""

import numpy as np

import dynsight

rng = np.random.default_rng(42)  # set the random seed

# Entropy of a discrete variable
n_sample = 10000
rolls = rng.integers(1, 7, size=n_sample)
dice_entropy = dynsight.analysis.compute_shannon(
    data=rolls.astype(float),
    data_range=(1, 6),
    n_bins=6,
    units="bit",
)

# Entropy of a discrete multivariate variable
n_sample = 10000
rolls = rng.integers(1, 7, size=(n_sample, 2))
dices_entropy = dynsight.analysis.compute_shannon_multi(
    data=rolls.astype(float),
    data_ranges=[(1, 6), (1, 6)],
    n_bins=[6, 6],
    units="bit",
)


# Entropy of a continuous variable
n_sample = 10000000
data_1 = rng.normal(loc=0.0, scale=1.0, size=n_sample)
data_2 = rng.normal(loc=0.0, scale=2.0, size=n_sample)

gauss_entropy_1 = dynsight.analysis.compute_kl_entropy(
    data=data_1,
    units="bit",
)
gauss_entropy_2 = dynsight.analysis.compute_kl_entropy(
    data=data_2,
    units="bit",
)
diff = gauss_entropy_2 - gauss_entropy_1


# Entropy of a continuous multivariate variable
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

gauss_entropy_1 = dynsight.analysis.compute_kl_entropy_multi(
    data=data_1,
    units="bit",
)
gauss_entropy_2 = dynsight.analysis.compute_kl_entropy_multi(
    data=data_2,
    units="bit",
)
diff_2d = gauss_entropy_2 - gauss_entropy_1
