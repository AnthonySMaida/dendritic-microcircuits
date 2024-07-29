import numpy as np


def get_rng(seed=42):
    return np.random.default_rng(seed=seed)  # seed needed for reproducibility


beta = 1.0 / 3.0  # beta = 1/lambda => lambda = 3. beta is scale param for rng.exponential.
learning_rate = 0.05

wt_mu, wt_sig = 0.0, 0.1  # initialize wts according to N(u, sig) if using Gaussian.

n_input_pyr_nrns = 2
n_hidden_pyr_nrns = 3
n_output_pyr_nrns = 2


def logsig(x, alpha=1.0):
    """
    logistic sigmoid
    soft ReLU is another possibility (supplementary data p. 3)

    :param x: input value
    :param alpha: affects slope near orig. Recommend alpha >= 1 to avoid shallow grad.
    :return:
    """
    return 1.0 / (1.0 + np.exp(-alpha * x))
