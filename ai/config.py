import numpy as np


def get_rng():
    return np.random.default_rng(seed=wt_init_seed)  # seed needed for reproducibility

def prRed(skk): print("\033[91m {}\033[00m" .format(skk))

wt_init_seed = 42

wt_mu, wt_sig = 0.0, 0.1  # initialize wts according to N(u, sig) if using Gaussian.
beta = 1.0 / 3.0  # beta = 1/lambda => lambda = 3. beta is scale param for rng.exponential.
learning_rate = 0.05

nudge1 = 1.0
nudge2 = 0.0

n_input_pyr_nrns = 2
n_hidden_pyr_nrns = 3
n_output_pyr_nrns = 2

# conductance default values
g_lk = 0.1
g_A = 0.8
g_B = 1.0
g_D = g_B
v_hat_B_P_coeff = 1.0 / (g_lk + g_A + g_B)  # used in adjust_wts_PP_ff()
v_hat_I_coeff = 1.0 / (g_lk + g_D)  # used in


def logsig(x, alpha=1.0):
    """
    logistic sigmoid
    soft ReLU is another possibility (supplementary data p. 3)

    :param x: input value
    :param alpha: affects slope near orig. Recommend alpha >= 1 to avoid shallow grad.
    :return:
    """
    return 1.0 / (1.0 + np.exp(-alpha * x))
