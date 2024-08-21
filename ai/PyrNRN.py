import numpy as np

from ai.utils import logsig


class PyrNRN:
    next_id: np.int32 = 1  # You can declare data types.

    def __init__(self, rng, beta: float, n_ff_wt: int, n_pi_lat_wt: int, n_fb_wt: int):  # wt_counts
        """
        :param n_ff_wt: num of incoming feedforward wts
        :param n_pi_lat_wt: num of incoming lateral wts
        :param n_fb_wt: num of incoming feedback wts
        """
        self.id_num = PyrNRN.next_id
        self.type = "pyr"
        self.soma_mp = 0.0  # somatic membrane potential
        self.apical_mp = 0.0  # apical membrane potential
        self.apical_fb = 0.0   # for debugging
        self.apical_lat = 0.0  # for debugging
        self.basal_mp = 0.0  # basal membrane potential
        self.basal_hat = 0.0  # predicted basal membrane potential
        self.basal_hat_act = 0.0  # predicted basal activation
        self.soma_act = 0.0  # activation value for soma
        self.apical_act = 0.0  # activation for apical dendrite
        self.apical_hat = 0.0  # predicted apical membrane potential
        self.apical_hat_act = 0.0  # used in W_PP_ff learning rule
        # Below: feedforward wts
        # self.W_PP_ff     = rng.normal(wt_mu, wt_sig, (n_ff_wt,))     if n_ff_wt else None
        self.W_PP_ff = rng.exponential(beta, (n_ff_wt,)) if n_ff_wt else None
        # Below: wts coming in from inhib 1, 2 ... but 0-based indexing.
        # self.W_PI_lat    = rng.normal(wt_mu, wt_sig, (n_PI_lat_wt,)) if n_PI_lat_wt else None
        self.W_PI_lat = -rng.exponential(beta, (n_pi_lat_wt,)) if n_pi_lat_wt else None
        # Below: feedback wts
        # self.W_PP_fb     = rng.normal(wt_mu, wt_sig, (n_fb_wt,))     if n_fb_wt else None
        self.W_PP_fb = rng.exponential(beta, (n_fb_wt,)) if n_fb_wt else None
        PyrNRN.next_id += 1

    def __repr__(self):  # Uses f string (format string)
        return f"""
    id_num:            {self.id_num},
    type:              {self.type},
    soma_mp:           {self.soma_mp}, 
    apical_mp:         {self.apical_mp},
    basal_mp:          {self.basal_mp},
    basal_hat:         {self.basal_hat},
    basal_hat_act:     {self.basal_hat_act},
    soma_activation:   {self.soma_act},
    apical_act:        {self.apical_act},
    incoming W_PP_ff:  {self.W_PP_ff},
    incoming W_PP_fb:  {self.W_PP_fb},
    incoming W_PI_lat: {self.W_PI_lat}"""

    # simplified update. See Section 5.3 of paper, point 1.
    def update_pyr_soma_ff(self):
        self.soma_mp = self.basal_mp  # start from basal_mp
        # 1.0 / (g_lk + g_A + g_B) = 1.0 / (0.1 + 0.8 + 1.0) = 1.0 / 1.9 = 0.5263157894736842
        self.basal_hat = .5263157894736842 * self.basal_mp
        self.basal_hat_act = logsig(self.basal_hat)
        self.soma_act = logsig(self.soma_mp)  # propagate to soma
