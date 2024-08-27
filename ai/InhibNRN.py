from ai.utils import logsig


class InhibNRN:
    next_id = 1

    def __init__(self, rng, beta: float, n_lat_wt: int):
        """
        :param n_lat_wt: lateral weight count
        """
        self.id_num = InhibNRN.next_id
        self.type = "inhib"
        self.soma_mp = 0.0
        self.dend_mp = 0.0
        self.dend_hat_mp = 0.0
        self.soma_act = 0.0
        self.dend_hat_mp_act = 0.0
        self.wtd_input_from_nudge = 0.0
        # self.W_IP_lat = rng.normal(wt_mu, wt_sig, (n_lat_wt,))  # replace Gaussian w/ exponential
        self.W_IP_lat = -rng.exponential(beta, (n_lat_wt,)) if n_lat_wt else None
        InhibNRN.next_id += 1

    def __repr__(self):
        return f"""
    id_num:            {self.id_num},
    type:              {self.type},
    soma_mp:           {self.soma_mp}, 
    dend_mp:           {self.dend_mp},
    soma_activation:   {self.soma_act},
    incoming W_IP_lat: {self.W_IP_lat}"""

    def update_inhib_soma_ff(self):  # no backward soma update for inhib unless inhib receives fb connection
        #  self.dend_mp is updated in FF sweep via "__do_ff_sweep()"
        self.dend_hat_mp = 0.909 * self.dend_mp  # assumes g_lk = 0.1. See Eqn 15.
        self.soma_mp = self.dend_mp
        self.soma_act = logsig(self.soma_mp)
        self.dend_hat_mp_act = logsig(self.dend_hat_mp)
