from ai.config import logsig, rng, beta


class InhibNRN:
    next_id = 1

    def __init__(self, n_lat_wt):
        """
        :param n_lat_wt: lateral weight count
        """
        self.id_num = InhibNRN.next_id
        self.type = "inhib"
        self.soma_mp = 0.0
        self.dend_mp = 0.0
        self.soma_act = 0.0
        # self.W_IP_lat   = rng.normal(wt_mu, wt_sig, (n_lat_wt,))
        self.W_IP_lat = -rng.exponential(beta, (n_lat_wt,)) if n_lat_wt else None
        InhibNRN.next_id += 1

    def __repr__(self):
        return f"""
    id_num:            {self.id_num},
    type:              {self.type},
    soma_mp:           {self.soma_mp}, 
    dend_mp:           {self.dend_mp},
    soma_activation:   {self.soma_act},
    incoming W_IP_lat: {self.W_IP_lat}
    """

    def update_inhib_soma_forward(self):  # there is no backward soma update for inhib
        self.soma_mp = self.dend_mp
        self.soma_act = logsig(self.soma_mp)
