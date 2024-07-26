import numpy as np

from ai.InhibNRN import InhibNRN
from ai.PyrNRN import PyrNRN
from ai.config import learning_rate, logsig


# Allocate storage, then allocate objects
# wt_counts are are per neuron. Thus, 'pyr_ff_wt_cnt' is the num of ff wts
# per neuron in the layer.
class Layer:
    next_id = 1

    def __init__(self, n_pyrs, n_inhibs, n_pyr_ff_wt, n_ip_lat_wt, n_pyr_fb_wt, n_pi_lat_wt):
        self.id_num = Layer.next_id
        self.pyrs = [PyrNRN(n_pyr_ff_wt, n_pi_lat_wt, n_pyr_fb_wt) for _ in range(n_pyrs)]
        self.inhibs = [InhibNRN(n_ip_lat_wt) for _ in range(n_inhibs)]
        Layer.next_id += 1

    def __repr__(self):
        temp = f"Layer id_num: {self.id_num}\n"

        if len(self.pyrs) > 0:
            temp += "\n".join(repr(p_nrn) for p_nrn in self.pyrs) + "\n"

        if len(self.inhibs) > 0:
            temp += "\n".join(repr(i_nrn) for i_nrn in self.inhibs) + "\n"

        return temp

    def apply_inputs_to_test_self_predictive_convergence(self):
        """apply uniform inputs of 0.5 to get the system running"""
        for i in range(len(self.pyrs)):
            self.pyrs[i].basal_mp = 0.5
            self.pyrs[i].update_pyr_soma_forward()

    def pyr_soma_activations(self):  # retrieve pyramid activations for layer
        """returns array of pyramid activations from current layer"""
        return np.array([x.soma_act for x in self.pyrs])

    def pyr_apical_activations(self):  # retrieve pyramid activations for layer
        """returns array of pyramid activations from current layer"""
        return np.array([x.apical_act for x in self.pyrs])

    def inhib_soma_activations(self):  # retrieve pyramid activations for layer
        """returns array of inhib activations from current layer"""
        return np.array([x.soma_act for x in self.inhibs])

    def inhib_dend_mps(self):
        """returns array of dendritic membrane potentials"""
        return np.array([x.dend_mp for x in self.inhibs])

    def update_dend_mps_via_ip(self):  # update dendritic membrane potentials
        """update inhibs via lateral IP wts"""
        temp = self.pyr_soma_activations()  # for current layer
        for i in range(len(self.inhibs)):
            self.inhibs[i].dend_mp = np.dot(self.inhibs[i].W_IP_lat, temp)
            self.inhibs[i].update_inhib_soma_forward()

    def print_ff_wts(self):
        for nrn in self.pyrs:
            print(nrn.W_PP_ff)

    def print_fb_wts(self):
        for nrn in self.pyrs:
            print(nrn.W_PP_fb)

    ########################
    # Output layer nudging #
    ########################

    def nudge_output_layer_neurons(self, targ_val1, targ_val2, lambda_nudge=0.8):
        """
        Assumes there are two pyr neurons in the output layer.
        Nudges the somatic membrane potential of both neurons and then updates their activations.
        """
        self.pyrs[0].soma_mp = (1 - lambda_nudge) * self.pyrs[0].basal_mp + lambda_nudge * targ_val1
        self.pyrs[0].soma_act = logsig(self.pyrs[0].soma_mp)
        self.pyrs[1].soma_mp = (1 - lambda_nudge) * self.pyrs[1].basal_mp + lambda_nudge * targ_val2
        self.pyrs[1].soma_act = logsig(self.pyrs[1].soma_mp)

    def update_pyrs_basal_and_soma_ff(self, prev_layer):
        """update pyrs bottom up from prev layer"""
        temp = prev_layer.pyr_soma_activations()  # get activations from prev layer
        for i in range(len(self.pyrs)):  # update each pyramid in current layer
            self.pyrs[i].basal_mp = np.dot(self.pyrs[i].W_PP_ff, temp)
            self.pyrs[i].update_pyr_soma_forward()

    def update_pyrs_apical_soma_fb(self, higher_layer):
        fb_acts = higher_layer.pyr_soma_activations()
        inhib_acts = self.inhib_soma_activations()
        for i in range(len(self.pyrs)):
            self.pyrs[i].apical_mp = np.dot(self.pyrs[i].W_PP_fb, fb_acts) + np.dot(self.pyrs[i].W_PI_lat, inhib_acts)
            self.pyrs[i].apical_act = logsig(self.pyrs[i].apical_mp)
            self.pyrs[i].apical_pred = 0.5 * self.pyrs[i].apical_mp  # approximation to Eqn (14)
            self.pyrs[i].apical_pred_act = logsig(self.pyrs[i].apical_pred)
            self.pyrs[i].soma_mp = (self.pyrs[i].basal_mp - self.pyrs[i].soma_mp) + (
                    self.pyrs[i].apical_mp - self.pyrs[i].soma_mp)
            self.pyrs[i].soma_act = logsig(self.pyrs[i].soma_mp)

    ########################
    # Three learning rules #
    ########################

    def adjust_wts_lat_pi(self):  # adjust PI wts for layer. Equation 16b
        for i in range(len(self.pyrs)):
            for j in range(len(self.inhibs)):  # V_rest is not include below b/c its value is zero.
                self.pyrs[i].W_PI_lat[j] -= learning_rate * self.pyrs[i].apical_mp * self.inhibs[j].soma_act
                # change for wt projecting to pyr i from inhib j.

    def adjust_wts_pp_ff(self, prev_layer):  # Adjust FF wts for layer. Equation 13
        pre_temp = prev_layer.pyr_soma_activations()  # Need activations from prev layer.
        post_temp = self.pyr_soma_activations() - self.pyr_apical_activations()
        for i in range(post_temp.size):
            for j in range(pre_temp.size):
                self.pyrs[i].W_PP_ff[j] += learning_rate * post_temp[i] * pre_temp[j]

    def adjust_wts_lat_ip(self):  # Equation 16a
        pre_temp = self.pyr_soma_activations()
        post_temp = self.inhib_soma_activations() - self.inhib_dend_mps()
        for i in range(post_temp.size):
            for j in range(pre_temp.size):
                self.inhibs[i].W_IP_lat[j] += learning_rate * post_temp[i] * pre_temp[j]

    ########################
    # Printing information #
    ########################

    def print_apical_mps(self):  # need to know if these are converging to 0 to assess self-predictive state
        """print the apical membrane potentials for the layer"""
        print(f"Layer {self.id_num}")
        for i in range(len(self.pyrs)):
            print(f"apical_mp: {self.pyrs[i].apical_mp}")

    def print_pyr_activations(self):
        """print the pyramical activation levels for a layer"""
        print(f"Layer {self.id_num} pyr activations")
        for i in range(len(self.pyrs)):
            print(f"soma act: {self.pyrs[i].soma_act}")
