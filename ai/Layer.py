import logging

import numpy as np

from ai.InhibNRN import InhibNRN
from ai.PyrNRN import PyrNRN
from ai.config import learning_rate, logsig

logger = logging.getLogger("ai.Layer")
logger.setLevel(logging.INFO)


# Allocate storage, then allocate objects
# wt_counts are per neuron. Thus, 'pyr_ff_wt_cnt' is the num of ff wts
# per neuron in the layer.
class Layer:
    next_id = 1

    def __init__(self, rng, n_pyrs, n_inhibs, n_pyr_ff_wt, n_ip_lat_wt, n_pyr_fb_wt, n_pi_lat_wt):
        self.id_num = Layer.next_id
        self.pyrs = [PyrNRN(rng, n_pyr_ff_wt, n_pi_lat_wt, n_pyr_fb_wt) for _ in range(n_pyrs)]
        self.inhibs = [InhibNRN(rng, n_ip_lat_wt) for _ in range(n_inhibs)]
        Layer.next_id += 1

    def __repr__(self):
        temp = f"Layer id_num: {self.id_num}\n"

        if len(self.pyrs) > 0:
            temp += "\n".join(repr(p_nrn) for p_nrn in self.pyrs) + "\n"

        if len(self.inhibs) > 0:
            temp += "\n".join(repr(i_nrn) for i_nrn in self.inhibs) + "\n"

        return temp

    def pyr_soma_mps(self):
        return np.array([x.soma_mp for x in self.pyrs])

    def pyr_basal_mps(self):
        return np.array([x.basal_mp for x in self.pyrs])

    def pyr_soma_acts(self):  # retrieve pyramid activations for layer
        """returns array of pyramid soma activations from current layer"""
        return np.array([x.soma_act for x in self.pyrs])

    def pyr_apical_acts(self):  # retrieve pyramid activations for layer
        """returns array of pyramid apical activations from current layer"""
        return np.array([x.apical_act for x in self.pyrs])

    def pyr_basal_hat_acts(self):
        """returns array of pyramid basal predicted activations from current layer"""
        return np.array([x.basal_hat_act for x in self.pyrs])

    def inhib_soma_acts(self):  # retrieve pyramid activations for layer
        """returns array of inhib activations from current layer"""
        return np.array([x.soma_act for x in self.inhibs])

    def inhib_dend_mps(self):
        """returns array of dendritic membrane potentials"""
        return np.array([x.dend_mp for x in self.inhibs])

    ######################################
    # FF and FB neuron update mechanisms #
    ######################################

    def apply_inputs_to_test_self_predictive_convergence(self):
        """apply uniform inputs of 0.5 to get the system running"""
        for i in range(len(self.pyrs)):
            self.pyrs[i].basal_mp = 0.5
            self.pyrs[i].update_pyr_soma_ff()  # updates soma mp and activation

    def update_dend_mps_via_ip(self):  # update dendritic membrane potentials
        """update inhibs via lateral IP wts"""
        temp = self.pyr_soma_acts()  # for current layer
        for i in range(len(self.inhibs)):
            self.inhibs[i].dend_mp = np.dot(self.inhibs[i].W_IP_lat, temp)
            self.inhibs[i].update_inhib_soma_ff()

    def update_pyrs_basal_and_soma_ff(self, prev_layer):
        """update pyrs bottom up from prev layer"""
        temp = prev_layer.pyr_soma_acts()  # get activations from prev layer
        for i in range(len(self.pyrs)):  # update each pyramid in current layer
            self.pyrs[i].basal_mp = np.dot(self.pyrs[i].W_PP_ff, temp)
            self.pyrs[i].update_pyr_soma_ff()

    def update_pyrs_apical_soma_fb(self, higher_layer):
        """propagates input from apical dends to soma"""
        fb_acts = higher_layer.pyr_soma_acts()
        inhib_acts = self.inhib_soma_acts()
        for i in range(len(self.pyrs)):
            self.pyrs[i].apical_mp = np.dot(self.pyrs[i].W_PP_fb, fb_acts) + np.dot(self.pyrs[i].W_PI_lat, inhib_acts)
            self.pyrs[i].apical_act = logsig(self.pyrs[i].apical_mp)
            self.pyrs[i].apical_hat = 0.5 * self.pyrs[i].apical_mp  # approximation to Eqn (14)
            self.pyrs[i].apical_hat_act = logsig(self.pyrs[i].apical_hat)
            self.pyrs[i].soma_mp = ((self.pyrs[i].basal_mp - self.pyrs[i].soma_mp)
                                    + (self.pyrs[i].apical_mp - self.pyrs[i].soma_mp))
            self.pyrs[i].soma_act = logsig(self.pyrs[i].soma_mp)

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

    #############
    # print wts #
    #############

    def print_ff_wts(self):
        for nrn in self.pyrs:
            logger.info(nrn.W_PP_ff)

    def print_fb_wts(self):
        for nrn in self.pyrs:
            logger.info(nrn.W_PP_fb)

    def print_ip_wts(self):
        for nrn in self.inhibs:
            logger.info(nrn.W_IP_lat)

    def print_pi_wts(self):
        for nrn in self.pyrs:
            logger.info(nrn.W_PI_lat)

    def print_fb_and_pi_wts_layer(self):
        logger.info(f"FB wts coming into Layer {self.id_num}")
        self.print_fb_wts()
        logger.info(f"PI wts within Layer {self.id_num}")
        self.print_pi_wts()

    def print_ff_and_ip_wts_for_layers(self, l_k_plus_1: "Layer"):
        logger.info(f"FF wts coming into Layer {l_k_plus_1.id_num}")
        l_k_plus_1.print_ff_wts()
        logger.info(f"IP wts within Layer {self.id_num}")
        self.print_ip_wts()

    ###########################################################################
    #                           Three learning rules                          #
    # The rules use doubly nested for loops to iterate over the pre- and      #
    # postsynaptic neurons to adjust the weights connecting them.             #
    # Vectorization of this code would require moving weight storage from the #
    # individual neurons to the layer.                                        #
    ###########################################################################

    def adjust_wts_lat_pi(self):  # adjust PI wts for layer. Eqn 16b
        for i in range(len(self.pyrs)):
            for j in range(len(self.inhibs)):  # V_rest is not include below b/c its value is zero.
                self.pyrs[i].W_PI_lat[j] -= learning_rate * self.pyrs[i].apical_mp * self.inhibs[j].soma_act
                # change for wt projecting to pyr i from inhib j.

    def adjust_wts_pp_ff(self, prev_layer):  # Adjust FF wts for layer. Eqn 13
        """This rule uses the basal predicted activation, NOT the apical."""
        pre = prev_layer.pyr_soma_acts()  # Need activations from prev layer.
        # post_a        = self.pyr_soma_acts()
        post_soma_mp = self.pyr_soma_mps()
        # post_b        = self.pyr_basal_hat_acts()
        post_basal_mp = self.pyr_basal_mps()
        post = post_soma_mp - post_basal_mp  # use unprocessed raw mps.
        # post          = post_a - post_b
        # post         = self.pyr_soma_acts() - self.pyr_basal_hat_acts() # original
        trigger_data_pt = np.array([[self.pyr_soma_acts()[0],
                             self.pyr_basal_hat_acts()[0],
                             post[0],
                             post_soma_mp[0],
                             post_basal_mp[0]]])
        for i in range(post.size):
            for j in range(pre.size):
                self.pyrs[i].W_PP_ff[j] += learning_rate * post[i] * pre[j]
        wt_data_pt = np.array([[self.pyrs[0].W_PP_ff[0]]])
        return trigger_data_pt, wt_data_pt

    def adjust_wts_lat_ip(self):  # Equation 16a
        pre = self.pyr_soma_acts()
        post = self.inhib_soma_acts() - self.inhib_dend_mps()
        for i in range(post.size):
            for j in range(pre.size):
                self.inhibs[i].W_IP_lat[j] += learning_rate * post[i] * pre[j]

    ########################
    # Printing information #
    ########################

    def print_apical_mps(self):  # need to know if these are converging to 0 to assess self-predictive state
        """print the apical membrane potentials for the layer"""
        logger.info(f"Layer {self.id_num}")
        for i in range(len(self.pyrs)):
            logger.info(f"apical_mp: {self.pyrs[i].apical_mp}")

    def print_pyr_activations(self):
        """print the pyramical activation levels for a layer"""
        logger.info(f"Layer {self.id_num} pyr activations")
        for i in range(len(self.pyrs)):
            logger.info(f"soma act: {self.pyrs[i].soma_act}")
