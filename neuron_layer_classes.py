#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 12:45:40 2024

@author: maida

Defines classes: Pyr_nrn, Inhib_nrn, Layer
"""

import numpy as np
import numpy.random

wt_init_seed = 42

rng = np.random.default_rng(seed = wt_init_seed) # needed for reproducibility
beta = 1.0/3.0           # beta = 1/lambda => lambda = 3. beta is scale param for rng.exponential.
learning_rate = 0.05

# conductance default values
g_lk = 0.1
g_A = 0.8
g_B = 1.0
g_D = g_B
v_hat_B_P_coeff = 1.0 / (g_lk + g_A + g_B) # used in adjust_wts_PP_ff()
v_hat_I_coeff   = 1.0 / (g_lk + g_D)       # used in 

# alpha affects slope near orig. Recommend alpha >= 1 to avoid shallow grad.
# soft ReLU is another possibility (supplementary data p. 3)
def logsig(x, alpha = 1.0): # logistic sigmoid
    return 1.0 / (1.0 + np.exp(-alpha*x))

""" class Pyr_nrn"""
# Params: 
#    n_ff_wt   : num of incoming feedforward wts
#    n_lat_wt  : num of incoming lateral wts
#    n_fb_wt   : num of incoming feedback wts
# Methods:
#    update_pyr_soma_FF()
class Pyr_nrn:
    next_id: np.int32 = 1    # You can declare data types.
    def __init__(self, n_ff_wt, n_PI_lat_wt, n_fb_wt):  # wt_counts
        self.id_num: np.int        = Pyr_nrn.next_id
        self.type = "pyr"
        self.soma_mp         = 0.0  # somatic membrane potential
        self.apical_mp       = 0.0  # apical membrane potential
        self.basal_mp        = 0.0  # basal membrane potential
        self.basal_hat       = 0.0  # predicted basal membrane potential
        self.basal_hat_act   = 0.0  # predicted basal activation   
        self.soma_act        = 0.0  # activation value for soma
        self.apical_act      = 0.0  # activation for apical dendrite
#        self.apical_hat      = 0.0  # predictied apical membrane potential
#        self.apical_hat_act  = 0.0  # used in W_PP_ff learning rule
        # Below: feedforward wts
        #self.W_PP_ff     = rng.normal(wt_mu, wt_sig, (n_ff_wt,))     if n_ff_wt else None
        self.W_PP_ff      = rng.exponential(beta, (n_ff_wt,))          if n_ff_wt else None
        # Below: wts coming in from inhib 1, 2 ... but 0-based indexing.
        #self.W_PI_lat    = rng.normal(wt_mu, wt_sig, (n_PI_lat_wt,)) if n_PI_lat_wt else None
        self.W_PI_lat     = -rng.exponential(beta, (n_PI_lat_wt,))     if n_PI_lat_wt else None
        # Below: feedback wts
        #self.W_PP_fb     = rng.normal(wt_mu, wt_sig, (n_fb_wt,))     if n_fb_wt else None                
        self.W_PP_fb      = rng.exponential(beta, (n_fb_wt,))          if n_fb_wt else None                
        Pyr_nrn.next_id += 1
        
    def __repr__(self): # Uses f string (format string)
        return f"""
    id_num:            {self.id_num},
    type:              {self.type},
    soma_mp:           {self.soma_mp}, 
    apical_mp:         {self.apical_mp},
    basal_mp:          {self.basal_mp},
    basal_hat:         {self.basal_hat}
    basal_hat_act:     {self.basal_hat_act}
    soma_activation:   {self.soma_act},
    apical_act:        {self.apical_act}
    incoming W_PP_ff:  {self.W_PP_ff},
    incoming W_PP_fb:  {self.W_PP_fb},
    incoming W_PI_lat: {self.W_PI_lat}
    """
    
    # simplified update. See Section 5.3 of paper, point 1.
    def update_pyr_soma_FF(self):
        self.soma_mp       = self.basal_mp         # start from basal_mp
        self.basal_hat     = v_hat_B_P_coeff * self.basal_mp
        self.basal_hat_act = logsig(self.basal_hat)
        self.soma_act      = logsig(self.soma_mp)  # propagate to soma

""" class Inhib_nrn"""
# Params:
#    n_lat_wt: lateral weight count
# Methods:
# update_inhib_soma_FF()
class Inhib_nrn:
    next_id = 1
    def __init__(self, n_lat_wt):
        self.id_num = Inhib_nrn.next_id
        self.type = "inhib"
        self.soma_mp         = 0.0
        self.dend_mp         = 0.0
        self.dend_hat_mp     = 0.0
        self.soma_act        = 0.0
        self.dend_hat_mp_act = 0.0
        #self.W_IP_lat       = rng.normal(wt_mu, wt_sig, (n_lat_wt,))
        self.W_IP_lat        = -rng.exponential(beta, (n_lat_wt,)) if n_lat_wt else None
        Inhib_nrn.next_id += 1
        
    def __repr__(self):
        return f"""
    id_num:            {self.id_num},
    type:              {self.type},
    soma_mp:           {self.soma_mp}, 
    dend_mp:           {self.dend_mp},
    soma_activation:   {self.soma_act},
    incoming W_IP_lat: {self.W_IP_lat}
    """
    
    def update_inhib_soma_FF(self): # there is no backward soma update for inhib
        self.soma_mp  = self.dend_mp
        self.soma_act = logsig(self.soma_mp)

""" class Layer"""
# Allocate storage, then allocate objects
# wt_counts are are per neuron. Thus, 'pyr_ff_wt_cnt' is the num of ff wts 
# per neuron in the layer.
# Methods:
#     apply_inputs_to_test_self_predictive_convergence()
#     pyr_soma_acts()
#     pyr_apical_acts()
#     inhib_soma_acts()
#     inhib_dend_mps()
#     update_dend_mps_via_IP()
#     nudge_output_layer_neuron(index, target_value = 0.9, lambda_nudge = 0.7)
#     update_pyrs_basal_and_soma_ff(prev_layer)
#     update_pyrs_apical_soma_fb(higher_layer)
#     update_pyrs_soma_fb()
#     adjust_wts_lat_PI()
#     adjust_wts_PP_ff(prev_layer)
#     adjust_wts_lat_IP()
#     print_apical_mps()
#     print_pyr_activations()
class Layer:
    next_id = 1
    def __init__(self, n_pyrs = None, n_inhibs = None, n_pyr_ff_wt= None, 
                 n_IP_lat_wt = None, n_pyr_fb_wt = None, n_PI_lat_wt = None):
        self.id_num   = Layer.next_id
        self.pyrs = np.arange(n_pyrs, dtype = object)
        for i in range(n_pyrs):
            self.pyrs[i] = Pyr_nrn(n_pyr_ff_wt, n_PI_lat_wt, n_pyr_fb_wt)
        self.inhibs = np.arange(n_inhibs, dtype=object)
        for i in range(n_inhibs):
            self.inhibs[i] = Inhib_nrn(n_IP_lat_wt)
        Layer.next_id += 1
            
    def __repr__(self):
        pyr_str = f"Layer id_num: {self.id_num}" + "\n"
        if self.pyrs.size > 0:
            for p_nrn in np.nditer(self.pyrs, flags = ['refs_ok'], op_flags = ['readonly']):
                pyr_str += p_nrn.__repr__() + "\n"
        inhib_str = ""
        if self.inhibs.size > 0:
            for i_nrn in np.nditer(self.inhibs, flags = ['refs_ok'], op_flags = ['readonly']):
                inhib_str += i_nrn.__repr__() + "\n"
        return pyr_str + inhib_str
    
    def pyr_soma_mps(self):
        return np.array([x.soma_mp for x in self.pyrs])
    
    def pyr_basal_mps(self):
        return np.array([x.basal_mp for x in self.pyrs])
    
    # returns array of pyramid soma activations from current layer
    def pyr_soma_acts(self): # retrieve pyramid activations for layer
        return np.array([x.soma_act for x in self.pyrs])

    # returns array of pyramid apical activations from current layer
    def pyr_apical_acts(self): # retrieve pyramid activations for layer
        return np.array([x.apical_act for x in self.pyrs])
    
    # returns array of pyramid basal predicted activations from current layer
    def pyr_basal_hat_acts(self):
        return np.array([x.basal_hat_act for x in self.pyrs])
    
    # returns array of inhib activations from current layer
    def inhib_soma_acts(self): # retrieve pyramid activations for layer
        return np.array([x.soma_act for x in self.inhibs])
    
    def inhib_dend_mps(self): # returns array of dendritic membrane potentials
        return np.array([x.dend_mp for x in self.inhibs])
    
    """ FF and FB neuron update mechanisms """
    # apply uniform inputs of 0.5 to get the system running
    def apply_inputs_to_test_self_predictive_convergence(self):
        for i in range(self.pyrs.size):
            self.pyrs[i].basal_mp = 0.5
            self.pyrs[i].update_pyr_soma_FF() # updates soma mp and activation
    
    # update inhibs via lateral IP wts
    def update_dend_mps_via_IP(self): # update dendritic membrane potentials
        temp = self.pyr_soma_acts() # for current layer
        for i in range(self.inhibs.size):
            self.inhibs[i].dend_mp = np.dot(self.inhibs[i].W_IP_lat, temp)
            self.inhibs[i].update_inhib_soma_FF()
    
    # update pyrs bottom up from prev layer
    def update_pyrs_basal_and_soma_FF(self, prev_layer):
        temp = prev_layer.pyr_soma_acts() # get activations from prev layer
        for i in range(self.pyrs.size):          # update each pyramid in current layer
            self.pyrs[i].basal_mp = np.dot(self.pyrs[i].W_PP_ff, temp)
            self.pyrs[i].update_pyr_soma_FF()
    
    # propagates input from apical dends to soma
    def update_pyrs_apical_soma_FB(self, higher_layer):
        fb_acts    = higher_layer.pyr_soma_acts()
        inhib_acts = self.inhib_soma_acts()
        for i in range(self.pyrs.size):
            self.pyrs[i].apical_mp      = np.dot(self.pyrs[i].W_PP_fb, fb_acts) + np.dot(self.pyrs[i].W_PI_lat, inhib_acts)
            self.pyrs[i].apical_act     = logsig(self.pyrs[i].apical_mp)
            self.pyrs[i].apical_hat     = 0.5 * self.pyrs[i].apical_mp # approximation to Eqn (14)
            self.pyrs[i].apical_hat_act = logsig(self.pyrs[i].apical_hat)
            self.pyrs[i].soma_mp        = (self.pyrs[i].basal_mp - self.pyrs[i].soma_mp) + (self.pyrs[i].apical_mp - self.pyrs[i].soma_mp)
            self.pyrs[i].soma_act       = logsig(self.pyrs[i].soma_mp)
    
    """ Output layer nudging """
    # Assumes there are two pyr neurons in the output layer. 
    # Nudges the somatic membrane potential of both neurons and then updates their activations.
    def nudge_output_layer_neurons(self, targ_val1, targ_val2, lambda_nudge = 0.8):
        self.pyrs[0].soma_mp   = (1 - lambda_nudge) * self.pyrs[0].basal_mp + lambda_nudge * targ_val1
        self.pyrs[0].soma_act  = logsig(self.pyrs[0].soma_mp)
        self.pyrs[1].soma_mp   = (1 - lambda_nudge) * self.pyrs[1].basal_mp + lambda_nudge * targ_val2
        self.pyrs[1].soma_act  = logsig(self.pyrs[1].soma_mp)

    """ print wts """
    def print_FF_wts(self):
        for nrn in self.pyrs:
            print(nrn.W_PP_ff)
            
    def print_FB_wts(self):
        for nrn in self.pyrs:
            print(nrn.W_PP_fb)

    """ Three learning rules:
        The rules use doubly nested for loops to iterate over the pre- and
        postsynaptic neurons to adjust the weights connecting them.
        Vectorization of this code would require moving weight storage from the 
        individual neurons to the layer. """
    def adjust_wts_lat_PI(self): # adjust PI wts for layer. Eqn 16b
        for i in range(self.pyrs.size):
            for j in range(self.inhibs.size): # V_rest is not include below b/c its value is zero.
                self.pyrs[i].W_PI_lat[j] -= learning_rate * self.pyrs[i].apical_mp * self.inhibs[j].soma_act
                # change for wt projecting to pyr i from inhib j.
        
    # This rule uses the basal predicted activation, NOT the apical.
    def adjust_wts_PP_FF(self, prev_layer): # Adjust FF wts for layer. Eqn 13
        pre           = prev_layer.pyr_soma_acts()   # Need activations from prev layer.
        #post_a        = self.pyr_soma_acts()
        post_soma_mp  = self.pyr_soma_mps()
        #post_b        = self.pyr_basal_hat_acts()
        post_basal_mp = self.pyr_basal_mps()
        post          = post_soma_mp - post_basal_mp # use unprocessed raw mps.
        #post          = post_a - post_b
        #post         = self.pyr_soma_acts() - self.pyr_basal_hat_acts() # original
        data_pt = np.array([[self.pyr_soma_acts()[0],
                             self.pyr_basal_hat_acts()[0], 
                             post[0],
                             post_soma_mp[0],
                             post_basal_mp[0]]])
        for i in range(post.size):
            for j in range(pre.size):
                self.pyrs[i].W_PP_ff[j] += learning_rate * post[i] * pre[j]
        return data_pt

    def adjust_wts_lat_IP(self): # Equation 16a
        pre  = self.pyr_soma_acts()
        post = self.inhib_soma_acts() - self.inhib_dend_mps() # 
        for i in range(post.size):
            for j in range(pre.size):
                self.inhibs[i].W_IP_lat[j] += learning_rate * post[i] * pre[j]
                
    """ Printing information """
    # print the apical membrane potentials for the layer
    def print_apical_mps(self): # need to know if these are convering to 0 to assess self-predictive state
        print(f"Layer {self.id_num}")
        for i in range(self.pyrs.size):
            print(f"apical_mp: {self.pyrs[i].apical_mp}")
    
    # print the pyramical activation levels for a layer
    def print_pyr_activations(self):
        print(f"Layer {self.id_num} pyr activations")
        for i in range(self.pyrs.size):
            print(f"soma act: {self.pyrs[i].soma_act}")
                
