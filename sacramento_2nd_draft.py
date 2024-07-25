#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 22:01:20 2024. Last revised 24 July 2024.

@author: anthony s maida

Implementation of a small version of model described in:
"Dendritic cortical microcircuits approximate the backpropagation algorithm,"
by Joao Sacramento, Rui Ponte Costa, Yoshua Bengio, and Walter Senn.
In *NeurIPS 2018*, Montreal, Canada.

This particular program studies the learning rules 
presented in that paper in a simplified context.

3 Layer Architecture
Layer 1: input  (2 pyramids, 1 interneron)
Layer 2: hidden (3 pyramics, 3 interneurons)
Layer 3: output (2 pyramids)

pyramical neurons are called "pyrs"
interneurons are called 'inhibs'

Implemented in numpy. The code is not vectorized but the
data structures used closely mimic the neural anatomy given in the paper.
"""

from neuron_layer_classes import Layer # imports numpy, numpy.random

wt_mu, wt_sig = 0.0, 0.1 # initialize wts according to N(u, sig) if using Gaussian.
learning_rate = 0.05

n_input_pyr_nrns  = 2
n_hidden_pyr_nrns = 3
n_output_pyr_nrns = 2

""" Build 3-layer network """
def build_small_three_layer_network():
# Layer 1 is the input layer w/ 2 pyrs and 1 inhib cell.
# No FF connections in input layer. They are postponed to receiving layer.
# Each pyramid projects a FF connection to each of 3 pyrs in Layer 2 (hidden).
# wts are always incoming weights.
    l1 = Layer(n_pyrs = n_input_pyr_nrns, n_inhibs = 1, 
               n_pyr_ff_wt = None, n_IP_lat_wt = 2, n_pyr_fb_wt = 3,
               n_PI_lat_wt = 1)
    print(f"""Building model
    Layer 1:
    ========
    {l1}""")
    
    # Layer 2 is hidden layer w/ 3 pyrs.
    # Also has 3 inhib neurons.
    # Has feedback connections to Layer 1
    l2 = Layer(n_pyrs = n_hidden_pyr_nrns, n_inhibs = 3, 
               n_pyr_ff_wt = 2, n_IP_lat_wt = 3, n_pyr_fb_wt = 2,
               n_PI_lat_wt = 3)
    print(f"""Layer 2:
    ========
    {l2}""")

    # Layer 3 is output layer w/ 2 pyrs. No inhib neurons.
    l3 = Layer(n_pyrs = n_output_pyr_nrns, n_inhibs = 0, 
               n_pyr_ff_wt = 3, n_IP_lat_wt = None, n_pyr_fb_wt = None,
               n_PI_lat_wt = None)
    print(f"""Layer3:
    =======
    {l3}""")
    print("Finished building model.")
    return l1, l2, l3

""" Define training and control funcions to run network """
""" Standard FF sweep with option to print network layer state after each step """
def do_FF_sweep(layer1, layer2, layer3, print_p = True): # the 'p' suffix means predicate
    if print_p:
        print("Starting FF sweep")
    layer1.apply_inputs_to_test_self_predictive_convergence()
    layer1.update_dend_mps_via_IP()
    if print_p:
        print(f"""Layer1_is_given_input_and_updated:
        ===============
        {layer1}""")
    
    layer2.update_pyrs_basal_and_soma_ff(layer1)
    layer2.update_dend_mps_via_IP()
    if print_p:
        print(f"""Layer2_FF_update_finished:
        ===============
        {layer2}""")

    layer3.update_pyrs_basal_and_soma_ff(layer2)
    if print_p:
        print(f"""Layer3_FF_update_finished:
        ===============
        {layer3}""")

""" Standard FB sweep with option to print network layer state after each step """
def do_FB_sweep(layer1, layer2, layer3, print_p = True):
    if print_p:
        print("Starting FB sweep")
    layer2.update_pyrs_apical_soma_fb(layer3)
    if print_p:
        print(f"""Layer2_w_apical_update_fb:
        ===============
        {layer2}""")

    layer1.update_pyrs_apical_soma_fb(layer2)
    if print_p:
        print(f"""Layer1_w_apical_update_fb:
        ===============
        {layer1}""")
    
""" Prints the pyr activations for all layers in the network, starting with the top layer. """
def print_pyr_activations_all_layers_topdown(layer1, layer2, layer3):
    layer3.print_pyr_activations() # before learning
    layer2.print_pyr_activations()
    layer1.print_pyr_activations()
    
def print_FF_and_FB_wts_last_layer(layer2, layer3):
    print("FF wts coming into Layer 3")
    layer3.print_FF_wts()
    print("FB wts coming into Layer 2")
    layer2.print_FB_wts()

""" Learning sweep that only uses Rule 16b """
# Set 'nudge_p = True' to assess supervised learning.
def learn_1_cycle_rule_16b_only(layer1, layer2, layer3, nudge_p = False): # does one learning step
    layer2.adjust_wts_lat_PI()      # adjust lateral PI wts in Layer 2
    layer1.adjust_wts_lat_PI()      # adjust lateral PI wts in Layer 1
    """ Do FF and FB sweeps so wt changes show their effects. """
    do_FF_sweep(layer1, layer2, layer3, print_p = False)
    if nudge_p:
        layer3.nudge_output_layer_neurons(2.0, -2.0, lambda_nudge = 0.8)
    do_FB_sweep(layer1, layer2, layer3, print_p = False)

""" Learning sweep that uses both rules 16b and 13 """
# does one learning step
def learn_1_cycle_rule_16b_and_rule_13(layer1, layer2, layer3, nudge_p = False):
    layer2.adjust_wts_lat_PI()      # adjust lateral PI wts in Layer 2
    #layer2.adjust_wts_lat_IP()      # adjust lateral IP wts in Layer 2
    layer3.adjust_wts_PP_ff(layer2) # adjust FF wts projecting to Layer 3
    layer1.adjust_wts_lat_PI()      # adjust lateral PI wts in Layer 1
    #layer1.adjust_wts_lat_IP()      # adjust lateral IP wts in Layer 1
    layer2.adjust_wts_PP_ff(layer1) # adjust FF wts projecting to Layer 2
    """ Do FF and FB sweeps so wt changes show their effects. """
    do_FF_sweep(layer1, layer2, layer3, print_p = False)
    if nudge_p:
        layer3.nudge_output_layer_neurons(2.0, -2.0, lambda_nudge = 0.8)
    do_FB_sweep(layer1, layer2, layer3, print_p = False)

""" Uses the 16b learning rule to train to a self-predictive state """
# Prints the apical membrane potentials and pyr activations at end
def train_to_self_predictive_state(n_steps = 40):
    do_FF_sweep(layer1, layer2, layer3, print_p = False)
    do_FB_sweep(layer1, layer2, layer3, print_p = False)
    for _ in range(n_steps): 
        learn_1_cycle_rule_16b_only(layer1, layer2, layer3) 
    print(f"Trained to self pred state: {n_steps} steps.")
    layer2.print_apical_mps()
    layer1.print_apical_mps()
    print_pyr_activations_all_layers_topdown(layer1, layer2, layer3)
    print("Above shows network in self predictive state.")

""" Script to run the 1st pilot experiment, version 1a, using only Rule 16b """
# This experiment allows the initialized network to evolve using ONLY rule 16b and
# tracks apical membrane potentials which should converge to zero.
def run_pilot_experiment_1a(layer1, layer2, layer3):
    do_FF_sweep(layer1, layer2, layer3) # prints state
    print("Finished 1st FF sweep: pilot_exp_1")
    do_FB_sweep(layer1, layer2, layer3) # prints state
    print("Finished 1st FB sweep: pilot_exp_1")
    n_of_learning_steps = 150
    print(f"Starting learning {n_of_learning_steps} steps for p_exp 1")
    with open('results_apical_layer2.py', 'w') as file_layer2, open('results_apical_layer1.py', 'w') as file_layer1:
        file_layer2.write("import numpy as np\n")
        file_layer2.write("import matplotlib.pyplot as plt\n")
        file_layer2.write("layer2_apical_2D = np.array([")
        file_layer1.write("import numpy as np\n")
        file_layer1.write("import matplotlib.pyplot as plt\n")
        file_layer1.write("layer1_apical_2D = np.array([")
        for _ in range(n_of_learning_steps): 
            learn_1_cycle_rule_16b_only(layer1, layer2, layer3)                 # <== only 1 learning rule is used.
            # Look at apical mps to see if converging to self-predictive state
            file_layer2.write(f"[{layer2.pyrs[0].apical_mp}, {layer2.pyrs[1].apical_mp}, {layer2.pyrs[2].apical_mp}], \n")
            file_layer1.write(f"[{layer1.pyrs[0].apical_mp}, {layer1.pyrs[1].apical_mp}], \n")
            layer2.print_apical_mps() # now that we are writing to a file, this code isn't necessary
            layer1.print_apical_mps()
        file_layer2.write("])\n")
        file_layer2.write("x_axis = np.arange(layer2_apical_2D.shape[0])\n")
        file_layer2.write("plt.plot(x_axis, layer2_apical_2D[:,0], x_axis, layer2_apical_2D[:,1], x_axis, layer2_apical_2D[:,2])\n")
        file_layer1.write("])\n")
        file_layer1.write("x_axis = np.arange(layer1_apical_2D.shape[0])\n")
        file_layer1.write("plt.plot(x_axis, layer1_apical_2D[:,0],x_axis, layer1_apical_2D[:,1])\n")
    print(f"Finished learning {n_of_learning_steps} steps for p_exp 1")
    print_pyr_activations_all_layers_topdown(layer1, layer2, layer3)

""" Script to run the 1st pilot experiment, version 1b, using Rules 16b and 13 """
# This experiment allows the initialized network to evolve using rules 16b AND 13
# tracks apical membrane potentials which should converge to zero.
def run_pilot_experiment_1b(layer1, layer2, layer3):
    do_FF_sweep(layer1, layer2, layer3) # prints state
    print("Finished 1st FF sweep: pilot_exp_1")
    do_FB_sweep(layer1, layer2, layer3) # prints state
    print("Finished 1st FB sweep: pilot_exp_1")
    n_of_learning_steps = 150
    print(f"Starting learning {n_of_learning_steps} steps for p_exp 1b")
    with open('results_apical_16b_and_13_layer2.py', 'w') as file_layer2, open('results_apical_16b_and_13_layer1.py', 'w') as file_layer1:
        file_layer2.write("import numpy as np\n")
        file_layer2.write("import matplotlib.pyplot as plt\n")
        file_layer2.write("layer2_apical_2D = np.array([")
        file_layer1.write("import numpy as np\n")
        file_layer1.write("import matplotlib.pyplot as plt\n")
        file_layer1.write("layer1_apical_2D = np.array([")
        for _ in range(n_of_learning_steps): 
            learn_1_cycle_rule_16b_and_rule_13(layer1, layer2, layer3)                 # <== uses TWO learning rule is used.
            # Look at apical mps to see if converging to self-predictive state
            file_layer2.write(f"[{layer2.pyrs[0].apical_mp}, {layer2.pyrs[1].apical_mp}, {layer2.pyrs[2].apical_mp}], \n")
            file_layer1.write(f"[{layer1.pyrs[0].apical_mp}, {layer1.pyrs[1].apical_mp}], \n")
            #layer2.print_apical_mps() # now that we are writing to a file, this code isn't necessary
            #layer1.print_apical_mps()
        file_layer2.write("])\n")
        file_layer2.write("x_axis = np.arange(layer2_apical_2D.shape[0])\n")
        file_layer2.write("plt.plot(x_axis, layer2_apical_2D[:,0], x_axis, layer2_apical_2D[:,1], x_axis, layer2_apical_2D[:,2])\n")
        file_layer1.write("])\n")
        file_layer1.write("x_axis = np.arange(layer1_apical_2D.shape[0])\n")
        file_layer1.write("plt.plot(x_axis, layer1_apical_2D[:,0],x_axis, layer1_apical_2D[:,1])\n")
    print(f"Finished learning {n_of_learning_steps} steps for p_exp 1")
    print_pyr_activations_all_layers_topdown(layer1, layer2, layer3)

""" Script to run the 2nd pilot experiment: 2a """
# This experiment checks to see if rule 16b alone with nudging can do effective learning
def run_pilot_experiment_2_rule_16b_only(layer1, layer2, layer3):
    train_to_self_predictive_state() # put network in self-predictive state
    # nudge the 1st neuron (index=0) in layer 3
    layer3.nudge_output_layer_neuron(2.0, -2.0, lambda_nudge = 0.8)
    print("Layer 3 activations after nudge.")    
    layer3.print_pyr_activations()
    print("Starting FB sweep")
    do_FB_sweep(layer1, layer2, layer3) # prints state
    print("Finished 1st FB sweep after nudge: pilot_exp_2") # shows effect of nudge in earlier layers
    print_pyr_activations_all_layers_topdown(layer1, layer2, layer3) # displays in topdown order
    n_of_learning_steps = 40
    print(f"Starting learning {n_of_learning_steps} steps for p_exp 2a")
    for _ in range(n_of_learning_steps): # train with rule 16b and maintain nudging
        learn_1_cycle_rule_16b_only(layer1, layer2, layer3, nudge_p = True) 
        # Check apical mps to see if converging to self-predictive state
        #layer2.print_apical_mps()
        #layer1.print_apical_mps()
    print(f"Finished learning {n_of_learning_steps} steps for p_exp 2a")
    print_pyr_activations_all_layers_topdown(layer1, layer2, layer3) # print activations while nudging is still on
    do_FF_sweep(layer1, layer2, layer3, print_p = False) # to get new activations without nudging
    print("Final activations after nudging is removed")
    print_pyr_activations_all_layers_topdown(layer1, layer2, layer3) # shows the true effect of learning
    # look at data in sheet "Only Rule 16b. No significant learning."

""" Script to run the 2nd pilot experiment: 2b """
# This experiment checks to see if rule 16b and Rule 13 together with nudging can do effective learning
def run_pilot_experiment_2b_rule_16b_and_rule_13(layer1, layer2, layer3):
    train_to_self_predictive_state(150) # put network in self-predictive state
    layer3.nudge_output_layer_neurons(2.0, -2.0, lambda_nudge = 0.8)
    print("Layer 3 activations after nudge.")    
    layer3.print_pyr_activations()
    print("Starting FB sweep")
    do_FB_sweep(layer1, layer2, layer3) # prints state
    print("Finished 1st FB sweep after nudge: pilot_exp_2b") # shows effect of nudge in earlier layers
    print_pyr_activations_all_layers_topdown(layer1, layer2, layer3) # displays in topdown order
    n_of_learning_steps = 200
    print(f"Starting learning {n_of_learning_steps} steps for p_exp 3b")
    with open('results_apical_16b_and_13_layer2_exp_2b.py', 'w') as file_layer2, open('results_apical_16b_and_13_layer1_exp_2b.py', 'w') as file_layer1:
        file_layer2.write("import numpy as np\n")
        file_layer2.write("import matplotlib.pyplot as plt\n")
        file_layer2.write("layer2_apical_2D_exp2b = np.array([")
        file_layer1.write("import numpy as np\n")
        file_layer1.write("import matplotlib.pyplot as plt\n")
        file_layer1.write("layer1_apical_2D_exp_2b = np.array([")
        for _ in range(n_of_learning_steps): # train with rule 16b and maintain nudging
            learn_1_cycle_rule_16b_and_rule_13(layer1, layer2, layer3, nudge_p = True) 
            # Check apical mps to see if converging to self-predictive state
            file_layer2.write(f"[{layer2.pyrs[0].apical_mp}, {layer2.pyrs[1].apical_mp}, {layer2.pyrs[2].apical_mp}], \n")
            file_layer1.write(f"[{layer1.pyrs[0].apical_mp}, {layer1.pyrs[1].apical_mp}], \n")
            #layer2.print_apical_mps()
            #layer1.print_apical_mps()
        file_layer2.write("])\n")
        file_layer2.write("x_axis = np.arange(layer2_apical_2D.shape[0])\n")
        file_layer2.write("plt.plot(x_axis, layer2_apical_2D[:,0], x_axis, layer2_apical_2D[:,1], x_axis, layer2_apical_2D[:,2])\n")
        file_layer1.write("])\n")
        file_layer1.write("x_axis = np.arange(layer1_apical_2D.shape[0])\n")
        file_layer1.write("plt.plot(x_axis, layer1_apical_2D[:,0],x_axis, layer1_apical_2D[:,1])\n")
    print(f"Finished learning {n_of_learning_steps} steps for p_exp 3b")
    print_pyr_activations_all_layers_topdown(layer1, layer2, layer3) # print activations while nudging is still on
    do_FF_sweep(layer1, layer2, layer3, print_p = False) # to get new activations without nudging
    print("Final activations after nudging is removed")
    print_pyr_activations_all_layers_topdown(layer1, layer2, layer3) # shows the true effect of learning
    # look at data in sheet "Only Rule 16b. No significant learning."

def run_pilot_exp_1b_concat_2b(layer1, layer2, layer3, steps_to_self_pred = 150):
    do_FF_sweep(layer1, layer2, layer3) # prints state
    print("Finished 1st FF sweep: pilot_exp_1")
    do_FB_sweep(layer1, layer2, layer3) # prints state
    print("Finished 1st FB sweep: pilot_exp_1")
    n_of_learning_steps_to_self_predictive = steps_to_self_pred
    print(f"Starting learning {n_of_learning_steps_to_self_predictive} steps to 1b self predictive.")
    with open('results_apical_16b_and_13_concat_layer2.py', 'w') as file_layer2, \
         open('results_apical_16b_and_13_concat_layer1.py', 'w') as file_layer1:
        file_layer2.write("import numpy as np\n")
        file_layer2.write("import matplotlib.pyplot as plt\n")
        file_layer2.write("layer2_apical_concat_2D = np.array([")
        file_layer1.write("import numpy as np\n")
        file_layer1.write("import matplotlib.pyplot as plt\n")
        file_layer1.write("layer1_apical_concat_2D = np.array([")
        for _ in range(n_of_learning_steps_to_self_predictive): 
            learn_1_cycle_rule_16b_and_rule_13(layer1, layer2, layer3)                 # <== uses TWO learning rule is used.
            # Look at apical mps to see if converging to self-predictive state
            file_layer2.write(f"[{layer2.pyrs[0].apical_mp}, {layer2.pyrs[1].apical_mp}, {layer2.pyrs[2].apical_mp}], \n")
            file_layer1.write(f"[{layer1.pyrs[0].apical_mp}, {layer1.pyrs[1].apical_mp}], \n")
            #layer2.print_apical_mps() # now that we are writing to a file, this code isn't necessary
            #layer1.print_apical_mps()
        layer3.nudge_output_layer_neurons(2.0, -2.0, lambda_nudge = 0.8)
        print("Layer 3 activations after nudge.")    
        layer3.print_pyr_activations()
        print("Starting FB sweep")
        do_FB_sweep(layer1, layer2, layer3) # prints state
        print("Finished 1st FB sweep after nudge: pilot_exp_2b") # shows effect of nudge in earlier layers
        print_pyr_activations_all_layers_topdown(layer1, layer2, layer3) # displays in topdown order
        n_of_learning_steps = 200
        print(f"Starting learning {n_of_learning_steps} steps for p_exp 3b")
        for _ in range(n_of_learning_steps): # train with rule 16b and maintain nudging
            learn_1_cycle_rule_16b_and_rule_13(layer1, layer2, layer3, nudge_p = True) 
            # Check apical mps to see if converging to self-predictive state
            file_layer2.write(f"[{layer2.pyrs[0].apical_mp}, {layer2.pyrs[1].apical_mp}, {layer2.pyrs[2].apical_mp}], \n")
            file_layer1.write(f"[{layer1.pyrs[0].apical_mp}, {layer1.pyrs[1].apical_mp}], \n")
        file_layer2.write("])\n")
        file_layer2.write("x_axis = np.arange(layer2_apical_concat_2D.shape[0])\n")
        file_layer2.write("plt.plot(x_axis, layer2_apical_concat_2D[:,0], x_axis, layer2_apical_concat_2D[:,1], x_axis, layer2_apical_concat_2D[:,2])\n")
        file_layer1.write("])\n")
        file_layer1.write("x_axis = np.arange(layer1_apical_concat_2D.shape[0])\n")
        file_layer1.write("plt.plot(x_axis, layer1_apical_concat_2D[:,0],x_axis, layer1_apical_concat_2D[:,1])\n")
        print(f"Finished learning {n_of_learning_steps} steps for p_exp 3b")
    print_pyr_activations_all_layers_topdown(layer1, layer2, layer3) # print activations while nudging is still on
    do_FF_sweep(layer1, layer2, layer3, print_p = False) # to get new activations without nudging
    print("Final activations after nudging is removed")
    print_pyr_activations_all_layers_topdown(layer1, layer2, layer3) # shows the true effect of learning
            

    


""" Do an experiment """
layer1, layer2, layer3 = build_small_three_layer_network()           # build the network
print_FF_and_FB_wts_last_layer(layer2, layer3)
#run_pilot_experiment_1a(layer1, layer2, layer3)
#run_pilot_experiment_1b(layer1, layer2, layer3)
#run_pilot_experiment_2_rule_16b_only(layer1, layer2, layer3)
#run_pilot_experiment_2b_rule_16b_and_rule_13(layer1, layer2, layer3) # run the experiment

""" 
    p_Exp 3: steps_to_self_pred = 150.
    p_Exp 3b: steps_to_self_pred = 250.
"""
run_pilot_exp_1b_concat_2b(layer1, layer2, layer3, steps_to_self_pred = 400)
print_FF_and_FB_wts_last_layer(layer2, layer3)

""" End of program """





















