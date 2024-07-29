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
from base64 import b64encode
from io import BytesIO

import numpy as np
from matplotlib import pyplot as plt

from ai.Layer import Layer
from ai.config import get_rng, n_input_pyr_nrns, n_hidden_pyr_nrns, n_output_pyr_nrns


def build_small_three_layer_network():
    """Build 3-layer network"""
    rng = get_rng()
    # Layer 1 is the input layer w/ 2 pyrs and 1 inhib cell.
    # No FF connections in input layer. They are postponed to receiving layer.
    # Each pyramid projects a FF connection to each of 3 pyrs in Layer 2 (hidden).
    # wts are always incoming weights.
    l1 = Layer(rng, n_input_pyr_nrns, 1, None, 2, 3, 1)
    print(f"""Building model
    Layer 1:
    ========
    {l1}""")

    # Layer 2 is hidden layer w/ 3 pyrs.
    # Also has 3 inhib neurons.
    # Has feedback connections to Layer 1
    l2 = Layer(rng, n_hidden_pyr_nrns, 3, 2, 3, 2, 3)
    print(f"""Layer 2:
    ========
    {l2}""")

    # Layer 3 is output layer w/ 2 pyrs. No inhib neurons.
    l3 = Layer(rng, n_output_pyr_nrns, 0, 3, None, None, None)
    print(f"""Layer3:
    =======
    {l3}""")
    print("Finished building model.")
    return l1, l2, l3


def do_ff_sweep(layer1: Layer, layer2: Layer, layer3: Layer, print_predicate=True):
    """
    Define training and control funcions to run network
    Standard FF sweep with option to print network layer state after each step
    """
    if print_predicate:
        print("Starting FF sweep")
    layer1.apply_inputs_to_test_self_predictive_convergence()
    layer1.update_dend_mps_via_ip()
    if print_predicate:
        print(f"""Layer1_is_given_input_and_updated:
        ===============
        {layer1}""")

    layer2.update_pyrs_basal_and_soma_ff(layer1)
    layer2.update_dend_mps_via_ip()
    if print_predicate:
        print(f"""Layer2_FF_update_finished:
        ===============
        {layer2}""")

    layer3.update_pyrs_basal_and_soma_ff(layer2)
    if print_predicate:
        print(f"""Layer3_FF_update_finished:
        ===============
        {layer3}""")


def do_fb_sweep(layer1: Layer, layer2: Layer, layer3: Layer, print_predicate=True):
    """Standard FB sweep with option to print network layer state after each step"""
    if print_predicate:
        print("Starting FB sweep")
    layer2.update_pyrs_apical_soma_fb(layer3)
    if print_predicate:
        print(f"""Layer2_w_apical_update_fb:
        ===============
        {layer2}""")

    layer1.update_pyrs_apical_soma_fb(layer2)
    if print_predicate:
        print(f"""Layer1_w_apical_update_fb:
        ===============
        {layer1}""")


def print_pyr_activations_all_layers_topdown(layer1: Layer, layer2: Layer, layer3: Layer):
    """Prints the pyr activations for all layers in the network, starting with the top layer"""
    layer3.print_pyr_activations()  # before learning
    layer2.print_pyr_activations()
    layer1.print_pyr_activations()


def print_ff_and_fb_wts_last_layer(layer2: Layer, layer3: Layer):
    print("FF wts coming into Layer 3")
    layer3.print_ff_wts()
    print("FB wts coming into Layer 2")
    layer2.print_fb_wts()


def learn_1_cycle_rule_16b_only(layer1: Layer, layer2: Layer, layer3: Layer, nudge_predicate=False):
    """
    Learning sweep that only uses Rule 16b
    Set 'nudge_p = True' to assess supervised learning.
    does one learning step
    """

    layer2.adjust_wts_lat_pi()  # adjust lateral PI wts in Layer 2
    layer1.adjust_wts_lat_pi()  # adjust lateral PI wts in Layer 1

    # Do FF and FB sweeps so wt changes show their effects.
    do_ff_sweep(layer1, layer2, layer3, print_predicate=False)
    if nudge_predicate:
        layer3.nudge_output_layer_neurons(2.0, -2.0, lambda_nudge=0.8)
    do_fb_sweep(layer1, layer2, layer3, print_predicate=False)


def learn_1_cycle_rule_16b_and_rule_13(layer1: Layer, layer2: Layer, layer3: Layer, nudge_predicate=False):
    """
    Learning sweep that uses both rules 16b and 13
    does one learning step
    """
    layer2.adjust_wts_lat_pi()  # adjust lateral PI wts in Layer 2
    # layer2.adjust_wts_lat_IP()      # adjust lateral IP wts in Layer 2

    layer3.adjust_wts_pp_ff(layer2)  # adjust FF wts projecting to Layer 3

    layer1.adjust_wts_lat_pi()  # adjust lateral PI wts in Layer 1
    # layer1.adjust_wts_lat_IP()      # adjust lateral IP wts in Layer 1

    layer2.adjust_wts_pp_ff(layer1)  # adjust FF wts projecting to Layer 2

    # Do FF and FB sweeps so wt changes show their effects.
    do_ff_sweep(layer1, layer2, layer3, print_predicate=False)
    if nudge_predicate:
        layer3.nudge_output_layer_neurons(2.0, -2.0, lambda_nudge=0.8)
    do_fb_sweep(layer1, layer2, layer3, print_predicate=False)


def train_to_self_predictive_state(layer1: Layer, layer2: Layer, layer3: Layer, n_steps=40):
    """
    Uses the 16b learning rule to train to a self-predictive state
    Prints the apical membrane potentials and pyr activations at end
    """

    do_ff_sweep(layer1, layer2, layer3, print_predicate=False)
    do_fb_sweep(layer1, layer2, layer3, print_predicate=False)
    for _ in range(n_steps):
        learn_1_cycle_rule_16b_only(layer1, layer2, layer3)
    print(f"Trained to self pred state: {n_steps} steps.")
    layer2.print_apical_mps()
    layer1.print_apical_mps()
    print_pyr_activations_all_layers_topdown(layer1, layer2, layer3)
    print("Above shows network in self predictive state.")


def run_pilot_experiment_1a(layer1: Layer, layer2: Layer, layer3: Layer):
    """
    Script to run the 1st pilot experiment, version 1a, using only Rule 16b
    This experiment allows the initialized network to evolve using ONLY rule 16b and
    tracks apical membrane potentials which should converge to zero.
    """

    do_ff_sweep(layer1, layer2, layer3)  # prints state
    print("Finished 1st FF sweep: pilot_exp_1")

    do_fb_sweep(layer1, layer2, layer3)  # prints state
    print("Finished 1st FB sweep: pilot_exp_1")

    n_of_learning_steps = 150
    print(f"Starting learning {n_of_learning_steps} steps for p_exp 1")

    data1 = []
    data2 = []
    for _ in range(n_of_learning_steps):
        learn_1_cycle_rule_16b_only(layer1, layer2, layer3)  # <== only 1 learning rule is used.
        data1.append([layer1.pyrs[0].apical_mp, layer1.pyrs[1].apical_mp])
        data2.append([layer2.pyrs[0].apical_mp, layer2.pyrs[1].apical_mp, layer2.pyrs[2].apical_mp])
    return data1, data2


def run_pilot_experiment_1b(layer1: Layer, layer2: Layer, layer3: Layer):
    """
    Script to run the 1st pilot experiment, version 1b, using Rules 16b and 13
    This experiment allows the initialized network to evolve using rules 16b AND 13
    tracks apical membrane potentials which should converge to zero.
    """

    do_ff_sweep(layer1, layer2, layer3)  # prints state
    print("Finished 1st FF sweep: pilot_exp_1")
    do_fb_sweep(layer1, layer2, layer3)  # prints state
    print("Finished 1st FB sweep: pilot_exp_1")
    n_of_learning_steps = 150
    print(f"Starting learning {n_of_learning_steps} steps for p_exp 1b")

    data1 = []
    data2 = []
    for _ in range(n_of_learning_steps):
        learn_1_cycle_rule_16b_and_rule_13(layer1, layer2, layer3)  # <== uses TWO learning rule is used.
        # Look at apical mps to see if converging to self-predictive state
        data1.append([layer1.pyrs[0].apical_mp, layer1.pyrs[1].apical_mp])
        data2.append([layer2.pyrs[0].apical_mp, layer2.pyrs[1].apical_mp, layer2.pyrs[2].apical_mp])

    return data1, data2


def run_pilot_experiment_2_rule_16b_only(layer1: Layer, layer2: Layer, layer3: Layer):
    """
    Script to run the 2nd pilot experiment: 2a
    This experiment checks to see if rule 16b alone with nudging can do effective learning
    """

    train_to_self_predictive_state(layer1, layer2, layer3)  # put network in self-predictive state
    # nudge the 1st neuron (index=0) in layer 3
    layer3.nudge_output_layer_neurons(2.0, -2.0, lambda_nudge=0.8)
    print("Layer 3 activations after nudge.")
    layer3.print_pyr_activations()
    print("Starting FB sweep")
    do_fb_sweep(layer1, layer2, layer3)  # prints state
    print("Finished 1st FB sweep after nudge: pilot_exp_2")  # shows effect of nudge in earlier layers
    print_pyr_activations_all_layers_topdown(layer1, layer2, layer3)  # displays in topdown order
    n_of_learning_steps = 40
    print(f"Starting learning {n_of_learning_steps} steps for p_exp 2a")
    for _ in range(n_of_learning_steps):  # train with rule 16b and maintain nudging
        learn_1_cycle_rule_16b_only(layer1, layer2, layer3, nudge_predicate=True)
        # Check apical mps to see if converging to self-predictive state
        # layer2.print_apical_mps()
        # layer1.print_apical_mps()
    print(f"Finished learning {n_of_learning_steps} steps for p_exp 2a")
    print_pyr_activations_all_layers_topdown(layer1, layer2, layer3)  # print activations while nudging is still on
    do_ff_sweep(layer1, layer2, layer3, print_predicate=False)  # to get new activations without nudging
    print("Final activations after nudging is removed")
    print_pyr_activations_all_layers_topdown(layer1, layer2, layer3)  # shows the true effect of learning
    # look at data in sheet "Only Rule 16b. No significant learning."


def run_pilot_experiment_2b_rule_16b_and_rule_13(layer1: Layer, layer2: Layer, layer3: Layer):
    """
    Script to run the 2nd pilot experiment: 2b
    This experiment checks to see if rule 16b and Rule 13 together with nudging can do effective learning
    """

    train_to_self_predictive_state(layer1, layer2, layer3, 150)  # put network in self-predictive state
    layer3.nudge_output_layer_neurons(2.0, -2.0, lambda_nudge=0.8)
    print("Layer 3 activations after nudge.")
    layer3.print_pyr_activations()
    print("Starting FB sweep")
    do_fb_sweep(layer1, layer2, layer3)  # prints state
    print("Finished 1st FB sweep after nudge: pilot_exp_2b")  # shows effect of nudge in earlier layers
    print_pyr_activations_all_layers_topdown(layer1, layer2, layer3)  # displays in topdown order
    n_of_learning_steps = 200
    print(f"Starting learning {n_of_learning_steps} steps for p_exp 3b")

    data1 = []
    data2 = []
    for _ in range(n_of_learning_steps):  # train with rule 16b and maintain nudging
        learn_1_cycle_rule_16b_and_rule_13(layer1, layer2, layer3, nudge_predicate=True)
        # Check apical mps to see if converging to self-predictive state
        data1.append([layer1.pyrs[0].apical_mp, layer1.pyrs[1].apical_mp])
        data2.append([layer2.pyrs[0].apical_mp, layer2.pyrs[1].apical_mp, layer2.pyrs[2].apical_mp])

    print_pyr_activations_all_layers_topdown(layer1, layer2, layer3)  # print activations while nudging is still on
    do_ff_sweep(layer1, layer2, layer3, print_predicate=False)  # to get new activations without nudging
    print("Final activations after nudging is removed")
    print_pyr_activations_all_layers_topdown(layer1, layer2, layer3)  # shows the true effect of learning
    # look at data in sheet "Only Rule 16b. No significant learning."
    return data1, data2


def run_pilot_exp_1b_concat_2b(layer1: Layer, layer2: Layer, layer3: Layer, steps_to_self_pred=150):
    do_ff_sweep(layer1, layer2, layer3)  # prints state
    print("Finished 1st FF sweep: pilot_exp_1")
    do_fb_sweep(layer1, layer2, layer3)  # prints state
    print("Finished 1st FB sweep: pilot_exp_1")
    n_of_learning_steps_to_self_predictive = steps_to_self_pred
    print(f"Starting learning {n_of_learning_steps_to_self_predictive} steps to 1b self predictive.")

    data1 = []
    data2 = []
    for _ in range(n_of_learning_steps_to_self_predictive):
        learn_1_cycle_rule_16b_and_rule_13(layer1, layer2, layer3)  # <== uses TWO learning rule is used.
        data1.append(list(map(lambda x: x.apical_mp, layer1.pyrs)))
        data2.append(list(map(lambda x: x.apical_mp, layer2.pyrs)))

    layer3.nudge_output_layer_neurons(2.0, -2.0, lambda_nudge=0.8)
    print("Layer 3 activations after nudge.")

    layer3.print_pyr_activations()
    print("Starting FB sweep")

    do_fb_sweep(layer1, layer2, layer3)  # prints state
    print("Finished 1st FB sweep after nudge: pilot_exp_2b")  # shows effect of nudge in earlier layers

    print_pyr_activations_all_layers_topdown(layer1, layer2, layer3)  # displays in topdown order
    n_of_learning_steps = 200
    print(f"Starting learning {n_of_learning_steps} steps for p_exp 3b")

    for _ in range(n_of_learning_steps):  # train with rule 16b and maintain nudging
        learn_1_cycle_rule_16b_and_rule_13(layer1, layer2, layer3, nudge_predicate=True)
        # Check apical mps to see if converging to self-predictive state
        data1.append([layer1.pyrs[0].apical_mp, layer1.pyrs[1].apical_mp])
        data2.append([layer2.pyrs[0].apical_mp, layer2.pyrs[1].apical_mp, layer2.pyrs[2].apical_mp])

    print(f"Finished learning {n_of_learning_steps} steps for p_exp 3b")
    print_pyr_activations_all_layers_topdown(layer1, layer2, layer3)  # print activations while nudging is still on
    do_ff_sweep(layer1, layer2, layer3, print_predicate=False)  # to get new activations without nudging
    print("Final activations after nudging is removed")
    print_pyr_activations_all_layers_topdown(layer1, layer2, layer3)  # shows the true effect of learning

    return data1, data2


def generate_plot(values: list[list[float]]):
    data = np.array(values)
    x_axis = np.arange(data.shape[0])
    fig, ax = plt.subplots()
    args = []
    for i in range(data.shape[1]):
        args.extend((x_axis, data[:, i]))
    ax.plot(*args)
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_str = b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str


def main():
    """ Do an experiment """
    layer1, layer2, layer3 = build_small_three_layer_network()  # build the network
    # print_ff_and_fb_wts_last_layer(layer2, layer3)
    # run_pilot_experiment_1a(layer1, layer2, layer3)
    # run_pilot_experiment_1b(layer1, layer2, layer3)
    # run_pilot_experiment_2_rule_16b_only(layer1, layer2, layer3)
    # run_pilot_experiment_2b_rule_16b_and_rule_13(layer1, layer2, layer3) # run the experiment

    # p_Exp 3: steps_to_self_pred = 150.
    # p_Exp 3b: steps_to_self_pred = 250.
    datasets = run_pilot_exp_1b_concat_2b(layer1, layer2, layer3, steps_to_self_pred=400)

    print_ff_and_fb_wts_last_layer(layer2, layer3)
    return datasets


if __name__ == '__main__':
    main()
