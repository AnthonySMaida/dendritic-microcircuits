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
import logging
from abc import abstractmethod
from typing import Callable, Tuple, List

import numpy as np

from ai.Layer import Layer
from ai.config import get_rng, n_input_pyr_nrns, n_hidden_pyr_nrns, n_output_pyr_nrns, nudge1, nudge2, wt_init_seed
from ai.utils import iter_with_prev

logger = logging.getLogger('ai.sacramento_main')
logger.setLevel(logging.INFO)


class Experiment:
    def __init__(self):
        self.rule13_post_data = np.array([[0, 0, 0, 0, 0]])
        self.rng = get_rng()
        self.datasets: List[List[float]] = []
        self.layers: List[Layer] = []

    def build_small_three_layer_network(self):
        """Build 3-layer network"""
        # Layer 1 is the input layer w/ 2 pyrs and 1 inhib cell.
        # No FF connections in input layer. They are postponed to receiving layer.
        # Each pyramid projects a FF connection to each of 3 pyrs in Layer 2 (hidden).
        # wts are always incoming weights.
        l1 = Layer(self.rng, n_input_pyr_nrns, 1, None, 2, 3, 1)
        logger.info("Building model...")
        logger.debug("""Layer 1:\n========\n%s""", l1)

        # Layer 2 is hidden layer w/ 3 pyrs.
        # Also has 3 inhib neurons.
        # Has feedback connections to Layer 1
        l2 = Layer(self.rng, n_hidden_pyr_nrns, 3, 2, 3, 2, 3)
        logger.debug("""Layer 2:\n========\n%s""", l2)

        # Layer 3 is output layer w/ 2 pyrs. No inhib neurons.
        l3 = Layer(self.rng, n_output_pyr_nrns, 0, 3, None, None, None)
        logger.debug("""Layer 3:\n========\n%s""", l3)

        self.datasets = [[], [], []]
        self.layers = [l1, l2, l3]

        logger.info("Finished building model.")

    def do_ff_sweep(self):
        """Standard FF sweep"""
        logger.debug("Starting FF sweep...")

        for prev, layer in iter_with_prev(self.layers):
            if prev is None:
                layer.apply_inputs_to_test_self_predictive_convergence()
            else:
                layer.update_pyrs_basal_and_soma_ff(prev)
            layer.update_dend_mps_via_ip()
            logger.debug(layer)

        logging.debug("FF sweep done.")

    def do_fb_sweep(self):
        """Standard FB sweep"""
        logger.debug("Starting FB sweep...")

        for prev, layer in iter_with_prev(reversed(self.layers)):
            if prev is None:  # Skip first layer (L3)
                continue
            # update current layer pyrs using somatic pyr acts from previous layer and inhib acts from current layer
            layer.update_pyrs_apical_soma_fb(prev)
            logger.debug(layer)

        logger.debug("FB sweep done.")

    def nudge_output_layer(self):
        self.print_ff_and_fb_wts_last_layer()

        last_layer = self.layers[-1]
        last_layer.nudge_output_layer_neurons(nudge1, nudge2, lambda_nudge=0.8)
        logger.debug("Layer %d activations after nudge.", last_layer.id_num)
        last_layer.print_pyr_activations()

        logger.info("Starting FB sweep")
        self.do_fb_sweep()  # prints state

        logger.info("Finished 1st FB sweep after nudge: pilot_exp_2b")  # shows effect of nudge in earlier layers
        self.print_pyr_activations_all_layers_topdown()

    def print_pyr_activations_all_layers_topdown(self):
        """Prints the pyr activations for all layers in the network, starting with the top layer"""
        for layer in reversed(self.layers):
            layer.print_pyr_activations()

    def print_ff_and_fb_wts_last_layer(self):
        """Print incoming and outgoing wts of last layer"""
        last_layer = self.layers[-1]
        prev_last_layer = self.layers[-2]
        logger.info("FF wts coming into Layer %d", last_layer.id_num)
        last_layer.print_ff_wts()
        logger.info("FB wts coming into Layer %d", prev_last_layer.id_num)
        prev_last_layer.print_fb_wts()

    def train_data(self, n_steps: int, *args, **kwargs):
        for _ in range(n_steps):
            self.train(*args, **kwargs)
            for data, layer in zip(self.datasets, self.layers):
                data.append(list(map(lambda x: x.apical_mp, layer.pyrs)))

    @abstractmethod
    def train(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def run(self):
        raise NotImplementedError


##########################################
# Learning sweep that only uses Rule 16b #
##########################################


def train_1_step_rule_16b_only(layer1: Layer, layer2: Layer, layer3: Layer, nudge_predicate=False):
    """
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


def train_1_step_rule_16b_and_rule_13(layer1: Layer, layer2: Layer, layer3: Layer, nudge_predicate=False):
    """
    Learning sweep that uses both rules 16b and 13
    does one training step
    """
    layer2.adjust_wts_lat_pi()  # adjust lateral PI wts in Layer 2
    # layer2.adjust_wts_lat_IP()      # adjust lateral IP wts in Layer 2

    data_pt = layer3.adjust_wts_pp_ff(layer2)  # adjust FF wts projecting to Layer 3

    # save data point: [soma_act, apical_hat_act, post_val[]
    global rule13_post_data
    rule13_post_data = np.concatenate((rule13_post_data, data_pt), axis=0)

    # continue learning
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
        train_1_step_rule_16b_only(layer1, layer2, layer3)
    logger.info("Trained to self pred state: %d steps.", n_steps)
    layer2.print_apical_mps()
    layer1.print_apical_mps()
    print_pyr_activations_all_layers_topdown(layer1, layer2, layer3)
    logger.info("Above shows network in self predictive state.")


def run_pilot_experiment_1a(layer1: Layer, layer2: Layer, layer3: Layer):
    """
    Script to run the 1st pilot experiment, version 1a, using only Rule 16b
    This experiment allows the initialized network to evolve using ONLY rule 16b and
    tracks apical membrane potentials which should converge to zero.
    """

    do_ff_sweep(layer1, layer2, layer3)  # prints state
    logger.info("Finished 1st FF sweep: pilot_exp_1")

    do_fb_sweep(layer1, layer2, layer3)  # prints state
    logger.info("Finished 1st FB sweep: pilot_exp_1")

    n_of_training_steps = 150
    logger.info("Starting training %d steps for p_exp 1", n_of_training_steps)

    data1 = []
    data2 = []
    for _ in range(n_of_training_steps):
        train_1_step_rule_16b_only(layer1, layer2, layer3)  # <== only 1 learning rule is used.
        data1.append([layer1.pyrs[0].apical_mp, layer1.pyrs[1].apical_mp])
        data2.append([layer2.pyrs[0].apical_mp, layer2.pyrs[1].apical_mp, layer2.pyrs[2].apical_mp])
    logger.info("Finished learning %d steps for p_exp 1", n_of_training_steps)
    print_pyr_activations_all_layers_topdown(layer1, layer2, layer3)
    return data1, data2


def run_pilot_experiment_1b(layer1: Layer, layer2: Layer, layer3: Layer):
    """
    Script to run the 1st pilot experiment, version 1b, using Rules 16b and 13
    This experiment allows the initialized network to evolve using rules 16b AND 13
    tracks apical membrane potentials which should converge to zero.
    """

    do_ff_sweep(layer1, layer2, layer3)  # prints state
    logger.info("Finished 1st FF sweep: pilot_exp_1")
    do_fb_sweep(layer1, layer2, layer3)  # prints state
    logger.info("Finished 1st FB sweep: pilot_exp_1")
    n_of_training_steps = 150
    logger.info("Starting learning %d steps for p_exp 1b", n_of_training_steps)

    data1 = []
    data2 = []
    for _ in range(n_of_training_steps):
        train_1_step_rule_16b_and_rule_13(layer1, layer2, layer3)  # <== uses TWO learning rule is used.
        # Look at apical mps to see if converging to self-predictive state
        data1.append([layer1.pyrs[0].apical_mp, layer1.pyrs[1].apical_mp])
        data2.append([layer2.pyrs[0].apical_mp, layer2.pyrs[1].apical_mp, layer2.pyrs[2].apical_mp])

    logger.info("Finished training %d steps for p_exp 1", n_of_training_steps)
    print_pyr_activations_all_layers_topdown(layer1, layer2, layer3)
    return data1, data2


def run_pilot_experiment_2_rule_16b_only(layer1: Layer, layer2: Layer, layer3: Layer):
    """
    Script to run the 2nd pilot experiment: 2a
    This experiment checks to see if rule 16b alone with nudging can do effective learning
    """

    train_to_self_predictive_state(layer1, layer2, layer3)  # put network in self-predictive state
    # nudge the 1st neuron (index=0) in layer 3
    nudge_output_layer(layer1, layer2, layer3)
    n_of_training_steps = 40
    logger.info("Starting training %d steps for p_exp 2a", n_of_training_steps)
    for _ in range(n_of_training_steps):  # train with rule 16b and maintain nudging
        train_1_step_rule_16b_only(layer1, layer2, layer3, nudge_predicate=True)
        # Check apical mps to see if converging to self-predictive state
        # layer2.print_apical_mps()
        # layer1.print_apical_mps()
    logger.info("Finished training %d steps for p_exp 2a", n_of_training_steps)
    print_pyr_activations_all_layers_topdown(layer1, layer2, layer3)  # print activations while nudging is still on
    do_ff_sweep(layer1, layer2, layer3, print_predicate=False)  # to get new activations without nudging
    logger.info("Final activations after nudging is removed")
    print_pyr_activations_all_layers_topdown(layer1, layer2, layer3)  # shows the true effect of learning
    # look at data in sheet "Only Rule 16b. No significant learning."


def run_pilot_experiment_2b_rule_16b_and_rule_13(layer1: Layer, layer2: Layer, layer3: Layer):
    """
    Script to run the 2nd pilot experiment: 2b
    This experiment checks to see if rule 16b and Rule 13 together with nudging can do effective learning
    """

    train_to_self_predictive_state(layer1, layer2, layer3, 150)  # put network in self-predictive state
    nudge_output_layer(layer1, layer2, layer3)

    n_of_training_steps = 200
    logger.info("Starting training %d steps for p_exp 3b", n_of_training_steps)

    data1 = []
    data2 = []
    data3 = []
    train_data(n_of_training_steps,
               lambda: train_1_step_rule_16b_and_rule_13(layer1, layer2, layer3, nudge_predicate=True),
               *(
                   (data1, layer1),
                   (data2, layer2),
                   (data3, layer3)
               ))

    logger.info("Finished training %d steps for p_exp 3b", n_of_training_steps)
    print_pyr_activations_all_layers_topdown(layer1, layer2, layer3)  # print activations while nudging is still on
    do_ff_sweep(layer1, layer2, layer3, print_predicate=False)  # to get new activations without nudging
    logger.info("Final activations after nudging is removed")
    print_pyr_activations_all_layers_topdown(layer1, layer2, layer3)  # shows the true effect of learning
    # look at data in sheet "Only Rule 16b. No significant learning."
    return data1, data2


class PilotExp1bConcat2b(Experiment):
    def __init__(self):
        super().__init__()

    def train_1_step_rule_16b_and_rule_13(self, nudge_predicate=False):
        """
        Learning sweep that uses both rules 16b and 13
        does one training step
        """
        l1, l2, l3 = self.layers
        l2.adjust_wts_lat_pi()  # adjust lateral PI wts in Layer 2
        # l2.adjust_wts_lat_IP()      # adjust lateral IP wts in Layer 2

        data_pt = l3.adjust_wts_pp_ff(l2)  # adjust FF wts projecting to Layer 3

        # save data point: [soma_act, apical_hat_act, post_val[]
        self.rule13_post_data = np.concatenate((self.rule13_post_data, data_pt), axis=0)

        # continue learning
        l1.adjust_wts_lat_pi()  # adjust lateral PI wts in Layer 1
        # l1.adjust_wts_lat_IP()      # adjust lateral IP wts in Layer 1

        l2.adjust_wts_pp_ff(l1)  # adjust FF wts projecting to Layer 2

        # Do FF and FB sweeps so wt changes show their effects.
        self.do_ff_sweep()
        if nudge_predicate:
            l3.nudge_output_layer_neurons(2.0, -2.0, lambda_nudge=0.8)
        self.do_fb_sweep()

    def train(self, nudge_predicate: bool):
        self.train_1_step_rule_16b_and_rule_13(nudge_predicate=nudge_predicate)

    def run(self, steps_to_self_pred=150):
        self.do_ff_sweep()  # prints state
        logger.info("Finished 1st FF sweep: pilot_exp_1")

        self.do_fb_sweep()  # prints state
        logger.info("Finished 1st FB sweep: pilot_exp_1")

        n_of_training_steps_to_self_predictive = steps_to_self_pred
        logger.info(f"Starting training {n_of_training_steps_to_self_predictive} steps to 1b self predictive.")

        self.train_data(n_of_training_steps_to_self_predictive, nudge_predicate=False)

        self.nudge_output_layer()

        n_of_training_steps = 200
        logger.info(f"Starting training {n_of_training_steps} steps for p_exp 3b")

        self.train_data(n_of_training_steps, nudge_predicate=True)

        logger.info(f"Finished training {n_of_training_steps} steps for p_exp 3b")
        self.print_pyr_activations_all_layers_topdown()  # print activations while nudging is still on

        self.do_ff_sweep()  # to get new activations without nudging

        logger.info("Final activations after nudging is removed")
        self.print_pyr_activations_all_layers_topdown()  # shows the true effect of learning


def main():
    """Do an experience"""
    experiment = PilotExp1bConcat2b()
    experiment.build_small_three_layer_network()

    # p_Exp 3: steps_to_self_pred = 150.
    # p_Exp 3b: steps_to_self_pred = 250.
    experiment.run(steps_to_self_pred=400)

    experiment.print_ff_and_fb_wts_last_layer()

    logger.info("nudge1 = %s; nudge2 = %s\n", nudge1, nudge2)
    logger.info("wt_init_seed = %d", wt_init_seed)

    return experiment.datasets


if __name__ == '__main__':
    main()
