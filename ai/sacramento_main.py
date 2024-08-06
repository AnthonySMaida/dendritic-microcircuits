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
Layer 1: input  (2 pyramids, 1 interneuron)
Layer 2: hidden (3 pyramids, 3 interneurons)
Layer 3: output (2 pyramids)

pyramidal neurons are called "pyrs"
interneurons are called 'inhibs'

Implemented in numpy. The code is not vectorized but the
data structures used closely mimic the neural anatomy given in the paper.
"""
import logging
from abc import abstractmethod
from typing import List

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
        """
        Prints all layer wts.
        Imposes nudge on the output layer and prints output layer activations.
        Does a FF sweep and then prints layer activations in reverse order.
        """
        self.layers[-2].print_fb_and_pi_wts_layer()
        self.layers[-2].print_ff_and_ip_wts_for_layers(self.layers[-1])

        logger.info("Imposing nudge")

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


class PilotExp1bConcat2b(Experiment):
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

    logger.info("nudge1 = %s; nudge2 = %s", nudge1, nudge2)
    logger.info("wt_init_seed = %d", wt_init_seed)

    return *experiment.datasets[:2], experiment.rule13_post_data[1:].tolist()


if __name__ == '__main__':
    main()
