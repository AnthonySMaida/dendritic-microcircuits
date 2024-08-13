#import logging
from abc import abstractmethod
from typing import List

import numpy as np

from ai.Layer import Layer
from ai.utils import iter_with_prev
from ai.colorized_logger import get_logger
from metrics import Graph

logger = get_logger('ai.experiments.Experiment')
#logger.setLevel(logging.DEBUG)


class Experiment:
    def __init__(self, wt_init_seed: int, beta: float, learning_rate: float, nudge1: float, nudge2: float):
        self._beta = beta
        self._learning_rate = learning_rate
        self._metrics = {}
        self._nudge1 = nudge1
        self._nudge2 = nudge2
        self._rng = np.random.default_rng(seed=wt_init_seed)

        self.layers: List[Layer] = []  # Tells Python list is made of layers
                                       # Helps w/ code completion

    @abstractmethod
    def extract_metrics(self) -> List[Graph]:  # declares return type
        """
        This method must return all data that will be plotted
        This should also process the "raw" data to be in the correct return format

        :return: list of `Graph`
        """
        raise NotImplementedError

    def build_small_three_layer_network(self, n_input_pyr_nrns: int, n_hidden_pyr_nrns: int, n_output_pyr_nrns: int):
        """Build 3-layer network"""
        # Layer 1 is the input layer w/ 2 pyrs and 1 inhib cell.
        # No FF connections in input layer. They are postponed to receiving layer.
        # Each pyramid projects a FF connection to each of 3 pyrs in Layer 2 (hidden).
        # wts are always incoming weights.
        l1 = Layer(self._learning_rate, self._rng, n_input_pyr_nrns, 1, None, 1, 3, self._beta, 2)
        logger.info("Building model...")
        logger.warning("""Layer 1:\n========\n%s""", l1)

        # Layer 2 is hidden layer w/ 3 pyrs.
        # Also has 3 inhib neurons.
        # Has feedback connections to Layer 1
        l2 = Layer(self._learning_rate, self._rng, n_hidden_pyr_nrns, 3, 2, 3, 2, self._beta, 3)
        logger.warning("""Layer 2:\n========\n%s""", l2)

        # Layer 3 is output layer w/ 2 pyrs. No inhib neurons.
        l3 = Layer(self._learning_rate, self._rng, n_output_pyr_nrns, 0, 3, None, None, self._beta, None)
        logger.warning("""Layer 3:\n========\n%s""", l3)

        self.layers = [l1, l2, l3]

        logger.info("Finished building model.")

    def do_ff_sweep(self):
        """Standard FF sweep"""
        logger.debug("Starting FF sweep...")

        # Iterates over layers from start to end. From ai.utils.
        for prev, layer in iter_with_prev(self.layers):  # Yields prev layer and current layer
            if prev is None:
                layer.apply_inputs_to_test_self_predictive_convergence()
            else:
                layer.update_pyrs_basal_and_soma_ff(prev)
            layer.update_dend_mps_via_ip()
            #logger.debug(layer)

        #logger.info("FF sweep done.")

    def do_fb_sweep(self):
        """Standard FB sweep"""
        logger.debug("Starting FB sweep...")

        for prev, layer in iter_with_prev(reversed(self.layers)):
            if prev is None:  # Skip first layer (L3)
                continue
            # update current layer pyrs using somatic pyr acts from previous layer and inhib acts from current layer
            layer.update_pyrs_apical_soma_fb(prev)
            #logger.debug(layer)

        #logger.info("FB sweep done.")

    def nudge_output_layer(self):
        """
        Prints all layer wts.
        Imposes nudge on the output layer and prints output layer activations.
        Does a FF sweep and then prints layer activations in reverse order.
        """
        self.layers[-2].print_fb_and_pi_wts_layer()
        self.layers[-2].print_ff_and_ip_wts_for_layers(self.layers[-1])

        last_layer = self.layers[-1]
        logger.debug("Layer %d activations before nudge.", last_layer.id_num)
        last_layer.print_pyr_activations()

        logger.info("Imposing nudge now")

        last_layer = self.layers[-1]
        last_layer.nudge_output_layer_neurons(self._nudge1, self._nudge2, lambda_nudge=0.8)
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

    def train_1_step_rule_16b_and_rule_13(self, use_nudge=False, use_rule_ip = False):
        """
        Learning step that uses both rules 16b and 13.
        Does one training step.
        """
        l1, l2, l3 = self.layers
        l2.adjust_wts_lat_pi()  # adjust lateral PI wts in Layer 2

        # l2.adjust_wts_lat_IP()      # adjust lateral IP wts in Layer 2

        # Adjust FF wts projecting to Layer 3.
        l3.adjust_wts_pp_ff(l2)  # adjust FF wts projecting to Layer 3

        # continue learning
        l1.adjust_wts_lat_pi()  # adjust lateral PI wts in Layer 1

        if use_rule_ip:  # also known as Rule 16a
            l1.adjust_wts_lat_IP()      # adjust lateral IP wts in Layer 1

        l2.adjust_wts_pp_ff(l1)  # adjust FF wts projecting to Layer 2

        # Do FF and FB sweeps so wt changes show their effects.
        self.do_ff_sweep()
        if use_nudge:
            l3.nudge_output_layer_neurons(self._nudge1, self._nudge2, lambda_nudge=0.8)
        self.do_fb_sweep()

    def train_and_save_apical_data(self, n_steps: int, *args, **kwargs):
        """
        Previously called: train_data
        Train 1 step.
        :param n_steps: int num of training steps
        :param args: No args.
        :param kwargs: nudge_predicate (True or False), to indicate if nudging happens.
        :return:
        """
        for _ in range(n_steps):
            self.hook_pre_train_step()
            self.train_1_step(*args, **kwargs)  # do training.
            self.hook_post_train_step()

    def hook_pre_train_step(self):
        """
        Hook called before each training step
        """
        pass

    def hook_post_train_step(self):
        """
        Hook called after each training step
        """
        pass

    @abstractmethod
    def train_1_step(self, *args, **kwargs):  # I would have never figured out the signature.
        """
        Formerly called "train()". Abstract method implemented in subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError
