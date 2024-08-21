#import logging
from abc import abstractmethod
from typing import List

import numpy as np

from ai.Layer import Layer
from ai.colorized_logger import get_logger
from metrics import Graph

logger = get_logger('ai.experiments.Experiment')
#logger.setLevel(logging.DEBUG)


class Experiment:
    def __init__(self, wt_init_seed: int, beta: float, learning_rate: float):
        self._beta = beta
        self._learning_rate = learning_rate
        self._metrics = {}
        self._rng_wts = np.random.default_rng(seed=wt_init_seed)

        self.layers: List[Layer] = []  # list is made of layers. List type assists code completion.

    def _hook_pre_train_step(self):
        """
        Hook called before each training step
        """
        pass

    def _hook_post_train_step(self):
        """
        Hook called after each training step
        """
        pass

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

    def build_small_three_layer_network(self, n_input_pyr_nrns: int = 2, n_hidden_pyr_nrns: int = 3, n_output_pyr_nrns: int = 2):
        """Build 3-layer network"""

        # Ensures the first layer is always 1 for a new network
        Layer.next_id = 1

        # Layer 1 is the input layer w/ 2 pyrs and 1 inhib cell.
        # No FF connections in input layer. They are postponed to receiving layer.
        # Each pyramid projects a FF connection to each of 3 pyrs in Layer 2 (hidden).
        # wts are always incoming weights.
        l1 = Layer(self._learning_rate, self._rng_wts, n_input_pyr_nrns, 1, None, 1, n_hidden_pyr_nrns, self._beta, 2)
        logger.info("Building model...")
        logger.warning("""Layer 1:\n========\n%s""", l1)

        # Layer 2 is hidden layer w/ 3 pyrs.
        # Also has 3 inhib neurons.
        # Has feedback connections to Layer 1
        l2 = Layer(self._learning_rate, self._rng_wts, n_hidden_pyr_nrns, 3, n_input_pyr_nrns, 3, n_output_pyr_nrns, self._beta, 3)
        logger.warning("""Layer 2:\n========\n%s""", l2)

        # Layer 3 is output layer w/ 2 pyrs. No inhib neurons.
        l3 = Layer(self._learning_rate, self._rng_wts, n_output_pyr_nrns, 0, n_hidden_pyr_nrns, None, None, self._beta, None)
        logger.warning("""Layer 3:\n========\n%s""", l3)

        self.layers = [l1, l2, l3]

        logger.info("Finished building model.")

    def train(self, n_steps: int, *args, **kwargs):
        """
        Previously called: train_data
        Train 1 step.
        :param n_steps: int num of training steps
        :param args: No args.
        :param kwargs: nudge_predicate (True or False), to indicate if nudging happens.
        :return:
        """
        for _ in range(n_steps):
            self._hook_pre_train_step()
            self._train_1_step(*args, **kwargs)  # do training.
            self._hook_post_train_step()

    @abstractmethod
    def _train_1_step(self, *args, **kwargs):  # I would have never figured out the signature.
        """
        Formerly called "train()". Abstract method implemented in subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def extract_metrics(self) -> List[Graph]:  # declares return type
        """
        This method must return all data that will be plotted
        This should also process the "raw" data to be in the correct return format

        :return: list of `Graph`
        """
        raise NotImplementedError

    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError
