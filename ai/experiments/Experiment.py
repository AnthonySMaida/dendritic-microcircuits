# import logging
from abc import abstractmethod
from typing import List, Tuple

import numpy as np
from werkzeug.datastructures.structures import MultiDict

from ai.Layer import Layer
from ai.colorized_logger import get_logger
from ai.utils import iter_with_prev
from metrics import Graph

logger = get_logger('ai.experiments.Experiment')


class Experiment:
    def __init__(self, params: MultiDict):
        self._wt_init_seed = params.get('wt_init_seed', 42, type=int)

        self._beta = params.get('beta', 1.0 / 3.0, type=float)  # for exponential sampling
        self._learning_rate = params.get('learning_rate', 0.05, type=float)
        self._metrics = {}
        self._rng_wts = np.random.default_rng(seed=self._wt_init_seed)

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

    def build_network(self, n_layer: int, *n_pyr_nrns: Tuple[int, int, int, int]):
        """
        Build a n-layer network

        :param n_layer: int number of layers
        :param n_pyr_nrns: Tuple of 4 ints, representing the number of pyramidal neurons, inhibitory neurons,
                            lateral inhibitory neurons, and lateral pyramidal neurons in each layer
        """
        layers = []
        for i, (prev_n_pyr, n_pyr) in zip(range(n_layer), iter_with_prev(n_pyr_nrns)):
            layer = Layer(i=i,
                          learning_rate=self._learning_rate,
                          rng=self._rng_wts,
                          beta=self._beta,
                          n_pyrs=n_pyr[0],
                          n_inhibs=n_pyr[1],
                          n_pyr_ff_wt=prev_n_pyr[0] if prev_n_pyr is not None else None,
                          n_pyr_fb_wt=n_pyr[0] if i < n_layer - 1 else None,
                          n_ip_lat_wt=n_pyr[2],
                          n_pi_lat_wt=n_pyr[3])
            layers.append(layer)
        self.layers = layers

    def build_small_two_layer_network(self, n_input_pyr_nrns: int = 2, n_output_pyr_nrns: int = 2):
        """Build 2-layer network
        Layer 1 is the input layer w/ 2 pyrs and 1 inhib cell.
        No FF connections in input layer. They are postponed to receiving layer.
        Each layer 1 pyr projects a FF connection to each of 2 pyrs in output layer.
        Wts associated w/ a neuron instance are always incoming wts.
        """
        logger.info("Building model...")

        l1 = Layer(1, self._learning_rate, self._rng_wts, n_input_pyr_nrns, 1, None, 1, 2, self._beta,2)
        logger.warning("""Layer 1:\n========\n%s""", l1)

        # Layer 2 is output layer w/ 2 pyrs. No inhib neurons.
        l2 = Layer(2, self._learning_rate, self._rng_wts, n_output_pyr_nrns, 0, n_input_pyr_nrns, None, None, self._beta,None )
        logger.warning("""Layer 2:\n========\n%s""", l2)

        self.layers = [l1, l2]
        logger.info("Finished building 2-layer model.")

    def build_small_three_layer_network(self, n_input_pyr_nrns: int = 2, n_hidden_pyr_nrns: int = 3,
                                        n_output_pyr_nrns: int = 2):
        """Build 3-layer network"""
        # Layer 1 is the input layer w/ 2 pyrs and 1 inhib cell.
        # No FF connections in input layer. They are postponed to receiving layer.
        # Each pyramid projects a FF connection to each of 3 pyrs in Layer 2 (hidden).
        # wts are always incoming weights.
        logger.info("Building model...")
        l1 = Layer(1, self._learning_rate, self._rng_wts, n_input_pyr_nrns, 1, None, 1, n_hidden_pyr_nrns, self._beta,
                   2)
        logger.warning("""Layer 1:\n========\n%s""", l1)

        # Layer 2 is hidden layer w/ 3 pyrs.
        # Also has 3 inhib neurons.
        # Has feedback connections to Layer 1
        l2 = Layer(2, self._learning_rate, self._rng_wts, n_hidden_pyr_nrns, 3, n_input_pyr_nrns, 3, n_output_pyr_nrns,
                   self._beta, 3)
        logger.warning("""Layer 2:\n========\n%s""", l2)

        # Layer 3 is output layer w/ 2 pyrs. No inhib neurons.
        l3 = Layer(3, self._learning_rate, self._rng_wts, n_output_pyr_nrns, 0, n_hidden_pyr_nrns, None, None,
                   self._beta, None)
        logger.warning("""Layer 3:\n========\n%s""", l3)

        self.layers = [l1, l2, l3]

        logger.info("Finished building 3-layer model.")

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
    def run(self):
        raise NotImplementedError
