from typing import List, Optional

import numpy as np

from ai.Layer import Layer
from ai.colorized_logger import get_logger
from ai.experiments.Experiment import Experiment
from ai.utils import iter_with_prev, create_column_vector
from metrics import Graph


logger = get_logger('ai.experiments.AndOrExperiment')

KEY_LAYER_1 = "layer1"
KEY_LAYER_2 = "layer2"


class AndOrExperiment(Experiment):
    def __init__(self, wt_init_seed: int, label_init_seed: int, beta: float, learning_rate: float):
        super().__init__(wt_init_seed, beta, learning_rate)

        self._metrics[KEY_LAYER_1] = np.empty(shape=(2, 0))
        self._metrics[KEY_LAYER_2] = np.empty(shape=(3, 0))

        self._X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self._Y = np.array([[0, 0], [0, 1], [0, 1], [1, 1]])

        self._current_index: Optional[int] = None
        self._current_X: Optional[np.ndarray] = None
        self._current_label: Optional[int] = None

        self._rng_labels = np.random.default_rng(seed=label_init_seed)

    def __do_ff_sweep(self):
        """Standard FF sweep"""
        logger.debug("Starting FF sweep...")

        # Iterates over layers from start to end. From ai.utils.
        for prev, layer in iter_with_prev(self.layers):  # Yields prev layer and current layer
            if prev is None:
                layer.apply_inputs_to_test_self_predictive_convergence(self._current_X)
            else:
                layer.update_pyrs_basal_and_soma_ff(prev)
            layer.update_dend_mps_via_ip()

    def __do_fb_sweep(self):
        """Standard FB sweep"""
        logger.debug("Starting FB sweep...")

        for prev, layer in iter_with_prev(reversed(self.layers)):
            if prev is None:
                continue
            layer.update_pyrs_apical_soma_fb(prev)

    def __gather_layer_metrics(self, key_post: str, key_wts: str, layer: Layer):
        soma_act = layer.pyr_soma_acts()[0]
        basal_hat_act = layer.pyr_basal_hat_acts()[0]
        post_soma_mp = layer.pyr_soma_mps()[0]
        post_basal_mp = layer.pyr_basal_mps()[0]
        post_val2 = post_soma_mp - post_basal_mp

        self._metrics[key_post] = np.append(
            self._metrics[key_post],
            create_column_vector(soma_act, basal_hat_act, post_soma_mp, post_basal_mp, post_val2),
            axis=1)
        self._metrics[key_wts] = np.append(self._metrics[key_wts],
                                           layer.pyrs[0].W_PP_ff[0])

    def __nudge_output_layer(self):
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
        last_layer.nudge_output_layer_neurons(self._current_label, lambda_nudge=0.8)
        logger.debug("Layer %d activations after nudge.", last_layer.id_num)
        last_layer.print_pyr_activations()

        logger.info("Starting FB sweep")
        self.__do_fb_sweep()  # prints state

        logger.info("Finished 1st FB sweep after nudge: pilot_exp_2b")  # shows effect of nudge in earlier layers
        self.print_pyr_activations_all_layers_topdown()

    def _hook_pre_train_step(self):
        self._current_index = index = self._rng_labels.integers(low=0, high=len(self._X))
        self._current_X = self._X[index]
        self._current_label = self._Y[index]

    def _hook_post_train_step(self):
        # Only record data if current index is 0
        if self._current_index != 0:
            return

        l1, l2 = self.layers
        self._metrics[KEY_LAYER_1] = np.append(
            self._metrics[KEY_LAYER_1],
            create_column_vector(*map(lambda p: p.apical_mp, l1.pyrs)),
            axis=1
        )
        self._metrics[KEY_LAYER_2] = np.append(
            self._metrics[KEY_LAYER_2],
            create_column_vector(*map(lambda p: p.apical_mp, l2.pyrs)),
            axis=1
        )

    def _train_1_step(self, *args, **kwargs):
        pass

    def extract_metrics(self) -> List[Graph]:
        pass

    def run(self, *args, **kwargs):
        pass
