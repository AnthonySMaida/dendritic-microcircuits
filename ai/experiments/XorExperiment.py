from typing import List, Optional

import numpy as np

from ai.Layer import Layer
from ai.colorized_logger import get_logger
from ai.experiments.Experiment import Experiment
from ai.utils import create_column_vector, iter_with_prev
from metrics import Graph, GraphType, Serie

logger = get_logger('ai.experiments.XorExperiment')

KEY_LAYER_1 = "layer1"
KEY_LAYER_2 = "layer2"
KEY_RULE_13_POST_DATA = "rule13_post_data"
KEY_RULE_13_WT_DATA = "rule13_wt_data"
KEY_RULE_13_POST_DATA_L1 = "rule13_post_data_l1"
KEY_RULE_13_WT_DATA_L1 = "rule13_wt_data_l1"
KEY_OUTPUT_LAYER_PYR_ACTS = "output_layer_acts"
KEY_HIDDEN_LAYER_PYR_ACTS = "hidden_layer_pyr_acts"
KEY_HIDDEN_LAYER_INHIB_ACTS = "hidden_layer_inhib_acts"


class XorExperiment(Experiment):
    def __init__(self, wt_init_seed: int, label_init_seed: int, beta: float, learning_rate: float):
        super().__init__(wt_init_seed, beta, learning_rate)

        self._metrics[KEY_LAYER_1] = np.empty(shape=(2, 0))
        self._metrics[KEY_LAYER_2] = np.empty(shape=(3, 0))
        self._metrics[KEY_RULE_13_POST_DATA] = np.empty(shape=(5, 0))
        self._metrics[KEY_RULE_13_WT_DATA] = np.empty(shape=(0,))
        self._metrics[KEY_RULE_13_POST_DATA_L1] = np.empty(shape=(5, 0))
        self._metrics[KEY_RULE_13_WT_DATA_L1] = np.empty(shape=(0,))
        self._metrics[KEY_OUTPUT_LAYER_PYR_ACTS] = np.empty(shape=(1, 0))
        self._metrics[KEY_HIDDEN_LAYER_PYR_ACTS] = np.empty(shape=(3, 0))
        self._metrics[KEY_HIDDEN_LAYER_INHIB_ACTS] = np.empty(shape=(3, 0))

        self._X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self._Y = np.array([0, 1, 1, 0])
        self._current_X: Optional[np.ndarray] = None
        self._current_label: Optional[int] = None

        self._rng_labels = np.random.default_rng(seed=label_init_seed)

    def extract_metrics(self):
        data1 = self._metrics[KEY_LAYER_1]
        data2 = self._metrics[KEY_LAYER_2]
        data3 = self._metrics[KEY_RULE_13_POST_DATA]
        data4 = self._metrics[KEY_RULE_13_WT_DATA]
        data5 = self._metrics[KEY_RULE_13_POST_DATA_L1]
        data6 = self._metrics[KEY_RULE_13_WT_DATA_L1]
        data7 = self._metrics[KEY_OUTPUT_LAYER_PYR_ACTS]
        data8 = self._metrics[KEY_HIDDEN_LAYER_PYR_ACTS]
        data9 = self._metrics[KEY_HIDDEN_LAYER_INHIB_ACTS]
        return [
            Graph(type=GraphType.LINE,
                  title="Layer 1 Apical MPs",
                  precision=2,
                  series=[
                      Serie("Apical MP 1", data1[0].tolist()),
                      Serie("Apical MP 2", data1[1].tolist()),
                  ],
                  xaxis="Training steps",
                  yaxis="Membrane potential (mV)"),
            Graph(type=GraphType.LINE,
                  title="Layer 2 Apical MPs",
                  precision=2,
                  series=[
                      Serie("Apical MP 1", data2[0].tolist()),
                      Serie("Apical MP 2", data2[1].tolist()),
                      Serie("Apical MP 3", data2[2].tolist()),
                  ],
                  xaxis="Training steps",
                  yaxis="Membrane potential (mV)"),
            Graph(type=GraphType.LINE,
                  title="Learning Rule PP_FF Triggers",
                  precision=2,
                  series=[
                      Serie("Soma act", data3[0].tolist()),
                      Serie("Basal hat act", data3[1].tolist()),
                      Serie("Post soma MP", data3[2].tolist()),
                      Serie("Post basal MP", data3[3].tolist()),
                      Serie("Post val", data3[4].tolist()),
                  ],
                  # xaxis="Training steps",
                  yaxis="..."),
            Graph(type=GraphType.LINE,
                  title="Learning Rule PP_FF wts",
                  precision=2,
                  series=[
                      Serie("PP_FF wt", data4.tolist()),
                  ],
                  xaxis="Training steps",
                  yaxis="..."),
            Graph(type=GraphType.LINE,
                  title="Learning Rule PP_FF Triggers L1",
                  precision=2,
                  series=[
                      Serie("Soma act", data5[0].tolist()),
                      Serie("Basal hat act", data5[1].tolist()),
                      Serie("Post soma MP", data5[2].tolist()),
                      Serie("Post basal MP", data5[3].tolist()),
                      Serie("Post val", data5[4].tolist()),
                  ],
                  xaxis="Training steps",
                  yaxis="..."),
            Graph(type=GraphType.LINE,
                  title="Learning Rule PP_FF wts L1",
                  precision=2,
                  series=[
                      Serie("PP_FF wt L1", data6.tolist()),
                  ],
                  xaxis="Training steps",
                  yaxis="..."),
            Graph(type=GraphType.LINE,
                  title="Layer 3 Soma Activations",
                  precision=2,
                  series=[
                      Serie("Soma act", data7[0].tolist()),
                  ],
                  xaxis="Training steps",
                  yaxis="Output activation"),
            Graph(type=GraphType.COLUMN,
                  title="Output Activations",
                  precision=4,
                  series=[
                      Serie("Neuron 1", [self._metrics[KEY_OUTPUT_LAYER_PYR_ACTS][0][399],  # 0.6535
                                         self._metrics[KEY_OUTPUT_LAYER_PYR_ACTS][0][400],  # 0.7165
                                         self._metrics[KEY_OUTPUT_LAYER_PYR_ACTS][0][598],  # 0.7310
                                         self._metrics[KEY_OUTPUT_LAYER_PYR_ACTS][0][599]]), # 0.7309
                  ],
                  categories=["Before Nudge", "Nudged", "After learning", "Nudged removed"],
                  yaxis="Activation level"),
            Graph(type=GraphType.LINE,
                  title="Layer 2 Soma Activations",
                  precision=2,
                  series=[
                      Serie("Soma act 1", data8[0].tolist()),
                      Serie("Soma act 2", data8[1].tolist()),
                      Serie("Soma act 3", data8[2].tolist()),
                  ],
                  xaxis="Training steps",
                  yaxis="Output activation"),
            Graph(type=GraphType.LINE,
                  title="Layer 2 Inhib Activations",
                  precision=2,
                  series=[
                      Serie("Soma act 1", data9[0].tolist()),
                      Serie("Soma act 2", data9[1].tolist()),
                      Serie("Soma act 3", data9[2].tolist()),
                  ],
                  xaxis="Training steps",
                  yaxis="Output activation")
        ]

    def hook_pre_train_step(self):
        index = self._rng_labels.integers(low=0, high=len(self._X))
        self._current_X = self._X[index]
        self._current_label = self._Y[index]

    def hook_post_train_step(self):
        l1, l2, l3 = self.layers
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

        self._gather_layer_metrics(KEY_RULE_13_POST_DATA_L1, KEY_RULE_13_WT_DATA_L1, l2)
        self._gather_layer_metrics(KEY_RULE_13_POST_DATA, KEY_RULE_13_WT_DATA, l3)

        self._metrics[KEY_OUTPUT_LAYER_PYR_ACTS] = np.append(
            self._metrics[KEY_OUTPUT_LAYER_PYR_ACTS],
            create_column_vector(*map(lambda p: p.soma_act, l3.pyrs)),
            axis=1)

        self._metrics[KEY_HIDDEN_LAYER_PYR_ACTS] = np.append(
            self._metrics[KEY_HIDDEN_LAYER_PYR_ACTS],
            create_column_vector(*map(lambda p: p.soma_act, l2.pyrs)),
            axis=1)

        self._metrics[KEY_HIDDEN_LAYER_INHIB_ACTS] = np.append(
            self._metrics[KEY_HIDDEN_LAYER_INHIB_ACTS],
            create_column_vector(*map(lambda p: p.soma_act, l2.inhibs)),
            axis=1)

    def _gather_layer_metrics(self, key_post: str, key_wts: str, layer: Layer):
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

    def do_ff_sweep(self):
        """Standard FF sweep"""
        logger.debug("Starting FF sweep...")

        # Iterates over layers from start to end. From ai.utils.
        for prev, layer in iter_with_prev(self.layers):  # Yields prev layer and current layer
            if prev is None:
                layer.apply_inputs_to_test_self_predictive_convergence(self._current_X)
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

    def train_1_step(self, nudge_predicate: bool):
        """
        This is the concrete version of the abstract train-1-step defined in superclass Experiment.
        :param nudge_predicate:
        :return:
        """
        l1, l2, l3 = self.layers
        l2.adjust_wts_lat_pi()  # adjust lateral PI wts in Layer 2

        # Adjust FF wts projecting to Layer 3.
        l3.adjust_wts_pp_ff(l2)  # adjust FF wts projecting to Layer 3

        # continue learning
        l1.adjust_wts_lat_pi()  # adjust lateral PI wts in Layer 1
        l2.adjust_wts_pp_ff(l1)  # adjust FF wts projecting to Layer 2

        # Do FF and FB sweeps so wt changes show their effects.
        self.do_ff_sweep()
        if nudge_predicate:
            l3.nudge_output_layer_neurons(self._current_label, lambda_nudge=0.8)
        self.do_fb_sweep()

    def run(self, self_prediction_steps: int, training_steps: int, after_training_steps: int):
        logger.info("START: Performing nudge experiment with rules 16b and 13.")
        # self.do_ff_sweep()  # prints state
        # logger.info("Finished 1st FF sweep: pilot_exp_1b_concat_2b")

        # self.do_fb_sweep()  # prints state
        # logger.info("Finished 1st FB sweep: pilot_exp_1b_concat_2b")

        logger.info(f"Starting training {self_prediction_steps} steps to XOR self predictive.")
        # trains and SAVES apical results in 'datasets' attr
        self.train(self_prediction_steps, nudge_predicate=False)

        # logger.info("Calling function to impose nudge.")
        # self._nudge_output_layer()

        logger.info(f"Starting training {training_steps} steps for XOR exp")

        # trains and APPENDS apical results in 'datasets' attr
        self.train(training_steps, nudge_predicate=True)
        logger.info(f"Finished training {training_steps} steps for XOR exp")

        self.train(after_training_steps, nudge_predicate=False)
        logger.info(f"Finished training {training_steps} steps for XOR exp")
        self.print_pyr_activations_all_layers_topdown()  # print activations while nudging is still on

        # self.do_ff_sweep()  # to get new activations without nudging

        logger.info("Final activations after nudging is removed")
        self.print_pyr_activations_all_layers_topdown()  # shows the true effect of learning
        logger.info("FINISH: Performing nudge experiment with rules 16b and 13.")

    def _nudge_output_layer(self):
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
        self.do_fb_sweep()  # prints state

        logger.info("Finished 1st FB sweep after nudge: pilot_exp_2b")  # shows effect of nudge in earlier layers
        self.print_pyr_activations_all_layers_topdown()
