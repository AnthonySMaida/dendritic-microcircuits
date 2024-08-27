import numpy as np

from ai.colorized_logger import get_logger
from ai.experiments.Experiment import Experiment
from ai.utils import create_column_vector, iter_with_prev
from metrics import Graph, Serie, GraphType

logger = get_logger('ai.experiments.NudgeExperiment')

KEY_LAYER_1 = "layer1"
KEY_LAYER_2 = "layer2"
KEY_RULE_13_POST_DATA = "rule13_post_data"
KEY_RULE_13_WT_DATA_L2_PYR0 = "rule13_wt_data_l2_pyr0"
KEY_RULE_13_WT_DATA_L2_PYR1 = "rule13_wt_data_l2_pyr1"
KEY_RULE_16B_WT_DATA_L2_PYR0 = "rule16b_wt_data_l2_pyr0"
KEY_RULE_16B_WT_DATA_L2_PYR1 = "rule16b_wt_data_l2_pyr1"
KEY_RULE_16B_WT_DATA_L2_PYR2 = "rule16b_wt_data_l2_pyr2"
KEY_RULE_13_POST_DATA_L1 = "rule13_post_data_l1"
KEY_RULE_13_WT_DATA_L1 = "rule13_wt_data_l1"
KEY_OUTPUT_LAYER_VALUES = "output_layer_values"
KEY_OUTPUT_LAYER_BASAL_MPS = "output_layer_basal_mps"
KEY_HIDDEN_LAYER_PYR_ACT_VALUES = "hidden_layer_pyr_act_values"
KEY_HIDDEN_LAYER_APICAL_FB_VALUES = "hidden_layer_calc_pyr_apical_values"
KEY_HIDDEN_LAYER_APICAL_LAT_VALUES = "hidden_layer_calc_pyr_lat_values"
KEY_HIDDEN_LAYER_INHIB_ACT_VALUES = "hidden_layer_inhib_act_values"
KEY_L2_BASAL_MINUS_SOMA_PYR_MP = "l2_basal_minus_soma_pyr_mp"
KEY_L2_APICAL_MINUS_SOMA_PYR_MP = "l2_apical_minus_soma_pyr_mp"


class BasicNudgeExper(Experiment):
    def __init__(self, wt_init_seed: int, beta: float, learning_rate: float, nudge1: float, nudge2: float):
        # call the constructor of the parent class (aka superclass)
        super().__init__(wt_init_seed, beta, learning_rate)

        self._nudge1 = nudge1
        self._nudge2 = nudge2

        self._metrics[KEY_LAYER_1] = np.empty(shape=(2, 0))
        self._metrics[KEY_LAYER_2] = np.empty(shape=(3, 0))
        self._metrics[KEY_RULE_13_POST_DATA] = np.empty(shape=(5, 0))
        self._metrics[KEY_RULE_13_WT_DATA_L2_PYR0] = np.empty(shape=(3, 0))
        self._metrics[KEY_RULE_13_WT_DATA_L2_PYR1] = np.empty(shape=(3, 0))
        self._metrics[KEY_RULE_16B_WT_DATA_L2_PYR0] = np.empty(shape=(3, 0))
        self._metrics[KEY_RULE_16B_WT_DATA_L2_PYR1] = np.empty(shape=(3, 0))
        self._metrics[KEY_RULE_16B_WT_DATA_L2_PYR2] = np.empty(shape=(3, 0))
        self._metrics[KEY_RULE_13_POST_DATA_L1] = np.empty(shape=(5, 0))
        self._metrics[KEY_RULE_13_WT_DATA_L1] = np.empty(shape=(0,))
        self._metrics[KEY_OUTPUT_LAYER_VALUES] = np.empty(shape=(2, 0))
        self._metrics[KEY_OUTPUT_LAYER_BASAL_MPS] = np.empty(shape=(2, 0))
        self._metrics[KEY_HIDDEN_LAYER_PYR_ACT_VALUES] = np.empty(shape=(3, 0))
        self._metrics[KEY_HIDDEN_LAYER_APICAL_FB_VALUES] = np.empty(shape=(3, 0))
        self._metrics[KEY_HIDDEN_LAYER_APICAL_LAT_VALUES] = np.empty(shape=(3, 0))
        self._metrics[KEY_HIDDEN_LAYER_INHIB_ACT_VALUES] = np.empty(shape=(3, 0))
        self._metrics[KEY_L2_BASAL_MINUS_SOMA_PYR_MP] = np.empty(shape=(3, 0))
        self._metrics[KEY_L2_APICAL_MINUS_SOMA_PYR_MP] = np.empty(shape=(3, 0))

    def __do_ff_sweep(self):
        """Standard FF sweep"""

        # Iterates over layers from start to end. From ai.utils.
        for prev, layer in iter_with_prev(self.layers):  # Yields prev layer and current layer
            if prev is None:
                layer.apply_inputs_to_test_self_predictive_convergence([.5, .5])
            else:
                layer.update_pyrs_basal_and_soma_ff(prev)
            layer.update_dend_mps_via_ip()  # update inhibs in layer

    def __do_fb_sweep(self):
        """Standard FB sweep"""
        for prev, layer in iter_with_prev(reversed(self.layers)):  # [l3, l2, l1]
            if prev is None:  # Skip first layer (L3)
                continue
            # update current layer pyrs using somatic pyr acts from previous layer and inhib acts from current layer
            layer.update_pyrs_apical_soma_fb(prev)  # in 2nd iter, prev = l3 and layer = l2

    def __train_1_step_rule_16b_and_rule_13(self, use_nudge=False, use_rule_ip=False):
        """
        Learning step that uses both rules 16b and 13.
        Does one training step.
        """
        l1, l2, l3 = self.layers
        l2.adjust_wts_lat_pi()  # adjust lateral PI wts in Layer 2

        # Adjust FF wts projecting to Layer 3.
        l3.adjust_wts_pp_ff(l2)  # adjust FF wts projecting to Layer 3

        # continue learning
        l1.adjust_wts_lat_pi()  # adjust lateral PI wts in Layer 1

        if use_rule_ip:  # also known as Rule 16a
            l1.adjust_wts_lat_ip()  # adjust lateral IP wts in Layer 1

        l2.adjust_wts_pp_ff(l1)  # adjust FF wts projecting to Layer 2

        # Do FF and FB sweeps so wt changes show their effects.
        self.__do_ff_sweep()
        if use_nudge:
            l3.nudge_output_layer_neurons(self._nudge1, self._nudge2, lambda_nudge=0.8)
        self.__do_fb_sweep()

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
        last_layer.nudge_output_layer_neurons(self._nudge1, self._nudge2, lambda_nudge=0.9)
        logger.debug("Layer %d activations after nudge.", last_layer.id_num)
        last_layer.print_pyr_activations()

        logger.info("Starting FB sweep")
        self.__do_fb_sweep()  # prints state

        logger.info("Finished 1st FB sweep after nudge: pilot_exp_2b")  # shows effect of nudge in earlier layers
        self.print_pyr_activations_all_layers_topdown()

    def _hook_post_train_step(self):
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

        soma_act = l3.pyr_soma_acts()[0]
        basal_hat_act = l3.pyr_basal_hat_acts()[0]
        post_soma_mp = l3.pyr_soma_mps()[0]
        post_basal_mp = l3.pyr_basal_mps()[0]
        post_val = post_soma_mp - post_basal_mp

        self._metrics[KEY_RULE_13_POST_DATA] = np.append(
            self._metrics[KEY_RULE_13_POST_DATA],
            create_column_vector(soma_act, basal_hat_act, post_soma_mp, post_basal_mp, post_val),
            axis=1)

        self._metrics[KEY_RULE_13_WT_DATA_L2_PYR0] = np.append(
            self._metrics[KEY_RULE_13_WT_DATA_L2_PYR0],
            create_column_vector(l3.pyrs[0].W_PP_ff[0], l3.pyrs[0].W_PP_ff[1], l3.pyrs[0].W_PP_ff[2]),
            axis=1
        )

        self._metrics[KEY_RULE_13_WT_DATA_L2_PYR1] = np.append(
            self._metrics[KEY_RULE_13_WT_DATA_L2_PYR1],
            create_column_vector(l3.pyrs[1].W_PP_ff[0], l3.pyrs[1].W_PP_ff[1], l3.pyrs[1].W_PP_ff[2]),
            axis=1
        )

        self._metrics[KEY_RULE_16B_WT_DATA_L2_PYR0] = np.append(
            self._metrics[KEY_RULE_16B_WT_DATA_L2_PYR0],
            create_column_vector(l2.pyrs[0].W_PI_lat[0], l2.pyrs[0].W_PI_lat[1], l2.pyrs[0].W_PI_lat[2]),
            axis=1
        )

        self._metrics[KEY_RULE_16B_WT_DATA_L2_PYR1] = np.append(
            self._metrics[KEY_RULE_16B_WT_DATA_L2_PYR1],
            create_column_vector(l2.pyrs[1].W_PI_lat[0], l2.pyrs[1].W_PI_lat[1], l2.pyrs[1].W_PI_lat[2]),
            axis=1
        )

        self._metrics[KEY_RULE_16B_WT_DATA_L2_PYR2] = np.append(
            self._metrics[KEY_RULE_16B_WT_DATA_L2_PYR2],
            create_column_vector(l2.pyrs[2].W_PI_lat[0], l2.pyrs[2].W_PI_lat[1], l2.pyrs[2].W_PI_lat[2]),
            axis=1
        )

        soma_act = l2.pyr_soma_acts()[0]
        basal_hat_act = l2.pyr_basal_hat_acts()[0]
        post_soma_mp = l2.pyr_soma_mps()[0]
        post_basal_mp = l2.pyr_basal_mps()[0]
        post_val2 = post_soma_mp - post_basal_mp

        self._metrics[KEY_RULE_13_POST_DATA_L1] = np.append(
            self._metrics[KEY_RULE_13_POST_DATA_L1],
            create_column_vector(soma_act, basal_hat_act, post_soma_mp, post_basal_mp, post_val2),
            axis=1)
        self._metrics[KEY_RULE_13_WT_DATA_L1] = np.append(self._metrics[KEY_RULE_13_WT_DATA_L1],
                                                          l2.pyrs[0].W_PP_ff[0])

        self._metrics[KEY_OUTPUT_LAYER_VALUES] = np.append(
            self._metrics[KEY_OUTPUT_LAYER_VALUES],
            create_column_vector(*map(lambda p: p.soma_act, l3.pyrs)),
            axis=1)

        self._metrics[KEY_OUTPUT_LAYER_BASAL_MPS] = np.append(
            self._metrics[KEY_OUTPUT_LAYER_BASAL_MPS],
            create_column_vector(*map(lambda p: p.basal_mp, l3.pyrs)),
            axis=1)

        self._metrics[KEY_HIDDEN_LAYER_PYR_ACT_VALUES] = np.append(
            self._metrics[KEY_HIDDEN_LAYER_PYR_ACT_VALUES],
            create_column_vector(*map(lambda p: p.soma_act, l2.pyrs)),
            axis=1)

        self._metrics[KEY_HIDDEN_LAYER_APICAL_FB_VALUES] = np.append(
            self._metrics[KEY_HIDDEN_LAYER_APICAL_FB_VALUES],
            create_column_vector(*map(lambda p: p.apical_fb, l2.pyrs)),
            axis=1)

        self._metrics[KEY_HIDDEN_LAYER_APICAL_LAT_VALUES] = np.append(
            self._metrics[KEY_HIDDEN_LAYER_APICAL_LAT_VALUES],
            create_column_vector(*map(lambda p: p.apical_lat, l2.pyrs)),
            axis=1)

        self._metrics[KEY_L2_BASAL_MINUS_SOMA_PYR_MP] = np.append(
            self._metrics[KEY_L2_BASAL_MINUS_SOMA_PYR_MP],
            create_column_vector(*map(lambda p: p.basal_minus_soma_mp, l2.pyrs)),
            axis=1)

        self._metrics[KEY_L2_APICAL_MINUS_SOMA_PYR_MP] = np.append(
            self._metrics[KEY_L2_APICAL_MINUS_SOMA_PYR_MP],
            create_column_vector(*map(lambda p: p.apical_minus_soma_mp, l2.pyrs)),
            axis=1)

        self._metrics[KEY_HIDDEN_LAYER_INHIB_ACT_VALUES] = np.append(
            self._metrics[KEY_HIDDEN_LAYER_INHIB_ACT_VALUES],
            create_column_vector(*map(lambda p: p.soma_act, l2.inhibs)),
            axis=1)

    def _train_1_step(self, nudge_predicate: bool):  # Signature matched its abstract method b/c *args can be empty.
        """
        This is the concrete version of the abstract train-1-step defined in superclass Experiment.
        :param nudge_predicate:
        :return:
        """
        self.__train_1_step_rule_16b_and_rule_13(use_nudge=nudge_predicate)  # defined in superclass

    def extract_metrics(self):
        data_l1 = self._metrics[KEY_LAYER_1]
        data_l2 = self._metrics[KEY_LAYER_2]
        triggers_l2 = self._metrics[KEY_RULE_13_POST_DATA]
        wts_r13_l2_pyr0 = self._metrics[KEY_RULE_13_WT_DATA_L2_PYR0]
        wts_r13_l2_pyr1 = self._metrics[KEY_RULE_13_WT_DATA_L2_PYR1]
        triggers_l1 = self._metrics[KEY_RULE_13_POST_DATA_L1]
        wts_l1 = self._metrics[KEY_RULE_13_WT_DATA_L1]
        soma_acts_l3 = self._metrics[KEY_OUTPUT_LAYER_VALUES]
        basal_mps_l3 = self._metrics[KEY_OUTPUT_LAYER_BASAL_MPS]
        wts_16b_l2_pyr0 = self._metrics[KEY_RULE_16B_WT_DATA_L2_PYR0]
        wts_16b_l2_pyr1 = self._metrics[KEY_RULE_16B_WT_DATA_L2_PYR1]
        wts_16b_l2_pyr2 = self._metrics[KEY_RULE_16B_WT_DATA_L2_PYR2]
        soma_acts_l2 = self._metrics[KEY_HIDDEN_LAYER_PYR_ACT_VALUES]
        apical_fb_l2 = self._metrics[KEY_HIDDEN_LAYER_APICAL_FB_VALUES]
        apical_lat_l2 = self._metrics[KEY_HIDDEN_LAYER_APICAL_LAT_VALUES]
        inhib_soma_acts_l2 = self._metrics[KEY_HIDDEN_LAYER_INHIB_ACT_VALUES]
        basal_minus_soma_mp_l2 = self._metrics[KEY_L2_BASAL_MINUS_SOMA_PYR_MP]
        apical_minus_soma_mp_l2 = self._metrics[KEY_L2_APICAL_MINUS_SOMA_PYR_MP]

        return [
            Graph(type=GraphType.LINE,
                  title="Layer 1 Apical MPs",
                  precision=2,
                  series=[
                      Serie("Apical MP 1", data_l1[0].tolist()),
                      Serie("Apical MP 2", data_l1[1].tolist()),
                  ],
                  xaxis="Training steps",
                  yaxis="Membrane potential (mV)"),
            Graph(type=GraphType.LINE,
                  title="Layer 2 Apical MPs",
                  precision=2,
                  series=[
                      Serie("Apical MP 1", data_l2[0].tolist()),
                      Serie("Apical MP 2", data_l2[1].tolist()),
                      Serie("Apical MP 3", data_l2[2].tolist()),
                  ],
                  xaxis="Training steps",
                  yaxis="Membrane potential (mV)"),
            Graph(type=GraphType.LINE,
                  title="LR W_PP_FF(3,2) Triggers in L3",
                  precision=2,
                  series=[
                      #Serie("Soma act", triggers_l2[0].tolist()),
                      #Serie("Basal hat act", triggers_l2[1].tolist()),
                      Serie("L3 post soma MP", triggers_l2[2].tolist()),
                      Serie("L3 post basal MP", triggers_l2[3].tolist()),
                      Serie("L3 post val", triggers_l2[4].tolist()),
                  ],
                  yaxis="..."),
            # This and the next graph interferred w/ each other when series strings were the same.
            Graph(type=GraphType.LINE,
                  title="PP_FF wts projecting to L3 Pyr0",
                  precision=2,
                  series=[
                      Serie("PP_FF wt[0] to P0", wts_r13_l2_pyr0[0].tolist()),
                      Serie("PP_FF wt[1] to P0", wts_r13_l2_pyr0[1].tolist()),
                      Serie("PP_FF wt[2] to P0", wts_r13_l2_pyr0[2].tolist())
                  ],
                  xaxis="Training steps",
                  yaxis="FF wt values to Pyr0 in L3"),
            Graph(type=GraphType.LINE,
                  title="PP_FF wts projecting to L3 Pyr1",
                  precision=2,
                  series=[
                      Serie("PP_FF wt[0] to P1", wts_r13_l2_pyr1[0].tolist()),
                      Serie("PP_FF wt[1] to P1", wts_r13_l2_pyr1[1].tolist()),
                      Serie("PP_FF wt[2] to P1", wts_r13_l2_pyr1[2].tolist())
                  ],
                  xaxis="Training steps",
                  yaxis="FF wt values to Pyr1 in L3"),
            Graph(type=GraphType.LINE,
                  title="PI_lat wts projecting to L2 Pyr0",
                  precision=2,
                  series=[
                      Serie("PI_lat wt[0] to P0", wts_16b_l2_pyr0[0].tolist()),
                      Serie("PI_lat wt[1] to P0", wts_16b_l2_pyr0[1].tolist()),
                      Serie("PI_lat wt[2] to P0", wts_16b_l2_pyr0[2].tolist())
                  ],
                  xaxis="Training steps",
                  yaxis="Lat wt values to Pyr0 in L2"),
            Graph(type=GraphType.LINE,
                  title="PI_lat wts projecting to L2 Pyr1",
                  precision=2,
                  series=[
                      Serie("PI_lat wt[0] to P1", wts_16b_l2_pyr1[0].tolist()),
                      Serie("PI_lat wt[1] to P1", wts_16b_l2_pyr1[1].tolist()),
                      Serie("PI_lat wt[2] to P1", wts_16b_l2_pyr1[2].tolist())
                  ],
                  xaxis="Training steps",
                  yaxis="Lat wt values to Pyr1 in L2"),
            Graph(type=GraphType.LINE,
                  title="PI_lat wts projecting to L2 Pyr2",
                  precision=2,
                  series=[
                      Serie("PI_lat wt[0] to P2", wts_16b_l2_pyr2[0].tolist()),
                      Serie("PI_lat wt[1] to P2", wts_16b_l2_pyr2[1].tolist()),
                      Serie("PI_lat wt[2] to P2", wts_16b_l2_pyr2[2].tolist())
                  ],
                  xaxis="Training steps",
                  yaxis="Lat wt values to Pyr2 in L2"),
            Graph(type=GraphType.LINE,
                  title="LR W_PP_FF(2,1) Triggers L2",
                  precision=2,
                  series=[
                      #Serie("Soma act", triggers_l1[0].tolist()),
                      #Serie("Basal hat act", triggers_l1[1].tolist()),
                      Serie("L2 post soma MP", triggers_l1[2].tolist()),
                      Serie("L2 post basal MP", triggers_l1[3].tolist()),
                      Serie("L2 post val", triggers_l1[4].tolist()),
                  ],
                  xaxis="Training steps",
                  yaxis="..."),
            Graph(type=GraphType.LINE,
                  title="PP_FF wt projecting to L2 Pyr0",
                  precision=2,
                  series=[
                      Serie("PP_FF wt L1", wts_l1.tolist()),
                  ],
                  xaxis="Training steps",
                  yaxis="..."),
            Graph(type=GraphType.LINE,
                  title="Layer 3 Pyr Soma Activations",
                  precision=2,
                  series=[
                      Serie("Act Pyr 0", soma_acts_l3[0].tolist()),
                      Serie("Act Pyr 1", soma_acts_l3[1].tolist()),
                  ],
                  xaxis="Training steps",
                  yaxis="Output activation"),
            Graph(type=GraphType.LINE,
                  title="Layer 3 Pyr Basal MPs",
                  precision=2,
                  series=[
                      Serie("Act Pyr 0", basal_mps_l3[0].tolist()),
                      Serie("Act Pyr 1", basal_mps_l3[1].tolist()),
                  ],
                  xaxis="Training steps",
                  yaxis="Basal MP"),
            Graph(type=GraphType.LINE,
                  title="Layer 2 Pyr Soma Activations",
                  precision=2,
                  series=[
                      Serie("Act Pyr 0", soma_acts_l2[0].tolist()),
                      Serie("Act Pyr 1", soma_acts_l2[1].tolist()),
                      Serie("Act Pyr 2", soma_acts_l2[2].tolist()),
                  ],
                  xaxis="Training steps",
                  yaxis="Pyr Hidden activation"),
            Graph(type=GraphType.LINE,
                  title="L2 Basal - Soma MP Pyr",
                  precision=2,
                  series=[
                      Serie("Basal-soma mp Pyr 0", basal_minus_soma_mp_l2[0].tolist()),
                      Serie("Basal-soma mp Pyr 1", basal_minus_soma_mp_l2[1].tolist()),
                      Serie("Basal-soma mp Pyr 2", basal_minus_soma_mp_l2[2].tolist()),
                  ],
                  xaxis="Training steps",
                  yaxis="L2 Pyr basal - soma mp"),
            Graph(type=GraphType.LINE,
                  title="L2 Apical - Soma MP Pyr",
                  precision=2,
                  series=[
                      Serie("Basal-soma mp Pyr 0", apical_minus_soma_mp_l2[0].tolist()),
                      Serie("Basal-soma mp Pyr 1", apical_minus_soma_mp_l2[1].tolist()),
                      Serie("Basal-soma mp Pyr 2", apical_minus_soma_mp_l2[2].tolist()),
                  ],
                  xaxis="Training steps",
                  yaxis="L2 Pyr apical - soma mp"),
            Graph(type=GraphType.LINE,
                  title="Layer 2 Pyr Apical FB Values",
                  precision=2,
                  series=[
                      Serie("Apical FB Pyr 0", apical_fb_l2[0].tolist()),
                      Serie("Apical FB Pyr 1", apical_fb_l2[1].tolist()),
                      Serie("Apical FB Pyr 2", apical_fb_l2[2].tolist()),
                  ],
                  xaxis="Training steps",
                  yaxis="Pyr Hidden apical FB values"),
            Graph(type=GraphType.LINE,
                  title="Layer 2 Pyr Apical LAT Values",
                  precision=2,
                  series=[
                      Serie("Apical LAT Pyr 0", apical_lat_l2[0].tolist()),
                      Serie("Apical LAT Pyr 1", apical_lat_l2[1].tolist()),
                      Serie("Apical LAT Pyr 2", apical_lat_l2[2].tolist()),
                  ],
                  xaxis="Training steps",
                  yaxis="Pyr Hidden apical LAT values"),
            Graph(type=GraphType.LINE,
                  title="Layer 2 Inhib Soma Activations",
                  precision=2,
                  series=[
                      Serie("Act Inhib 0", inhib_soma_acts_l2[0].tolist()),
                      Serie("Act Inhib 1", inhib_soma_acts_l2[1].tolist()),
                      Serie("Act Inhib 2", inhib_soma_acts_l2[2].tolist()),
                  ],
                  xaxis="Training steps",
                  yaxis="Inhib Hidden activation"),
            Graph(type=GraphType.COLUMN,
                  title="Output Activations",
                  precision=4,
                  series=[
                      Serie("Neuron 1", [self._metrics[KEY_OUTPUT_LAYER_VALUES][0][399],  # 0.6535
                                         self._metrics[KEY_OUTPUT_LAYER_VALUES][0][400],  # 0.7165
                                         self._metrics[KEY_OUTPUT_LAYER_VALUES][0][598],  # 0.7310
                                         self._metrics[KEY_OUTPUT_LAYER_VALUES][0][599]]),  # 0.7309
                      Serie("Neuron 2", [self._metrics[KEY_OUTPUT_LAYER_VALUES][1][399],  # 0.6051,
                                         self._metrics[KEY_OUTPUT_LAYER_VALUES][1][400],  # 0.5213,
                                         self._metrics[KEY_OUTPUT_LAYER_VALUES][1][598],  # 0.5000,
                                         self._metrics[KEY_OUTPUT_LAYER_VALUES][1][599]])  # 0.5002
                  ],
                  categories=["Before Nudge", "Nudged", "After learning", "Nudged removed"],
                  yaxis="Activation level")
        ]

    def run(self, self_prediction_steps: int, training_steps: int, after_training_steps: int):
        logger.info("START: Performing nudge experiment with rules 16b and 13.")
        self.__do_ff_sweep()  # prints state
        logger.info("Finished 1st FF sweep: nudge_experiment")

        self.__do_fb_sweep()  # prints state
        logger.info("Finished 1st FB sweep: nudge_experiment")

        logger.info(f"Starting training {self_prediction_steps} steps to 1b2b self predictive.")
        # trains and SAVES apical results in 'datasets' attr
        self.train(self_prediction_steps, nudge_predicate=False)

        logger.info("Calling function to impose nudge.")
        self.__nudge_output_layer()

        logger.info(f"Starting training {training_steps} steps for p_exp 3b")

        # trains and APPENDS apical results in 'datasets' attr
        self.train(training_steps, nudge_predicate=True)
        logger.info(f"Finished training {training_steps} steps for p_exp 3b")

        self.train(after_training_steps, nudge_predicate=False)
        logger.info(f"Finished training {training_steps} steps for p_exp 3b")
        self.print_pyr_activations_all_layers_topdown()  # print activations while nudging is still on

        # self.do_ff_sweep()  # to get new activations without nudging

        logger.info("Final activations after nudging is removed")
        self.print_pyr_activations_all_layers_topdown()  # shows the true effect of learning
        logger.info("FINISH: Performing nudge experiment with rules 16b and 13.")
