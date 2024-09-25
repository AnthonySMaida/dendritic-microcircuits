import numpy as np
from werkzeug.datastructures import MultiDict

from ai.experiments.Experiment import Experiment
from ai.utils import create_column_vector, iter_with_prev
from metrics import Graph, Serie, GraphType

KEY_LAYER_1 = "layer1"
KEY_LAYER_2 = "layer2"
KEY_RULE_13_POST_DATA = "rule13_post_data"
KEY_RULE_13_WT_DATA_L2_PYR0 = "rule13_wt_data_l2_pyr0"
KEY_RULE_13_WT_DATA_L2_PYR1 = "rule13_wt_data_l2_pyr1"
KEY_RULE_16B_WT_DATA_L2_PYR0 = "rule16b_wt_data_l2_pyr0"
KEY_RULE_16B_WT_DATA_L2_PYR1 = "rule16b_wt_data_l2_pyr1"
KEY_RULE_16B_WT_DATA_L2_PYR2 = "rule16b_wt_data_l2_pyr2"
KEY_RULE_13_POST_DATA_L1_0_TO_L2_0 = "rule13_post_data_l1[0]_to_l2[0]"
KEY_RULE_13_POST_DATA_L1_0_TO_L2_1 = "rule13_post_data_l1[0]_to_l2[01]"
KEY_RULE_13_WT_DATA_L1_TO_L2_PYR0 = "rule13_wt_data_l1_to_l2_pyr0"
KEY_RULE_13_WT_DATA_L1_TO_L2_PYR1 = "rule13_wt_data_l1_to_l2_pyr1"
KEY_OUTPUT_LAYER_VALUES = "output_layer_values"
KEY_OUTPUT_LAYER_SOMA_MPS = "output_layer_soma_mps"
KEY_OUTPUT_LAYER_BASAL_MPS = "output_layer_basal_mps"
KEY_HIDDEN_LAYER_PYR_ACT_VALUES = "hidden_layer_pyr_act_values"
KEY_HIDDEN_LAYER_APICAL_FB_VALUES = "hidden_layer_calc_pyr_apical_values"
KEY_HIDDEN_LAYER_APICAL_LAT_VALUES = "hidden_layer_calc_pyr_lat_values"
KEY_HIDDEN_LAYER_WTD_INPUT_FROM_NUDGE = "hidden_layer_wtd_input_from_nudge"
KEY_HIDDEN_LAYER_INHIB_DENDR_MP_VALUES = "hidden_layer_calc_inhib_dendr_mp_values"
KEY_HIDDEN_LAYER_INHIB_SOMA_MP_VALUES = "hidden_layer_calc_inhib_soma_mp_values"
KEY_HIDDEN_LAYER_INHIB_ACT_VALUES = "hidden_layer_inhib_act_values"
KEY_L2_BASAL_MINUS_SOMA_PYR_MP = "l2_basal_minus_soma_pyr_mp"
KEY_L2_APICAL_MINUS_SOMA_PYR_MP = "l2_apical_minus_soma_pyr_mp"


class NudgeExperFB(Experiment):
    def __init__(self, params: MultiDict):
        # call the constructor of the parent class (aka superclass)
        super().__init__(params)

        self.__nudge1 = params.get('nudge1', 1.0, type=float)
        self.__nudge2 = params.get('nudge2', 0.0, type=float)
        self.__nudge_fb_weight = params.get('nudge_fb_weight', 3.0, type=float)
        self.__n_pyr_layer1 = params.get('n_pyr_layer1', 2, type=int)
        self.__n_pyr_layer2 = params.get('n_pyr_layer2', 3, type=int)
        self.__n_pyr_layer3 = params.get('n_pyr_layer3', 2, type=int)
        self._self_prediction_steps = params.get('self_prediction_steps', 400, type=int)
        self._training_steps = params.get('training_steps', 190, type=int)
        self._after_training_steps = params.get('after_training_steps', 10, type=int)

        self._metrics[KEY_LAYER_1] = np.empty(shape=(2, 0))
        self._metrics[KEY_LAYER_2] = np.empty(shape=(3, 0))
        self._metrics[KEY_RULE_13_POST_DATA] = np.empty(shape=(5, 0))
        self._metrics[KEY_RULE_13_WT_DATA_L2_PYR0] = np.empty(shape=(3, 0))
        self._metrics[KEY_RULE_13_WT_DATA_L2_PYR1] = np.empty(shape=(3, 0))
        self._metrics[KEY_RULE_16B_WT_DATA_L2_PYR0] = np.empty(shape=(3, 0))
        self._metrics[KEY_RULE_16B_WT_DATA_L2_PYR1] = np.empty(shape=(3, 0))
        self._metrics[KEY_RULE_16B_WT_DATA_L2_PYR2] = np.empty(shape=(3, 0))
        self._metrics[KEY_RULE_13_POST_DATA_L1_0_TO_L2_0] = np.empty(shape=(5, 0))
        self._metrics[KEY_RULE_13_POST_DATA_L1_0_TO_L2_1] = np.empty(shape=(5, 0))
        self._metrics[KEY_RULE_13_WT_DATA_L1_TO_L2_PYR0] = np.empty(shape=(2, 0))
        self._metrics[KEY_RULE_13_WT_DATA_L1_TO_L2_PYR1] = np.empty(shape=(2, 0))
        self._metrics[KEY_OUTPUT_LAYER_VALUES] = np.empty(shape=(2, 0))
        self._metrics[KEY_OUTPUT_LAYER_SOMA_MPS] = np.empty(shape=(2, 0))
        self._metrics[KEY_OUTPUT_LAYER_BASAL_MPS] = np.empty(shape=(2, 0))
        self._metrics[KEY_HIDDEN_LAYER_PYR_ACT_VALUES] = np.empty(shape=(3, 0))
        self._metrics[KEY_HIDDEN_LAYER_APICAL_FB_VALUES] = np.empty(shape=(3, 0))
        self._metrics[KEY_HIDDEN_LAYER_APICAL_LAT_VALUES] = np.empty(shape=(3, 0))
        self._metrics[KEY_HIDDEN_LAYER_WTD_INPUT_FROM_NUDGE] = np.empty(shape=(3, 0))
        self._metrics[KEY_HIDDEN_LAYER_INHIB_DENDR_MP_VALUES] = np.empty(shape=(3, 0))
        self._metrics[KEY_HIDDEN_LAYER_INHIB_SOMA_MP_VALUES] = np.empty(shape=(3, 0))
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
            layer.update_dend_mps_via_ip()

    def __do_fb_sweep(self, use_nudge=False):  # this version assumes the interneurons have nudging feedback connections
        """Standard FB sweep"""
        nudge_fb_weight = self.__nudge_fb_weight
        for prev, layer in iter_with_prev(reversed(self.layers)):  # [l3, l2, l1]
            if prev is None:  # Skip first layer (L3)
                continue      # Would pass do the same thing?
            if layer.id_num == 2:  # handles the nudging FB connections to Layer 2. Only sends FB to 2 inhibs.
                if use_nudge:
                    layer.inhibs[0].wtd_input_from_nudge = nudge_fb_weight * prev.pyrs[0].soma_act  # for data collection
                    # Eqn below is not exactly right. I looked at Fig 5 and forgot to look at Eqns 10 & 23.
                    layer.inhibs[0].dend_mp += nudge_fb_weight * prev.pyrs[0].soma_act
                    layer.inhibs[0].update_inhib_soma_ff()

                    layer.inhibs[1].wtd_input_from_nudge = nudge_fb_weight * prev.pyrs[1].soma_act  # for data collection
                    layer.inhibs[1].dend_mp += nudge_fb_weight * prev.pyrs[1].soma_act
                    layer.inhibs[1].update_inhib_soma_ff()
                else:
                    layer.inhibs[0].wtd_input_from_nudge = 0.0
                    layer.inhibs[1].wtd_input_from_nudge = 0.0
            # update current layer pyrs using somatic pyr acts from previous layer and inhib acts from current layer
            layer.update_pyrs_apical_soma_fb(prev)  # in 2nd iter, prev = l3 and layer = l2

    def __train_1_step_rule_16b_and_rule_13(self,
                                            use_nudge=False,
                                            use_rule_pi=True,
                                            use_rule_ip=False):
        """
        Learning step that uses both rules 16b and 13.
        Does one training step.
        """
        l1, l2, l3 = self.layers
        # Adjust hidden layer lateral wts if enabled.
        if use_rule_pi:
            l2.adjust_wts_lat_pi()  # adjust lateral PI wts in Layer 2

        if use_rule_ip:
            l2.adjust_wts_lat_ip()

        # Always adjust FF wts projecting to Layer 3.
        l3.adjust_wts_pp_ff(l2)  # adjust FF wts projecting to Layer 3

        # Adjust input layer lateral wts if enabled
        if use_rule_pi:
            l1.adjust_wts_lat_pi()  # adjust lateral PI wts in Layer 1

        if use_rule_ip:  # also known as Rule 16a
            l1.adjust_wts_lat_ip()      # adjust lateral IP wts in Layer 1

        l2.adjust_wts_pp_ff(l1)  # adjust FF wts projecting to Layer 2

        # Do FF and FB sweeps so wt changes show their effects.
        self.__do_ff_sweep()
        if use_nudge:
            l3.nudge_output_layer_neurons(self.__nudge1, self.__nudge2, lambda_nudge=0.9)
            self.__do_fb_sweep(use_nudge=True)
        else:
            self.__do_fb_sweep(use_nudge=False)

    def __nudge_output_layer(self):
        """
        Prints all layer wts.
        Imposes nudge on the output layer and prints output layer activations.
        Does a FF sweep and then prints layer activations in reverse order.
        """
        self.layers[-2].print_fb_and_pi_wts_layer()
        self.layers[-2].print_ff_and_ip_wts_for_layers(self.layers[-1])

        last_layer = self.layers[-1]
        self._logger.debug("Layer %d activations before nudge.", last_layer.id_num)
        last_layer.print_pyr_activations()

        self._logger.info("Imposing nudge now")

        last_layer = self.layers[-1]
        last_layer.nudge_output_layer_neurons(self.__nudge1, self.__nudge2, lambda_nudge=0.9)  # the work
        self._logger.debug("Layer %d activations after nudge.", last_layer.id_num)
        last_layer.print_pyr_activations()

        self._logger.info("Starting FB sweep")
        self.__do_fb_sweep(use_nudge=True)  # prints state

        self._logger.info("Finished 1st FB sweep after nudge: pilot_exp_2b")  # shows effect of nudge in earlier layers
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

        soma_act = l3.pyr_soma_acts[0]
        basal_hat_act = l3.pyr_basal_hat_acts[0]
        post_soma_mp = l3.pyr_soma_mps[0]
        post_basal_mp = l3.pyr_basal_mps[0]
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

        soma_act = l2.pyr_soma_acts[0]
        basal_hat_act = l2.pyr_basal_hat_acts[0]
        post_soma_mp = l2.pyr_soma_mps[0]
        post_basal_mp = l2.pyr_basal_mps[0]
        post_val2 = post_soma_mp - post_basal_mp

        self._metrics[KEY_RULE_13_POST_DATA_L1_0_TO_L2_0] = np.append(
            self._metrics[KEY_RULE_13_POST_DATA_L1_0_TO_L2_0],
            create_column_vector(soma_act, basal_hat_act, post_soma_mp, post_basal_mp, post_val2),
            axis=1)

        self._metrics[KEY_RULE_13_WT_DATA_L1_TO_L2_PYR0] = np.append(
            self._metrics[KEY_RULE_13_WT_DATA_L1_TO_L2_PYR0],
            create_column_vector(l2.pyrs[0].W_PP_ff[0], l2.pyrs[0].W_PP_ff[1]),
            axis=1)

        soma_act = l2.pyr_soma_acts[1]
        basal_hat_act = l2.pyr_basal_hat_acts[1]
        post_soma_mp = l2.pyr_soma_mps[1]
        post_basal_mp = l2.pyr_basal_mps[1]
        post_val2 = post_soma_mp - post_basal_mp

        self._metrics[KEY_RULE_13_POST_DATA_L1_0_TO_L2_1] = np.append(
            self._metrics[KEY_RULE_13_POST_DATA_L1_0_TO_L2_1],
            create_column_vector(soma_act, basal_hat_act, post_soma_mp, post_basal_mp, post_val2),
            axis=1)

        self._metrics[KEY_RULE_13_WT_DATA_L1_TO_L2_PYR1] = np.append(
            self._metrics[KEY_RULE_13_WT_DATA_L1_TO_L2_PYR1],
            create_column_vector(l2.pyrs[1].W_PP_ff[0], l2.pyrs[1].W_PP_ff[1]),
            axis=1)

        self._metrics[KEY_OUTPUT_LAYER_VALUES] = np.append(
            self._metrics[KEY_OUTPUT_LAYER_VALUES],
            create_column_vector(*map(lambda p: p.soma_act, l3.pyrs)),
            axis=1)

        self._metrics[KEY_OUTPUT_LAYER_SOMA_MPS] = np.append(
            self._metrics[KEY_OUTPUT_LAYER_SOMA_MPS],
            create_column_vector(*map(lambda p: p.soma_mp, l3.pyrs)),
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

        self._metrics[KEY_HIDDEN_LAYER_WTD_INPUT_FROM_NUDGE] = np.append(
            self._metrics[KEY_HIDDEN_LAYER_WTD_INPUT_FROM_NUDGE],
            create_column_vector(*map(lambda p: p.wtd_input_from_nudge, l2.inhibs)),
            axis=1)

        self._metrics[KEY_HIDDEN_LAYER_INHIB_DENDR_MP_VALUES] = np.append(
            self._metrics[KEY_HIDDEN_LAYER_INHIB_DENDR_MP_VALUES],
            create_column_vector(*map(lambda p: p.dend_mp, l2.inhibs)),
            axis=1)

        self._metrics[KEY_HIDDEN_LAYER_INHIB_SOMA_MP_VALUES] = np.append(
            self._metrics[KEY_HIDDEN_LAYER_INHIB_SOMA_MP_VALUES],
            create_column_vector(*map(lambda p: p.soma_mp, l2.inhibs)),
            axis=1)

        self._metrics[KEY_HIDDEN_LAYER_INHIB_ACT_VALUES] = np.append(
            self._metrics[KEY_HIDDEN_LAYER_INHIB_ACT_VALUES],
            create_column_vector(*map(lambda p: p.soma_act, l2.inhibs)),
            axis=1)

    def _train_1_step(self, use_nudge: bool, **kwargs):  # Signature matches abstract method b/c *args can be empty.
        """
        This is the concrete version of the abstract train-1-step defined in superclass Experiment.
        :param use_nudge:
        :return:
        """
        self.__train_1_step_rule_16b_and_rule_13(use_nudge=use_nudge, **kwargs)  # defined in superclass

    def _run_init(self):
        self._logger.info("START: Performing nudge experiment with rules 16b and 13.")
        self.__do_ff_sweep()  # prints state
        self._logger.info("Finished 1st FF sweep: nudge_experiment")
        self.__do_fb_sweep()  # prints state
        self._logger.info("Finished 1st FB sweep: nudge_experiment")

    def _run_self_predict(self):
        self.train(self._self_prediction_steps, use_nudge=False)

    def _run_train(self):
        self.train(self._training_steps, use_nudge=True)

    def _run_after_training(self):
        self.train(self._after_training_steps, use_nudge=False)

    def build_network(self, *args, **kwargs):
        self.build_small_three_layer_network(self.__n_pyr_layer1, self.__n_pyr_layer2, self.__n_pyr_layer3)

    def extract_metrics(self):
        data_l1 = self._metrics[KEY_LAYER_1]
        data_l2 = self._metrics[KEY_LAYER_2]
        triggers_l2 = self._metrics[KEY_RULE_13_POST_DATA]
        wts_r13_l2_pyr0 = self._metrics[KEY_RULE_13_WT_DATA_L2_PYR0]
        wts_r13_l2_pyr1 = self._metrics[KEY_RULE_13_WT_DATA_L2_PYR1]
        triggers_l1_0_to_L2_0 = self._metrics[KEY_RULE_13_POST_DATA_L1_0_TO_L2_0]
        triggers_l1_0_to_L2_1 = self._metrics[KEY_RULE_13_POST_DATA_L1_0_TO_L2_1]
        wts_l1_to_l1_pyr0 = self._metrics[KEY_RULE_13_WT_DATA_L1_TO_L2_PYR0]
        wts_l1_to_l1_pyr1 = self._metrics[KEY_RULE_13_WT_DATA_L1_TO_L2_PYR1]
        soma_acts_l3 = self._metrics[KEY_OUTPUT_LAYER_VALUES]
        soma_mps_l3 = self._metrics[KEY_OUTPUT_LAYER_SOMA_MPS]
        basal_mps_l3 = self._metrics[KEY_OUTPUT_LAYER_BASAL_MPS]
        wts_16b_l2_pyr0 = self._metrics[KEY_RULE_16B_WT_DATA_L2_PYR0]
        wts_16b_l2_pyr1 = self._metrics[KEY_RULE_16B_WT_DATA_L2_PYR1]
        wts_16b_l2_pyr2 = self._metrics[KEY_RULE_16B_WT_DATA_L2_PYR2]
        soma_acts_l2 = self._metrics[KEY_HIDDEN_LAYER_PYR_ACT_VALUES]
        apical_fb_l2 = self._metrics[KEY_HIDDEN_LAYER_APICAL_FB_VALUES]
        apical_lat_l2 = self._metrics[KEY_HIDDEN_LAYER_APICAL_LAT_VALUES]
        inhib_wtd_inp_nudge = self._metrics[KEY_HIDDEN_LAYER_WTD_INPUT_FROM_NUDGE]
        inhib_dendr_mps_l2 = self._metrics[KEY_HIDDEN_LAYER_INHIB_DENDR_MP_VALUES]
        inhib_soma_mps_l2 = self._metrics[KEY_HIDDEN_LAYER_INHIB_SOMA_MP_VALUES]
        inhib_soma_acts_l2 = self._metrics[KEY_HIDDEN_LAYER_INHIB_ACT_VALUES]
        basal_minus_soma_mp_l2 = self._metrics[KEY_L2_BASAL_MINUS_SOMA_PYR_MP]
        apical_minus_soma_mp_l2 = self._metrics[KEY_L2_APICAL_MINUS_SOMA_PYR_MP]

        return [
            Graph(type=GraphType.LINE,
                  title="Layer 1 Apical MPs",
                  caption="Layer 1 holds the input layer neurons. This graph shows their apical potentials.",
                  precision=2,
                  series=[
                      Serie("Apical MP 0", data_l1[0].tolist()),
                      Serie("Apical MP 1", data_l1[1].tolist()),
                  ],
                  xaxis="Training steps",
                  yaxis="Membrane potential (mV)"),
            Graph(type=GraphType.LINE,
                  title="Layer 2 Apical MPs",
                  caption="Layer 2 holds the hidden layer neurons. Compare this nudge response to the basic nudge experiment w/o feedback. The nudge occurs at timestep 400.",
                  precision=2,
                  series=[
                      Serie("Apical MP 0", data_l2[0].tolist()),
                      Serie("Apical MP 1", data_l2[1].tolist()),
                      Serie("Apical MP 2", data_l2[2].tolist()),
                  ],
                  xaxis="Training steps",
                  yaxis="Membrane potential (mV)"),
            Graph(type=GraphType.LINE,
                  title="LR W_PP_FF(3[0],2[0]) Triggers in L3",
                  caption="Trigger of the learning rule for the FF wt connecting pyr neuron 0 in layer 2 to pyr 0 in layer 3. The learning trigger is the difference (yellow) btw the somatic (blue) and basal (green) mps of pyr 0 in layer 3. The nudge at timestep 400 causes the somatic mp to jump. Yellow estimates the amount of wt change per timestep.",
                  precision=2,
                  series=[
                      #Serie("Soma act", triggers_l2[0].tolist()),
                      #Serie("Basal hat act", triggers_l2[1].tolist()),
                      Serie("L3 post soma MP", triggers_l2[2].tolist()),
                      Serie("L3 post basal MP", triggers_l2[3].tolist()),
                      Serie("L3 post val", triggers_l2[4].tolist()),
                  ],
                  # xaxis="Training steps",
                  yaxis="..."),
            Graph(type=GraphType.LINE,  # This and the next graph interferred w/ each other when series strings were the same.
                  title="PP_FF wts projecting to L3 Pyr0",
                  caption="Above shows the three weights projecting to pyr neuron 0 in layer 3 from the three pyr neurons in layer 2. The blue graph shows the wt adjustment calculated by the learning rule in the previous panel.",
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
                  caption="Above shows the three weights projecting to pyr neuron 1 in layer 3 from the three pyr neurons in layer 2.",
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
                  caption="Shows lateral wt values projecting to Pyr0 in the hidden layer. Each of 3 inhib neurons in layer 2 project to Pyr0",
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
                  title="LR W_PP_FF(2[0],1[0]) Triggers L2",
                  caption="Trigger of the learning rule for the FF wt connecting pyr neuron 0 in layer 1 to pyr 0 in layer 2.",
                  precision=2,
                  series=[
                      #Serie("Soma act", triggers_l1[0].tolist()),
                      #Serie("Basal hat act", triggers_l1[1].tolist()),
                      Serie("L2 post soma MP", triggers_l1_0_to_L2_0[2].tolist()),
                      Serie("L2 post basal MP", triggers_l1_0_to_L2_0[3].tolist()),
                      Serie("L2 post val", triggers_l1_0_to_L2_0[4].tolist()),
                  ],
                  xaxis="Training steps",
                  yaxis="..."),
            Graph(type=GraphType.LINE,
                  title="PP_FF wts projecting to L2 Pyr0",
                  caption="Both plots above correspond to previous panel b/c they share the same trigger.",
                  precision=2,
                  series=[
                      Serie("PP_FF wt from L1_pyr0 to L2_pyr0", wts_l1_to_l1_pyr0[0].tolist()),
                      Serie("PP_FF wt from L1_pyr1 to L2_pyr0", wts_l1_to_l1_pyr0[1].tolist()),
                  ],
                  xaxis="Training steps",
                  yaxis="..."),
            Graph(type=GraphType.LINE,
                  title="LR W_PP_FF(2[1],1[0]) Triggers L2",
                  caption="Trigger of the learning rule for the FF wt connecting pyr neuron 0 in layer 1 to pyr 1 in layer 2. Why is nudged soma mp transient. Blue above is consistent w/ green in Panel 15. Why is nudge soma transient here but persistent in Panel 3.",
                  precision=2,
                  series=[
                      #Serie("Soma act", triggers_l1[0].tolist()),
                      #Serie("Basal hat act", triggers_l1[1].tolist()),
                      Serie("L2 post soma MP", triggers_l1_0_to_L2_1[2].tolist()),
                      Serie("L2 post basal MP", triggers_l1_0_to_L2_1[3].tolist()),
                      Serie("L2 post val", triggers_l1_0_to_L2_1[4].tolist()),
                  ],
                  xaxis="Training steps",
                  yaxis="..."),
            Graph(type=GraphType.LINE,
                  title="PP_FF wts projecting to L2 Pyr1",
                  caption="Both plots above correspond to previous the panel (11) b/c they share the same trigger.",
                  precision=2,
                  series=[
                      Serie("PP_FF wt from L1_pyr0 to L2_pyr1", wts_l1_to_l1_pyr1[0].tolist()),
                      Serie("PP_FF wt from L1_pyr1 to L2_pyr1", wts_l1_to_l1_pyr1[1].tolist()),
                  ],
                  xaxis="Training steps",
                  yaxis="..."),
            Graph(type=GraphType.LINE,
                  title="Layer 3 Pyr Soma Activations",
                  caption="The output nudges lower the activation level if the wt vals are too high. Use beta=0.1 to keep the initial wts low.",
                  precision=2,
                  series=[
                      Serie("Act Pyr 0", soma_acts_l3[0].tolist()),
                      Serie("Act Pyr 1", soma_acts_l3[1].tolist()),
                  ],
                  xaxis="Training steps",
                  yaxis="Output activation"),
            Graph(type=GraphType.LINE,
                  caption="Since soma mp of pyr0 is > 1, a nudge of 1 causes a decrease. Continued decrease results from gradual decrease of basal mps (see next Panel).",
                  title="Layer 3 Pyr Soma MPs",
                  precision=3,
                  series=[
                      Serie("Mem Pot Pyr 0", soma_mps_l3[0].tolist()),
                      Serie("Mem Pot Pyr 1", soma_mps_l3[1].tolist()),
                  ],
                  xaxis="Training steps",
                  yaxis="Soma MP"),
            Graph(type=GraphType.LINE,
                  title="Layer 3 Pyr Basal MPs",
                  caption="Basal mp decrease after nudge is not caused by direct somatic FB.",
                  precision=3,
                  series=[
                      Serie("Mem Pot Pyr 0", basal_mps_l3[0].tolist()),
                      Serie("Mem Pot Pyr 1", basal_mps_l3[1].tolist()),
                  ],
                  xaxis="Training steps",
                  yaxis="Basal MP"),
            Graph(type=GraphType.LINE,
                  title="Layer 2 Pyr Soma Activations",
                  caption="The nudge response is puzzling in the above graph b/c it is transient and the nudge is persistent. Activation of 0.5 indicates mp is zero.",
                  precision=2,
                  series=[
                      Serie("Act Pyr 0", soma_acts_l2[0].tolist()),
                      Serie("Act Pyr 1", soma_acts_l2[1].tolist()),
                      Serie("Act Pyr 2", soma_acts_l2[2].tolist()),
                  ],
                  xaxis="Training steps",
                  yaxis="Pyr Hidden activation"),
            Graph(type=GraphType.LINE,
                  title="L2 Basal MP minus Soma MP in Pyr",
                  caption="Somatic membrane potential subtracted from basal membrane potential for pyramidal neurons in layer 2. One of two factors to compute change in somatic potential. To trigger Rule 13 learning in Layer 1.",
                  precision=2,
                  series=[
                      Serie("Basal-soma mp Pyr 0", basal_minus_soma_mp_l2[0].tolist()),
                      Serie("Basal-soma mp Pyr 1", basal_minus_soma_mp_l2[1].tolist()),
                      Serie("Basal-soma mp Pyr 2", basal_minus_soma_mp_l2[2].tolist()),
                  ],
                  xaxis="Training steps",
                  yaxis="L2 Pyr basal - soma mp"),
            Graph(type=GraphType.LINE,
                  title="L2 Apical MP minus Soma MP in Pyr",
                  caption="Somatic membrane potential subtracted from apical membrane potential for pyramidal neurons in layer 2. 2nd factor to compute change in somatic potential. Learning of Rule 13 for Layer 1 ends when nudge response ends.",
                  precision=2,
                  series=[
                      Serie("Apical-soma mp Pyr 0", apical_minus_soma_mp_l2[0].tolist()),
                      Serie("Apical-soma mp Pyr 1", apical_minus_soma_mp_l2[1].tolist()),
                      Serie("Apical-soma mp Pyr 2", apical_minus_soma_mp_l2[2].tolist()),
                  ],
                  xaxis="Training steps",
                  yaxis="L2 Pyr apical - soma mp"),
            Graph(type=GraphType.LINE,
                  title="Layer 2 Pyr Apical FB input Values",
                  caption="Wtd sum of FB from apical activations in Layer 3. Response to nudge is persistent.",
                  precision=2,
                  series=[
                      Serie("Apical FB Pyr 0", apical_fb_l2[0].tolist()),
                      Serie("Apical FB Pyr 1", apical_fb_l2[1].tolist()),
                      Serie("Apical FB Pyr 2", apical_fb_l2[2].tolist()),
                  ],
                  xaxis="Training steps",
                  yaxis="Pyr Hidden apical FB values"),
            Graph(type=GraphType.LINE,
                  title="Layer 2 Pyr Apical LAT input Values",
                  caption="Wtd sum of of lateral input to pyrs from inhib activations in Layer 2. Reponse to nudge is persistent.",
                  precision=2,
                  series=[
                      Serie("Apical LAT Pyr 0", apical_lat_l2[0].tolist()),
                      Serie("Apical LAT Pyr 1", apical_lat_l2[1].tolist()),
                      Serie("Apical LAT Pyr 2", apical_lat_l2[2].tolist()),
                  ],
                  xaxis="Training steps",
                  yaxis="Pyr Hidden apical LAT values"),
            Graph(type=GraphType.LINE,
                  title="Layer 2 wtd input to inhib from nudge",
                  caption="Layer 3 pyr soma activation multiplied by nudge weight. This contributes to dendritic mp of inhib neuron in layer 2. Connections from layer 3 to layer 2 are pt-2-pt. The nudge wt for soma 2 is zero.",
                  precision=2,
                  series=[
                      Serie("wtd nudge Inhib 0", inhib_wtd_inp_nudge[0].tolist()),
                      Serie("wtd nudge Inhib 1", inhib_wtd_inp_nudge[1].tolist()),
                      Serie("wtd nudge Inhib 2", inhib_wtd_inp_nudge[2].tolist()),
                  ],
                  xaxis="Training steps",
                  yaxis="Inhib wtd nudge input"),
            Graph(type=GraphType.LINE,
                  title="Layer 2 Inhib Dendritic MPs",
                  caption="Layer 2 inhib dendritic mps influenced by nudge feedback in previous panel.",
                  precision=2,
                  series=[
                      Serie("Dendr MP Inhib 0", inhib_dendr_mps_l2[0].tolist()),
                      Serie("Dendr MP Inhib 1", inhib_dendr_mps_l2[1].tolist()),
                      Serie("Dendr MP Inhib 2", inhib_dendr_mps_l2[2].tolist()),
                  ],
                  xaxis="Training steps",
                  yaxis="Inhib Hidden Dendritic MP"),
            Graph(type=GraphType.LINE,
                  title="Layer 2 Inhib Soma MPs",
                  caption="Layer 2 inhib somatic mps corresponding to previous panel. Current flows from dendrite to soma.",
                  precision=2,
                  series=[
                      Serie("Soma MP Inhib 0", inhib_soma_mps_l2[0].tolist()),
                      Serie("Soma MP Inhib 1", inhib_soma_mps_l2[1].tolist()),
                      Serie("Soma MP Inhib 2", inhib_soma_mps_l2[2].tolist()),
                  ],
                  xaxis="Training steps",
                  yaxis="Inhib Hidden Soma MP"),
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
                  caption="Activations of the two output neurons in response to nudging events. Compare w/ Panel 13. Although the relative activations look good at the end, the activation of pyr1 has decreased.",
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
                  categories=["Before Nudge","Nudged", "After learning", "Nudged removed"],
                  yaxis="Activation level")
        ]

    def run(self):
        self._run_init()

        self._logger.info(f"Starting training {self._self_prediction_steps} steps to 1b2b self predictive.")
        self._run_self_predict()
        self._logger.info(f"Total training steps completed so far: {self._training_steps_completed}.")

        self._logger.info(f"Starting training {self._training_steps} steps for Nudge exp FB.")
        self._run_train()
        self._logger.info(f"Finished training {self._training_steps} steps for Nudge exp FB.")
        self._logger.info(f"Total training steps completed so far: {self._training_steps_completed}.")
        self.print_pyr_activations_all_layers_topdown()  # print activations while nudging is still on

        self._run_after_training()
        self._logger.info(f"Finished training {self._after_training_steps} steps for p_exp 3b")
        self._logger.info(f"Total training steps completed so far: {self._training_steps_completed}.")

        self._logger.info("Final activations after nudging is removed")
        self.print_pyr_activations_all_layers_topdown()  # shows the true effect of learning
        self._logger.info("FINISH: Performing nudge experiment with rules 16b and 13.")
