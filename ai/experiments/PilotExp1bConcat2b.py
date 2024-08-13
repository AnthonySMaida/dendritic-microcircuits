import numpy as np

from ai.experiments.Experiment import Experiment
from ai.colorized_logger import get_logger
from ai.utils import create_column_vector
from metrics import Graph, Serie, GraphType

logger = get_logger('ai.experiments.PilotExp1bConcat2b')
#logger.setLevel(logging.DEBUG)

KEY_LAYER_1 = "layer1"
KEY_LAYER_2 = "layer2"
KEY_RULE_13_POST_DATA = "rule13_post_data"
KEY_RULE_13_WT_DATA = "rule13_wt_data"
KEY_RULE_13_POST_DATA_L1 = "rule13_post_data_l1"
KEY_RULE_13_WT_DATA_L1 = "rule13_wt_data_l1"


class PilotExp1bConcat2b(Experiment):
    def __init__(self, wt_init_seed: int, beta: float, learning_rate: float, nudge1: float, nudge2: float):
        # call the constructor of the parent class (aka superclass)
        super().__init__(wt_init_seed, beta, learning_rate, nudge1, nudge2)

        self._metrics[KEY_LAYER_1] = np.empty(shape=(2, 0))
        self._metrics[KEY_LAYER_2] = np.empty(shape=(3, 0))
        self._metrics[KEY_RULE_13_POST_DATA] = np.empty(shape=(5, 0))
        self._metrics[KEY_RULE_13_WT_DATA] = np.empty(shape=(0,))
        self._metrics[KEY_RULE_13_POST_DATA_L1] = np.empty(shape=(5,0))
        self._metrics[KEY_RULE_13_WT_DATA_L1] = np.empty(shape=(0,))

    def extract_metrics(self):
        data1 = self._metrics[KEY_LAYER_1]
        data2 = self._metrics[KEY_LAYER_2]
        data3 = self._metrics[KEY_RULE_13_POST_DATA]
        data4 = self._metrics[KEY_RULE_13_WT_DATA]
        data5 = self._metrics[KEY_RULE_13_POST_DATA_L1]
        data6 = self._metrics[KEY_RULE_13_WT_DATA_L1]
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
            Graph(title="Learning Rule PP_FF Triggers L1",
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
            Graph(title="Learning Rule PP_FF wts L1",
                  precision=2,
                  series=[
                      Serie("PP_FF wt L1", data6.tolist()),
                  ],
                  xaxis="Training steps",
                  yaxis="..."),
            Graph(type=GraphType.COLUMN,
                  title="test",
                  precision=2,
                  series=[
                      Serie("serie1", [1, 2, 3]),
                        Serie("serie2", [4, 5, 6]),
                  ],
                  categories=["cat1", "cat2", "cat3"],
                  yaxis="...")
        ]

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

        soma_act = l3.pyr_soma_acts()[0]
        basal_hat_act = l3.pyr_basal_hat_acts()[0]
        post_soma_mp = l3.pyr_soma_mps()[0]
        post_basal_mp = l3.pyr_basal_mps()[0]
        post_val = post_soma_mp - post_basal_mp

        self._metrics[KEY_RULE_13_POST_DATA] = np.append(
            self._metrics[KEY_RULE_13_POST_DATA],
            create_column_vector(soma_act, basal_hat_act, post_soma_mp, post_basal_mp, post_val),
            axis=1)
        self._metrics[KEY_RULE_13_WT_DATA] = np.append(self._metrics[KEY_RULE_13_WT_DATA],
                                                       l3.pyrs[0].W_PP_ff[0])
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


    def train_1_step(self, nudge_predicate: bool):  # Signature matched its abstract method b/c *args can be empty.
        """
        This is the concrete version of the abstract train-1-step defined in superclass Experiment.
        :param nudge_predicate:
        :return:
        """
        self.train_1_step_rule_16b_and_rule_13(use_nudge=nudge_predicate)  # defined in superclass

    def run(self, self_prediction_steps: int, training_steps: int):
        logger.info("START: Performing nudge experiment with rules 16b and 13.")
        self.do_ff_sweep()  # prints state
        logger.info("Finished 1st FF sweep: pilot_exp_1b_concat_2b")

        self.do_fb_sweep()  # prints state
        logger.info("Finished 1st FB sweep: pilot_exp_1b_concat_2b")

        logger.info(f"Starting training {self_prediction_steps} steps to 1b2b self predictive.")

        # trains and SAVES apical results in 'datasets' attr
        self.train_and_save_apical_data(self_prediction_steps, nudge_predicate=False)

        logger.info("Calling function to impose nudge.")
        self.nudge_output_layer()

        logger.info(f"Starting training {training_steps} steps for p_exp 3b")

        # trains and APPENDS apical results in 'datasets' attr
        self.train_and_save_apical_data(training_steps, nudge_predicate=True)

        logger.info(f"Finished training {training_steps} steps for p_exp 3b")
        self.print_pyr_activations_all_layers_topdown()  # print activations while nudging is still on

        self.do_ff_sweep()  # to get new activations without nudging

        logger.info("Final activations after nudging is removed")
        self.print_pyr_activations_all_layers_topdown()  # shows the true effect of learning
        logger.info("FINISH: Performing nudge experiment with rules 16b and 13.")
