import logging

import numpy as np

from ai.experiments.Experiment import Experiment
from metrics import Serie

logger = logging.getLogger('ai.experiments.PilotExp1bConcat2b')
logger.setLevel(logging.INFO)

KEY_LAYER_1 = "layer1"
KEY_LAYER_2 = "layer2"
KEY_RULE_13_POST_DATA = "rule13_post_data"
KEY_RULE_13_WT_DATA = "rule13_wt_data"


class PilotExp1bConcat2b(Experiment):
    def __init__(self):
        super().__init__()  # We call the constructor of the parent class
        self._metrics[KEY_LAYER_1] = np.empty(shape=(0, 2))  # empty matrix for 2-sized vectors
        self._metrics[KEY_LAYER_2] = np.empty(shape=(0, 3))  # empty matrix for 3-sized vectors
        self._metrics[KEY_RULE_13_POST_DATA] = np.empty(shape=(0, 5))  # empty matrix for 5-sized vectors
        self._metrics[KEY_RULE_13_WT_DATA] = np.empty(shape=(0,))  # empty vector

    def extract_metrics(self):
        data1 = np.transpose(self._metrics[KEY_LAYER_1]).tolist()
        data2 = np.transpose(self._metrics[KEY_LAYER_2]).tolist()
        data3 = np.transpose(self._metrics[KEY_RULE_13_POST_DATA]).tolist()
        data4 = np.transpose(self._metrics[KEY_RULE_13_WT_DATA]).tolist()
        return {
            "Layer 1 Apical MPs": {
                "precision": 2,
                "series": [
                    Serie("Apical MP 1", data1[0]),
                    Serie("Apical MP 2", data1[1]),
                ],
                "xaxis": "Training steps",
                "yaxis": "Membrane potential (mV)"
            },
            "Layer 2 Apical MPs": {
                "precision": 2,
                "series": [
                    Serie("Apical MP 1", data2[0]),
                    Serie("Apical MP 2", data2[1]),
                    Serie("Apical MP 3", data2[2]),
                ],
                "xaxis": "Training steps",
                "yaxis": "Membrane potential (mV)"
            },
            "Learning Rule PP_FF Triggers": {
                "precision": 2,
                "series": [
                    Serie("Soma act", data3[0]),
                    Serie("Basal act", data3[1]),
                    Serie("Post value", data3[2]),
                    Serie("Soma mp", data3[3]),
                    Serie("Basal mp", data3[4]),
                ],
                "xaxis": "Training steps",
                "yaxis": "..."
            },
            "Learning Rule PP_FF wts": {
                "precision": 2,
                "series": [
                    Serie("Weight value", data4),
                ],
                "xaxis": "Training steps",
                "yaxis": "..."
            }
        }

    def hook_post_train_step(self):
        l1, l2, l3 = self.layers
        self._metrics[KEY_LAYER_1] = np.concatenate((
            self._metrics[KEY_LAYER_1],
            np.array([[l1.pyrs[0].apical_mp, l1.pyrs[1].apical_mp]])
        ), axis=0)
        self._metrics[KEY_LAYER_2] = np.concatenate((
            self._metrics[KEY_LAYER_2],
            np.array([[l2.pyrs[0].apical_mp, l2.pyrs[1].apical_mp, l2.pyrs[2].apical_mp]])
        ), axis=0)

        post_soma_mp = l3.pyr_soma_mps()
        post_basal_mp = l3.pyr_basal_mps()
        post = post_soma_mp - post_basal_mp
        trigger_data_pt = np.array([[l3.pyr_soma_acts()[0],
                                     l3.pyr_basal_hat_acts()[0],
                                     post[0],
                                     post_soma_mp[0],
                                     post_basal_mp[0]]])
        wt_data_pt = np.array([l3.pyrs[0].W_PP_ff[0]])

        # save data point: [soma_act, basal_hat_act, post_val, soma_mp, basal_mp]
        self._metrics["rule13_post_data"] = np.concatenate((self._metrics["rule13_post_data"], trigger_data_pt), axis=0)
        # save data point: [FF_wt_value]
        self._metrics["rule13_wt_data"] = np.concatenate((self._metrics["rule13_wt_data"], wt_data_pt), axis=0)

    def train_1_step(self, nudge_predicate: bool):  # Signature matched its abstract method b/c *args can be empty.
        self.train_1_step_rule_16b_and_rule_13(nudge_predicate=nudge_predicate)  # defined in superclass

    def run(self, steps_to_self_pred=150):
        self.do_ff_sweep()  # prints state
        logger.info("Finished 1st FF sweep: pilot_exp_1")

        self.do_fb_sweep()  # prints state
        logger.info("Finished 1st FB sweep: pilot_exp_1")

        n_of_training_steps_to_self_predictive = steps_to_self_pred
        logger.info(f"Starting training {n_of_training_steps_to_self_predictive} steps to 1b self predictive.")

        # trains and SAVES apical results in 'datasets' attr
        self.train_and_save_apical_data(n_of_training_steps_to_self_predictive, nudge_predicate=False)

        self.nudge_output_layer()

        n_of_training_steps = 200
        logger.info(f"Starting training {n_of_training_steps} steps for p_exp 3b")

        # trains and APPENDS apical results in 'datasets' attr
        self.train_and_save_apical_data(n_of_training_steps, nudge_predicate=True)

        logger.info(f"Finished training {n_of_training_steps} steps for p_exp 3b")
        self.print_pyr_activations_all_layers_topdown()  # print activations while nudging is still on

        self.do_ff_sweep()  # to get new activations without nudging

        logger.info("Final activations after nudging is removed")
        self.print_pyr_activations_all_layers_topdown()  # shows the true effect of learning
