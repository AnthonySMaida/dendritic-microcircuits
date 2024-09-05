from typing import List, Optional

import numpy as np
from werkzeug.datastructures import MultiDict

from ai.colorized_logger import get_logger
from ai.experiments.Experiment import Experiment
from ai.utils import create_column_vector
from metrics import Graph, GraphType, Serie

logger = get_logger('ai.experiments.AndOrExperiment')

KEY_WTS_FB = "Weights FB"
KEY_WTS_FF = "Weights FF"
KEY_OUTPUT_ACTIVATIONS_XOR = "Output activations XOR"


class Xor2LayerExperiment(Experiment):
    def __init__(self, params: MultiDict):
        super().__init__(params)

        self.__label_init_seed = params.get('label_init_seed', 42, type=int)
        self.__self_prediction_steps = params.get('self_prediction_steps', 400, type=int)
        self.__training_steps = params.get('training_steps', 190, type=int)
        self.__test_steps = params.get('test_steps', 10, type=int)

        self._X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self._Y = np.array([[0], [1], [1], [0]])

        self._current_step = 0
        self._nudge_steps = [0] * 4
        self._test_steps = [0] * 4
        self._current_X_index: Optional[int] = None
        self._current_X: Optional[np.ndarray] = None
        self._current_label: Optional[np.ndarray] = None

        self._rng_labels = np.random.default_rng(seed=self.__label_init_seed)

        self._metrics[KEY_WTS_FB] = np.empty(shape=(2, 0))
        self._metrics[KEY_WTS_FF] = np.empty(shape=(2, 0))
        self._metrics[KEY_OUTPUT_ACTIVATIONS_XOR] = [np.empty(shape=(0,)) for _ in range(len(self._X))]

    def __do_ff_sweep(self):
        """Standard FF sweep"""
        self.layers[0].apply_inputs_to_test_self_predictive_convergence(self._current_X.tolist())
        self.layers[0].update_dend_mps_via_ip()
        self.layers[1].update_pyrs_basal_and_soma_ff(self.layers[0])
        self.layers[1].update_dend_mps_via_ip()

    def __do_fb_sweep(self):
        """Standard FB sweep"""
        self.layers[0].update_pyrs_apical_soma_fb(self.layers[1])

    def __nudge_output_layer(self):
        self.layers[1].nudge_output_layer_neurons(*self._current_label)
        self.__do_fb_sweep()

    def _hook_pre_train_step(self):
        self._current_step += 1
        if self._current_step == self.__self_prediction_steps:
            for i in range(4):
                self._nudge_steps[i] = len(self._metrics[KEY_OUTPUT_ACTIVATIONS_XOR][i])
        if self._current_step == self.__self_prediction_steps + self.__training_steps:
            for i in range(4):
                self._test_steps[i] = len(self._metrics[KEY_OUTPUT_ACTIVATIONS_XOR][i])
        self._current_X_index = index = self._rng_labels.integers(low=0, high=len(self._X))
        self._current_X = self._X[index]
        self._current_label = self._Y[index]

    def _hook_post_train_step(self):
        l1, l2 = self.layers

        self._metrics[KEY_WTS_FB] = np.append(self._metrics[KEY_WTS_FB],
                                              create_column_vector(l1.pyrs[0].W_PP_fb[0], l1.pyrs[1].W_PP_fb[0]),
                                              axis=1)
        self._metrics[KEY_WTS_FF] = np.append(self._metrics[KEY_WTS_FF],
                                              create_column_vector(l2.pyrs[0].W_PP_ff[0], l2.pyrs[0].W_PP_ff[1]),
                                              axis=1)
        self._metrics[KEY_OUTPUT_ACTIVATIONS_XOR][self._current_X_index] = np.append(
            self._metrics[KEY_OUTPUT_ACTIVATIONS_XOR][self._current_X_index],
            l2.pyr_soma_mps[0]
        )

    def _train_1_step(self, nudge_predicate: bool):
        l1, l2 = self.layers
        l1.adjust_wts_lat_pi()
        l2.adjust_wts_pp_ff(l1)

        self.__do_ff_sweep()
        if nudge_predicate:
            l2.nudge_output_layer_neurons(*self._current_label, lambda_nudge=.2)
        self.__do_fb_sweep()

    def __extract_output_activations_metrics(self, key: str, label_index: int) -> List[Graph]:
        data = self._metrics[key]

        return [Graph(type=GraphType.LINE,
                      title=f"{key}: X={self._X[i]}; Y={self._Y[i][label_index]}",
                      precision=2,
                      series=[
                          Serie("Soma MP", data[i].tolist()),
                      ],
                      xaxis="Training steps",
                      yaxis="Membrane potential (mV)",
                      extra={
                          "annotations": {
                              "xaxis": [
                                  {"x": self._nudge_steps[i], "label": {"text": "Nudge applied"}},
                                  {"x": self._test_steps[i], "label": {"text": "Nudge removed"}}
                              ],
                              "yaxis": [
                                  {"y": .5, "borderColor": "red"}
                              ]
                          }
                      })
                for i in range(len(self._X))]

    def build_network(self, *args, **kwargs):
        self.build_small_two_layer_network(2, 1, 0)

    def extract_metrics(self) -> List[Graph]:
        extra = {
            "annotations": {
                "xaxis": [
                    {"x": self.__self_prediction_steps, "label": {"text": "Nudge applied"}},
                    {"x": self.__self_prediction_steps + self.__training_steps, "label": {"text": "Nudge removed"}}
                ],
                "yaxis": [
                    {"y": .5, "borderColor": "red"}
                ]
            }
        }
        ff_wts = Graph(type=GraphType.LINE,
                       title="Feed-forward weights",
                       caption="These weights are coming from layer 1 to layer 2",
                       precision=2,
                       series=[
                           Serie("FF Wts 1,1 -> 2,1", self._metrics[KEY_WTS_FF][0].tolist()),
                           Serie("FF Wts 1,2 -> 2,1", self._metrics[KEY_WTS_FF][1].tolist())
                       ],
                       xaxis="Training steps",
                       yaxis="Weight value",
                       extra=extra)
        fb_wts = Graph(type=GraphType.LINE,
                       title="Feed-back weights",
                       caption="These weights are coming from layer 2 to layer 1",
                       precision=2,
                       series=[
                           Serie("FB Wts 2,1 -> 1,1", self._metrics[KEY_WTS_FB][0].tolist()),
                           Serie("FB Wts 2,1 -> 1,2", self._metrics[KEY_WTS_FB][1].tolist())
                       ],
                       xaxis="Training steps",
                       yaxis="Weight value",
                       extra=extra)
        output_acts_xor = self.__extract_output_activations_metrics(KEY_OUTPUT_ACTIVATIONS_XOR, 0)

        return [ff_wts, fb_wts] + output_acts_xor

    def run(self):
        self.train(self.__self_prediction_steps, nudge_predicate=False)
        self.__nudge_output_layer()
        self.train(self.__training_steps, nudge_predicate=True)
        self.train(self.__test_steps, nudge_predicate=False)
