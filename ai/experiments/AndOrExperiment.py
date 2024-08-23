from typing import List, Optional

import numpy as np

from ai.colorized_logger import get_logger
from ai.experiments.Experiment import Experiment
from ai.utils import create_column_vector
from metrics import Graph, GraphType, Serie

logger = get_logger('ai.experiments.AndOrExperiment')

KEY_LAYER_1 = "layer1"
KEY_LAYER_2 = "layer2"


class AndOrExperiment(Experiment):
    def __init__(self, wt_init_seed: int, label_init_seed: int, beta: float, learning_rate: float):
        super().__init__(wt_init_seed, beta, learning_rate)

        self._metrics[KEY_LAYER_1] = [np.empty(shape=(2, 0)) for _ in range(4)]
        self._metrics[KEY_LAYER_2] = [np.empty(shape=(2, 0)) for _ in range(4)]

        self._X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self._Y = np.array([[0, 0], [0, 1], [0, 1], [1, 1]])

        self._current_step = 0
        self._nudge_steps = [0] * 4
        self._current_X_index: Optional[int] = None
        self._current_X: Optional[np.ndarray] = None
        self._current_label: Optional[np.ndarray] = None

        self._rng_labels = np.random.default_rng(seed=label_init_seed)

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
        if self._current_step == 400:
            for i in range(4):
                self._nudge_steps[i] = len(self._metrics[KEY_LAYER_1][i][0])
        self._current_X_index = index = self._rng_labels.integers(low=0, high=len(self._X))
        self._current_X = self._X[index]
        self._current_label = self._Y[index]

    def _hook_post_train_step(self):
        # Only record data if current index is 0
        l1, l2 = self.layers

        self._metrics[KEY_LAYER_1][self._current_X_index] = np.append(
            self._metrics[KEY_LAYER_1][self._current_X_index],
            create_column_vector(*map(lambda p: p.apical_mp, l1.pyrs)),
            axis=1
        )
        self._metrics[KEY_LAYER_2][self._current_X_index] = np.append(
            self._metrics[KEY_LAYER_2][self._current_X_index],
            create_column_vector(*map(lambda p: p.apical_mp, l2.pyrs)),
            axis=1
        )

    def _train_1_step(self, nudge_predicate: bool):
        l1, l2 = self.layers
        l1.adjust_wts_lat_pi()
        l2.adjust_wts_pp_ff(l1)

        self.__do_ff_sweep()
        if nudge_predicate:
            l2.nudge_output_layer_neurons(*self._current_label, lambda_nudge=.8)
        self.__do_fb_sweep()

    def __extract_layer_metrics(self, key: str) -> List[Graph]:
        data = self._metrics[key]

        return [Graph(type=GraphType.LINE,
                      title=f"{key} Apical MPs (input {i}: {self._X[i]})",
                      precision=2,
                      series=[
                          Serie("Apical MP 1", data[i][0].tolist()),
                          Serie("Apical MP 2", data[i][1].tolist()),
                      ],
                      xaxis="Training steps",
                      yaxis="Membrane potential (mV)",
                      extra={
                          "annotations": {
                              "xaxis": [
                                  {"x": self._nudge_steps[i], "label": {"text": "nudged"}}
                              ]
                          }
                      })
                for i in range(4)]

    def extract_metrics(self) -> List[Graph]:
        return self.__extract_layer_metrics(KEY_LAYER_1) + self.__extract_layer_metrics(KEY_LAYER_2)

    def run(self, self_prediction_steps: int, training_steps: int):
        # self.__do_ff_sweep()
        # self.__do_fb_sweep()

        self.train(self_prediction_steps, nudge_predicate=False)

        self.__nudge_output_layer()

        self.train(training_steps, nudge_predicate=True)
