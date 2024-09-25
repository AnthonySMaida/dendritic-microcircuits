"""
Created on Thu Jul  4 22:01:20 2024. Last revised 24 July 2024.

@author: Anthony S. Maida, Rémi Marseault

Implementation of a small version of model described in:
"Dendritic cortical microcircuits approximate the backpropagation algorithm,"
by Joao Sacramento, Rui Ponte Costa, Yoshua Bengio, and Walter Senn.
In *NeurIPS 2018*, Montreal, Canada.

This particular program studies the learning rules 
presented in that paper in a simplified context.

3 Layer Architecture
Layer 1: input  (2 pyramids, 1 interneuron)
Layer 2: hidden (3 pyramids, 3 interneurons)
Layer 3: output (2 pyramids)

pyramidal neurons are called "pyrs"
interneurons are called 'inhibs'

Implemented in numpy. The code is not vectorized but the
data structures used closely mimic the neural anatomy given in the paper.
"""

import logging
from typing import List

from werkzeug.datastructures import MultiDict

from ai.colorized_logger import ColoredFormatter
from ai.experiments import EXPERIMENTS, ExperimentMetaData, KEYS
from metrics import Graph, GraphType

logger = logging.getLogger("ai")
handler = logging.StreamHandler()
formatter = ColoredFormatter("[%(levelname)s] %(asctime)s: %(name)s :: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


def run_experiment(exp: ExperimentMetaData, params: MultiDict = None) -> List[Graph]:
    if params is None:
        params = MultiDict()

    experiment = exp.class_(params)
    experiment.build_network()
    experiment.run()

    metrics = experiment.extract_metrics()

    i = 1
    for metric in metrics:
        if metric.type == GraphType.EMPTY:
            continue
        metric.title = f"{i}: {metric.title}"
        i += 1

    return metrics


def main(experiment_name: KEYS, params: MultiDict) -> List[Graph]:
    exp = EXPERIMENTS.get(experiment_name)
    if exp is None:
        raise ValueError(f"Unknown experiment: {experiment_name}")

    return run_experiment(exp, params)
