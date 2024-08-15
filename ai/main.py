#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

from typing import List

from werkzeug.datastructures import MultiDict

from ai.colorized_logger import get_logger
from ai.experiments import KEYS, PilotExp1bConcat2b, XorExperiment
from metrics import Graph

logger = get_logger('ai.sacramento_main')
#logger.setLevel(logging.DEBUG)


def pilot_exp_1b_concat_2b(params: MultiDict = None) -> List[Graph]:  # Why is MultiDict needed?
    """Do an experiment"""
    if params is None:
        params = MultiDict()

    wt_init_seed = params.get('wt_init_seed', 42, type=int)
    beta = params.get('beta', 1.0 / 3.0, type=float)  # beta = 1/lambda => lambda = 3. beta is scale param for rng.exponential.
    learning_rate = params.get('learning_rate', 0.05, type=float)
    nudge1 = params.get('nudge1', 1.0, type=float)
    nudge2 = params.get('nudge2', 0.0, type=float)
    n_pyr_by_layer = (
        params.get('n_pyr_layer1', 2, type=int),
        params.get('n_pyr_layer2', 3, type=int),
        params.get('n_pyr_layer3', 2, type=int),
    )
    self_prediction_steps = params.get('self_prediction_steps', 400, type=int)
    training_steps = params.get('training_steps', 190, type=int)
    after_training_steps = params.get('after_training_steps', 10, type=int)

    logger.info('Starting sacramento_main')
    experiment = PilotExp1bConcat2b(wt_init_seed, beta, learning_rate, nudge1, nudge2)  # make instance
    experiment.build_small_three_layer_network(*n_pyr_by_layer)
    logger.info('Finished building network')
    logger.info('Starting to run experiment')
    experiment.run(self_prediction_steps, training_steps, after_training_steps)
    experiment.print_ff_and_fb_wts_last_layer()

    logger.info("nudge1 = %s; nudge2 = %s", nudge1, nudge2)
    logger.info("wt_init_seed = %d", wt_init_seed)

    return experiment.extract_metrics()

def xor_experiment(params: MultiDict = None) -> List[Graph]:
    if params is None:
        params = MultiDict()

    wt_init_seed = params.get('wt_init_seed', 42, type=int)
    beta = params.get('beta', 1.0 / 3.0, type=float)  # beta = 1/lambda => lambda = 3. beta is scale param for rng.exponential.
    learning_rate = params.get('learning_rate', 0.05, type=float)
    n_pyr_by_layer = (
        params.get('n_pyr_layer1', 2, type=int),
        params.get('n_pyr_layer2', 3, type=int),
        params.get('n_pyr_layer3', 2, type=int),
    )
    self_prediction_steps = params.get('self_prediction_steps', 400, type=int)
    training_steps = params.get('training_steps', 190, type=int)
    after_training_steps = params.get('after_training_steps', 10, type=int)

    experiment = XorExperiment(wt_init_seed, beta, learning_rate)
    experiment.build_small_three_layer_network(*n_pyr_by_layer)
    experiment.run(self_prediction_steps, training_steps, after_training_steps)
    experiment.print_ff_and_fb_wts_last_layer()

    return experiment.extract_metrics()


def main(experiment_name: str, params: MultiDict) -> List[Graph]:
    match experiment_name:
        case KEYS.PILOT_EXP_1B_CONCAT_2B:
            return pilot_exp_1b_concat_2b(params)
        case KEYS.XOR_EXPERIMENT:
            return xor_experiment(params)
        case _:
            raise ValueError(f"Unknown experiment: {experiment_name}")
