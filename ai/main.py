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
from ai.config import nudge1, nudge2, wt_init_seed
from ai.experiments import PilotExp1bConcat2b
from ai.colorized_logger import get_logger


logger = get_logger('ai.sacramento_main')
#logger.setLevel(logging.DEBUG)


def main():
    """Do an experiment"""
    wt_init_seed = 42
    beta = 1.0 / 3.0  # beta = 1/lambda => lambda = 3. beta is scale param for rng.exponential.
    learning_rate = 0.05
    nudge1 = 1.0
    nudge2 = 0.0
    n_pyr_by_layer = (2, 3, 2)
    self_prediction_steps = 400
    training_steps = 200

    logger.info('Starting sacramento_main')
    experiment = PilotExp1bConcat2b(wt_init_seed, beta, learning_rate, nudge1, nudge2)  # make instance
    experiment.build_small_three_layer_network(*n_pyr_by_layer)
    logger.info('Finished building network')
    logger.info('Starting to run experiment')
    experiment.run(self_prediction_steps, training_steps)
    experiment.print_ff_and_fb_wts_last_layer()

    logger.info("nudge1 = %s; nudge2 = %s", nudge1, nudge2)
    logger.info("wt_init_seed = %d", wt_init_seed)

    return experiment.extract_metrics()


if __name__ == '__main__':
    main()
