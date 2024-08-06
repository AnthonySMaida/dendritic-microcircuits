#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 22:01:20 2024. Last revised 24 July 2024.

@author: anthony s maida

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

from ai.config import nudge1, nudge2, wt_init_seed
from ai.experiments import PilotExp1bConcat2b

logger = logging.getLogger('ai.sacramento_main')
logger.setLevel(logging.INFO)


def main():
    """Do an experience"""
    experiment = PilotExp1bConcat2b()
    experiment.build_small_three_layer_network()

    # p_Exp 3: steps_to_self_pred = 150.
    # p_Exp 3b: steps_to_self_pred = 250.
    experiment.run(steps_to_self_pred=400)

    experiment.print_ff_and_fb_wts_last_layer()

    logger.info("nudge1 = %s; nudge2 = %s", nudge1, nudge2)
    logger.info("wt_init_seed = %d", wt_init_seed)

    return *experiment.datasets[:2], experiment.rule13_post_data[1:].tolist()


if __name__ == '__main__':
    main()
