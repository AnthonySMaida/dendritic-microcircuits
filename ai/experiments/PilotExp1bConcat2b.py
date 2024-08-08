import logging

from ai.experiments.Experiment import Experiment

logger = logging.getLogger('ai.experiments.PilotExp1bConcat2b')
logger.setLevel(logging.INFO)


class PilotExp1bConcat2b(Experiment):
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
