from ai.experiments import NudgeExperFB

class NudgeExperFB3(NudgeExperFB):
    def _run_train(self):
        self.train(self._training_steps, use_nudge=True, use_rule_pi=False)

    def _run_after_training(self):
        self.train(self._after_training_steps, use_nudge=False, use_rule_pi=False)
