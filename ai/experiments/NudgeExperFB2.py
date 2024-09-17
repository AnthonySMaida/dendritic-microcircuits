from ai.experiments import NudgeExperFB


class NudgeExperFB2(NudgeExperFB):
    def _run_train(self):
        self.train(self._training_steps, use_nudge=True, use_rule_pi=False)
