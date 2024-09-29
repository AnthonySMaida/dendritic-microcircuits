from ai.experiments import NudgeExperFB
from ai.utils import iter_with_prev


class NudgeExperFB4(NudgeExperFB):
    def _run_train(self):
        self.train(self._training_steps, use_nudge=True, use_rule_pi=False)

    def _run_after_training(self):
        self.train(self._after_training_steps, use_nudge=False, use_rule_pi=False)

    def __do_fb_sweep(self, use_nudge=False):  # this version assumes interneurons have nudging feedback connections
        """Standard FB sweep"""
        nudge_fb_weight = self.__nudge_fb_weight
        for prev, layer in iter_with_prev(reversed(self.layers)):  # [l3, l2, l1]
            if prev is None:  # Skip first layer (L3)
                continue      # Go to the next iteration.
            if layer.id_num == 2:  # handles the nudging FB connections to Layer 2. Only sends FB to 2 inhibs.
                if use_nudge:
                    layer.inhibs[0].wtd_input_from_nudge = nudge_fb_weight * prev.pyrs[0].soma_act  # for data collection
                    # Eqn below is not exactly right. I looked at Fig 5 and forgot to look at Eqns 10 & 23.
                    layer.inhibs[0].dend_mp += nudge_fb_weight * prev.pyrs[0].soma_act
                    layer.inhibs[0].update_inhib_soma_ff()

                    layer.inhibs[1].wtd_input_from_nudge = nudge_fb_weight * prev.pyrs[1].soma_act  # for data collection
                    layer.inhibs[1].dend_mp += nudge_fb_weight * prev.pyrs[1].soma_act
                    layer.inhibs[1].update_inhib_soma_ff()
                else:
                    layer.inhibs[0].wtd_input_from_nudge = 0.0
                    layer.inhibs[1].wtd_input_from_nudge = 0.0
            # update current layer pyrs using somatic pyr acts from previous layer and inhib acts from current layer
            # in 2nd iter, prev = l3 and layer = l2
            layer.update_pyrs_apical_soma_fb(
                prev,
                disable_apical=self._training_steps_completed >= self._self_prediction_steps + self._training_steps
            )
