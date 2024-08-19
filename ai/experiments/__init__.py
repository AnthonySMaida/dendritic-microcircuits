from .KEYS import KEYS
from .Experiment import Experiment
from .NudgeExperiment import NudgeExperiment
from .XorExperiment import XorExperiment

EXPERIMENTS: dict[str, dict[str, str | Experiment]] = {
    KEYS.NUDGE_EXPERIMENT: {
        "title": "Nudge Experiment",
        "class": NudgeExperiment,
    },
    KEYS.XOR_EXPERIMENT: {
        "title": "XOR Experiment",
        "class": XorExperiment,
    },
}
