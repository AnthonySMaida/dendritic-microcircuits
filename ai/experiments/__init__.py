from .AndOrExperiment import AndOrExperiment
from .Experiment import Experiment
from .KEYS import KEYS
from .BasicNudgeExper import BasicNudgeExper
from .NudgeExperFB import NudgeExperFB
from .XorExperiment import XorExperiment

EXPERIMENTS: dict[str, dict[str, str | Experiment]] = {
    KEYS.AND_OR_EXPERIMENT: {
        "title": "And/Or Experiment",
        "class": AndOrExperiment
    },
    KEYS.NUDGE_EXPERIMENT: {
        "title": "Basic Nudge Experiment",
        "class": BasicNudgeExper,
    },
    KEYS.NUDGE_EXPERIMENT_FB: {
        "title": "Nudge Experiment with FB",
        "class": NudgeExperFB,
    },
    KEYS.XOR_EXPERIMENT: {
        "title": "XOR Experiment",
        "class": XorExperiment,
    },
}
