from .KEYS import KEYS
from .Experiment import Experiment
from .PilotExp1bConcat2b import PilotExp1bConcat2b
from .XorExperiment import XorExperiment

EXPERIMENTS: dict[str, dict[str, str | Experiment]] = {
    KEYS.PILOT_EXP_1B_CONCAT_2B: {
        "title": "Pilot Experiment 1b Concat 2b",
        "class": PilotExp1bConcat2b,
    },
    KEYS.XOR_EXPERIMENT: {
        "title": "XOR Experiment",
        "class": XorExperiment,
    },
}
