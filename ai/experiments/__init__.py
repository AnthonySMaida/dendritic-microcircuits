from dataclasses import dataclass

from .AndOrExperiment import AndOrExperiment
from .Experiment import Experiment
from .KEYS import KEYS
from .BasicNudgeExper import BasicNudgeExper
from .NudgeExperFB import NudgeExperFB
from .XorExperiment import XorExperiment


@dataclass
class ExperimentMetaData:
    title: str
    class_: type[Experiment]  # what does underscore mean?

EXPERIMENTS: dict[KEYS, ExperimentMetaData] = {
    KEYS.AND_OR_EXPERIMENT: ExperimentMetaData("And/Or Experiment", AndOrExperiment),
    KEYS.NUDGE_EXPERIMENT: ExperimentMetaData("Basic Nudge Experiment", BasicNudgeExper),
    KEYS.NUDGE_EXPERIMENT_FB: ExperimentMetaData("Nudge Experiment with FB", NudgeExperFB),
    KEYS.XOR_EXPERIMENT: ExperimentMetaData("XOR Experiment", XorExperiment),
}