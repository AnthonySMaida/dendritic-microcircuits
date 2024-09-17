from dataclasses import dataclass

from .AndOrExperiment import AndOrExperiment
from .ApicalConvergenceTest import ApicalConvergenceTest
from .BasicNudgeExper import BasicNudgeExper
from .Experiment import Experiment
from .KEYS import KEYS
from .NudgeExperFB import NudgeExperFB
from .NudgeExperFB2 import NudgeExperFB2
from .XorExperiment import XorExperiment


@dataclass
class ExperimentMetaData:
    title: str
    short_description: str
    long_description: str
    class_: type[Experiment]


short = "Lorem ipsum odor amet, consectetuer adipiscing elit."
long = "Lorem ipsum odor amet, consectetuer adipiscing elit. Magna tempus et efficitur ad; justo faucibus dapibus sollicitudin dictumst. Erat non aptent tincidunt nibh praesent nullam tristique?"

EXPERIMENTS: dict[KEYS, ExperimentMetaData] = {
    KEYS.AND_OR_EXPERIMENT: ExperimentMetaData(title="And/Or Experiment",
                                               short_description=short,
                                               long_description=long,
                                               class_=AndOrExperiment),
    KEYS.BASIC_NUDGE_EXPERIMENT: ExperimentMetaData(title="Basic Nudge Experiment",
                                                    short_description=short,
                                                    long_description=long,
                                                    class_=BasicNudgeExper),
    KEYS.NUDGE_EXPERIMENT_FB: ExperimentMetaData(title="Nudge Experiment with FB",
                                                 short_description=short,
                                                 long_description=long,
                                                 class_=NudgeExperFB),
    KEYS.NUDGE_EXPERIMENT_FB_2: ExperimentMetaData(title="Nudge Experiment w/ FB and w/out lat",
                                                   short_description=short,
                                                   long_description=long,
                                                   class_=NudgeExperFB2),
    KEYS.XOR_EXPERIMENT: ExperimentMetaData(title="XOR Experiment",
                                            short_description=short,
                                            long_description=long,
                                            class_=XorExperiment),
    KEYS.APICAL_CONVERGENCE_TEST: ExperimentMetaData(title="Apical Convergence Test",
                                                     short_description=short,
                                                     long_description=long,
                                                     class_=ApicalConvergenceTest),
}
