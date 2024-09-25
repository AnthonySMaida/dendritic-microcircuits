from dataclasses import dataclass

from .AndOrExperiment import AndOrExperiment
from .ApicalConvergenceTest import ApicalConvergenceTest
from .BasicNudgeExper import BasicNudgeExper
from .Experiment import Experiment
from .KEYS import KEYS
from .NudgeExperFB import NudgeExperFB
from .NudgeExperFB2 import NudgeExperFB2
from .NudgeExperFB3 import NudgeExperFB3
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
                                               short_description=("Network with single layer of weights. Two inputs. Two Outputs. "
                                                                  "One output unit learns 'and' the other 'or'."
                                                                  ),
                                               long_description=("In theory both 'and' and 'or' can be learned using one layer of weights."
                                                                 "This experiment tests if the dendritic model can learn those predicates."
                                                                 ),
                                               class_=AndOrExperiment),
    KEYS.APICAL_CONVERGENCE_TEST: ExperimentMetaData(title="Apical Convergence Test",
                                                     short_description=("The apical convergence to zero happens earlier in the output layer than in the hidden layer when lat PI learning is enabled for both layers. "
                                                                        "This experiment tries to figure out why by turning of lat PI learning in the output layer."
                                                                        ),
                                                     long_description=("Compares convergence of apical membrane potentials to zero for the output "
                                                                       "layer and the hidden layer. The hidden layer in Panel 1 converges more "
                                                                       "slowly than the output layer in Panel 2."
                                                                       ),
                                                     class_=ApicalConvergenceTest),
    KEYS.BASIC_NUDGE_EXPERIMENT: ExperimentMetaData(title="Basic Nudge Experiment",
                                                    short_description="Tests nudging without feedback weights.",
                                                    long_description=long,
                                                    class_=BasicNudgeExper),
    KEYS.NUDGE_EXPERIMENT_FB: ExperimentMetaData(title="Nudge Experiment with FB",
                                                 short_description="This experiment tests the ability to learn a nudge (label) and uses nudge feedback connections.",
                                                 long_description=("This experiment uses point-to-point nudge feedback connections from pyramidal neurons in the output layer to matching inhbitory neurons in the hidden layer. "
                                                                   "The purpose of this experiment is to asses if this nudge feedback improves learning in the first feedforward layer."
                                                                   ),
                                                 class_=NudgeExperFB),
    KEYS.NUDGE_EXPERIMENT_FB_2: ExperimentMetaData(title="Nudge Experiment w/ FB and w/out lat learning for training",
                                                   short_description="This experiment is like 'Nudge Experiment with FB' but turns off lateral inhibitory learning during the training phase.",
                                                   long_description=("Examines the hypothesis that Rule 16b in the hidden layer prevents communication of the nudge signal to the trigger region "
                                                                     "of the FF rule. Rule 16b is turned off during training to test if learning improves."
                                                                     ),
                                                   class_=NudgeExperFB2),
    KEYS.NUDGE_EXPERIMENT_FB_3: ExperimentMetaData(title="Nudge Experiment w/ FB and w/out lat learning for training and after training",
                                                   short_description="This experiment is like 'Nudge Experiment with FB' but turns off lateral inhibitory learning during the training phase and after training phase.",
                                                   long_description=("Examines the hypothesis that Rule 16b in the hidden layer prevents communication of the nudge signal to the trigger region "
                                                                     "of the FF rule. Rule 16b is turned off during training to test if learning improves."
                                                                     ),
                                                   class_=NudgeExperFB3),
    KEYS.XOR_EXPERIMENT: ExperimentMetaData(title="XOR Experiment",
                                            short_description=("The 'XOR' problem is known to require two weight layers."
                                                               "This experiment tests if a two weight-layer model can learn the problem."
                                                               ),
                                            long_description=long,
                                            class_=XorExperiment)
}
