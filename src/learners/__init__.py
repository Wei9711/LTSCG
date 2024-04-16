from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .sopcg_learner import SopcgLearner
from .dcg_learner import DCGLearner
from .ppo_learner import PPOLearner


from .LTSCG_inf_learner import LTSCG_InferLearner
from .LTSCG_pre_learner import LTSCG_PredictLearner
from .LTSCG_learner import LTSCGLearner
REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["sopcg_learner"] = SopcgLearner
REGISTRY["dcg_learner"] = DCGLearner
REGISTRY["ppo_learner"] = PPOLearner


REGISTRY["LTSCG_infer_learner"] = LTSCG_InferLearner
REGISTRY["LTSCG_prefict_learner"] = LTSCG_PredictLearner
REGISTRY["LTSCG_learner"] = LTSCGLearner