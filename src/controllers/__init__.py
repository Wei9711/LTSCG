REGISTRY = {}

from .basic_controller import BasicMAC
from .sopcg_controller import SopcgMAC
from .dcg_controller import DeepCoordinationGraphMAC
from .dicg_controller import DICGraphMAC
from .ltscg_controller import LTSCG_GraphMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["sopcg_mac"] = SopcgMAC
REGISTRY["dcg_mac"] = DeepCoordinationGraphMAC
REGISTRY["dicg_mac"] = DICGraphMAC
REGISTRY["Ltscg_mac"] = LTSCG_GraphMAC